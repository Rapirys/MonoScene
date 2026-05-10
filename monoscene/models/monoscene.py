import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from monoscene.loss.CRP_loss import compute_super_CP_multilabel_loss
from monoscene.loss.sparse_ssc_loss import (
    sparse_ce_ssc_loss,
    sparse_geo_scal_loss,
    sparse_sem_scal_loss,
)
from monoscene.loss.ssc_loss import CE_ssc_loss, KL_sep, geo_scal_loss, sem_scal_loss
from monoscene.loss.sscMetrics import SSCMetrics
from monoscene.models.flosp import FLoSP
from monoscene.models.unet2d import UNet2D
from monoscene.models.unet3d_kitti import UNet3D as UNet3DKitti
from monoscene.models.unet3d_nyu import UNet3D as UNet3DNYU


class MonoScene(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        feature,
        class_weights,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=True,
        fp_loss=True,
        project_res=[],
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        lr=1e-4,
        weight_decay=1e-4,
        use_visible_mask=False,
    ):
        super().__init__()

        self.project_res = project_res
        self.fp_loss = fp_loss
        self.dataset = dataset
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_visible_mask = use_visible_mask

        self.n_classes = n_classes
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        self.projects = self.build_projects(full_scene_size)
        self.net_3d_decoder = self.build_3d_decoder(
            feature,
            full_scene_size,
            n_relations,
            context_prior,
        )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

    def build_projects(self, full_scene_size):
        return nn.ModuleDict()

    def build_3d_decoder(self, feature, full_scene_size, n_relations, context_prior):
        raise NotImplementedError

    def visible_eval_mask(self, batch):
        if not self.use_visible_mask or "observed_mask" not in batch:
            return None
        visible_mask = batch["observed_mask"]
        if len(visible_mask) == 0:
            return None
        return torch.stack(visible_mask).cpu().numpy()

    def log_loss(self, step_type, name, loss):
        self.log(step_type + "/" + name, loss.detach(), on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def on_validation_epoch_end(self):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def on_test_epoch_end(self):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        return [optimizer], [scheduler]


class DenseMonoScene(MonoScene):
    def build_projects(self, full_scene_size):
        projects = {}
        for scale_2d in self.scale_2ds:
            projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        return nn.ModuleDict(projects)

    def build_3d_decoder(self, feature, full_scene_size, n_relations, context_prior):
        decoders = {
            "NYU": lambda: UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            ),
            "kitti": lambda: UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=self.project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            ),
        }
        return decoders[self.dataset]()

    def forward(self, batch):
        img = batch["img"]
        bs = len(img)

        x_rgb = self.net_rgb(img)

        x3ds = []
        for i in range(bs):
            x3d = None
            for scale_2d in self.project_res:
                scale_2d = int(scale_2d)
                projection_key = "projected_pix_{}".format(self.project_scale)
                fov_key = "fov_mask_{}".format(self.project_scale)
                projected_pix = batch[projection_key][i].cuda()
                fov_mask = batch[fov_key][i].cuda()

                projected = self.projects[str(scale_2d)](
                    x_rgb["1_" + str(scale_2d)][i],
                    projected_pix // scale_2d,
                    fov_mask,
                )
                x3d = projected if x3d is None else x3d + projected
            x3ds.append(x3d)

        input_dict = {"x3d": torch.stack(x3ds)}
        return self.net_3d_decoder(input_dict)

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

        if self.context_prior and self.relation_loss:
            loss_rel_ce = compute_super_CP_multilabel_loss(
                out_dict["P_logits"], batch["CP_mega_matrices"]
            )
            loss += loss_rel_ce
            self.log_loss(step_type, "loss_relation_ce_super", loss_rel_ce)

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log_loss(step_type, "loss_ssc", loss_ssc)

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log_loss(step_type, "loss_sem_scal", loss_sem_scal)

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log_loss(step_type, "loss_geo_scal", loss_geo_scal)

        if self.fp_loss and step_type != "test":
            frustum_loss = self.frustum_proportion_loss(batch, ssc_pred, bs)
            loss += frustum_loss
            self.log_loss(step_type, "loss_frustums", frustum_loss)

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true, nonempty=self.visible_eval_mask(batch))

        self.log_loss(step_type, "loss", loss)
        return loss

    def frustum_proportion_loss(self, batch, ssc_pred, bs):
        frustums_masks = torch.stack(batch["frustums_masks"])
        frustums_class_dists = torch.stack(batch["frustums_class_dists"]).float()
        n_frustums = frustums_class_dists.shape[1]

        pred_prob = F.softmax(ssc_pred, dim=1)
        batch_cnt = frustums_class_dists.sum(0)

        frustum_loss = 0
        frustum_nonempty = 0
        for frus in range(n_frustums):
            frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
            prob = frustum_mask * pred_prob
            prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
            prob = prob.reshape(self.n_classes, -1)
            cum_prob = prob.sum(dim=1)

            total_cnt = torch.sum(batch_cnt[frus])
            total_prob = prob.sum()
            if total_prob > 0 and total_cnt > 0:
                frustum_target_proportion = batch_cnt[frus] / total_cnt
                frustum_loss += KL_sep(cum_prob / total_prob, frustum_target_proportion)
                frustum_nonempty += 1
        return frustum_loss / frustum_nonempty


class SparseMonoScene(MonoScene):

    def build_3d_decoder(self, feature, full_scene_size, n_relations, context_prior):
        import spconv.pytorch as spconv
        from monoscene.models.sparse_unet3d import SparseUNet3D

        self.sparse_tensor = spconv.SparseConvTensor
        return SparseUNet3D(
            self.n_classes,
            feature=feature,
            full_scene_size=full_scene_size,
            n_relations=n_relations,
            context_prior=context_prior,
        )

    def forward(self, batch):
        img = batch["img"]
        x_rgb = self.net_rgb(img)
        coords, features, query_coords, targets = self.sparse_inputs(batch, x_rgb, img.device)

        spatial_shape = tuple(batch["target"].shape[1:])
        x_sparse = self.sparse_tensor(
            features,
            coords,
            spatial_shape=list(spatial_shape),
            batch_size=img.shape[0],
        )

        out = self.net_3d_decoder(x_sparse)
        query_rows = self.coord_rows(out["query_coords"], query_coords, spatial_shape)
        out["query_coords"] = query_coords
        out["ssc_logit_sparse"] = out["ssc_logit_sparse"][query_rows]
        out["ssc_logit"] = self.dense_grid_from_sparse(
            out["query_coords"], out["ssc_logit_sparse"], spatial_shape, img.shape[0]
        )
        out["ssc_target_sparse"] = targets
        return out

    def sparse_inputs(self, batch, x_rgb, device):
        coords, features, query_coords, targets = [], [], [], []
        for batch_idx in range(batch["img"].shape[0]):
            lift_mask = self.lift_mask(batch, batch_idx, device)
            query_mask = self.query_mask(batch, batch_idx, device)

            projected_pix = batch["projected_pix_1"][batch_idx].to(device).long()
            fov_mask = batch["fov_mask_1"][batch_idx].to(device).bool()

            projected_pix = projected_pix[lift_mask.reshape(-1)]
            fov_mask = fov_mask[lift_mask.reshape(-1)]
            feature = self.lift_features(x_rgb, batch_idx, projected_pix, fov_mask)

            lift_coords = lift_mask.nonzero(as_tuple=False).int()
            query_coord = query_mask.nonzero(as_tuple=False).int()

            coords.append(self.add_batch_column(lift_coords, batch_idx))
            features.append(feature)
            query_coords.append(self.add_batch_column(query_coord, batch_idx))
            targets.append(batch["target"][batch_idx].to(device)[query_mask])

        coords = torch.cat(coords)
        features = torch.cat(features)
        query_coords = torch.cat(query_coords)
        coords, features, query_coords = self.merge_sparse_inputs(coords, features, query_coords)
        return coords, features, query_coords, torch.cat(targets)

    def lift_mask(self, batch, batch_idx, device):
        return batch["observed_mask"][batch_idx].to(device).bool()

    def query_mask(self, batch, batch_idx, device):
        return batch["observed_mask"][batch_idx].to(device).bool()

    def merge_sparse_inputs(self, coords, features, query_coords):
        n = coords.shape[0]
        all_coords = torch.cat((coords, query_coords), dim=0)
        coords, inv = torch.unique(all_coords, dim=0, return_inverse=True)

        merged_features = features.new_zeros((coords.shape[0], features.shape[1]))
        merged_features.index_add_(0, inv[:n], features)
        return coords, merged_features, query_coords

    def add_batch_column(self, coord, batch_idx):
        batch_column = coord.new_full((coord.shape[0], 1), batch_idx)
        return torch.cat((batch_column, coord), dim=1)

    def lift_features(self, x_rgb, batch_idx, projected_pix, fov_mask):
        feature = None
        for scale_2d in self.project_res:
            scale_2d = int(scale_2d)
            gathered = self.lift_visible_features(
                x_rgb[f"1_{scale_2d}"][batch_idx],
                projected_pix,
                fov_mask,
                scale_2d,
            )
            feature = gathered if feature is None else feature + gathered
        return feature

    def lift_visible_features(self, x2d, projected_pix, fov_mask, scale_2d):
        _, h, w = x2d.shape
        pix = projected_pix // scale_2d
        pix_x = pix[:, 0].clamp_(0, w - 1)
        pix_y = pix[:, 1].clamp_(0, h - 1)

        features = x2d[:, pix_y, pix_x].transpose(0, 1).contiguous()
        if fov_mask.all():
            return features

        features[~fov_mask] = 0
        return features

    def coord_rows(self, coords, query_coords, spatial_shape):
        keys = self.coord_keys(coords, spatial_shape)
        query_keys = self.coord_keys(query_coords, spatial_shape)
        order = keys.argsort()
        return order[torch.searchsorted(keys[order], query_keys)]

    def coord_keys(self, coords, spatial_shape):
        b, x, y, z = coords.long().T
        x_size, y_size, z_size = spatial_shape
        return ((b * x_size + x) * y_size + y) * z_size + z

    def dense_grid_from_sparse(self, coords, values, spatial_shape, batch_size, fill_value=0):
        b, x, y, z = coords.long().T
        if values.ndim == 1:
            dense = values.new_full((batch_size, *spatial_shape), fill_value)
            dense[b, x, y, z] = values
            return dense

        dense = values.new_full(
            (batch_size, values.shape[1], *spatial_shape), fill_value
        )
        dense[b, :, x, y, z] = values
        return dense

    def step(self, batch, step_type, metric):
        loss = 0
        out_dict = self(batch)
        logits = out_dict["ssc_logit_sparse"]
        target = out_dict["ssc_target_sparse"]

        if self.context_prior and self.relation_loss:
            loss_rel_ce = compute_super_CP_multilabel_loss(
                out_dict["P_logits"], batch["CP_mega_matrices"]
            )
            loss += loss_rel_ce
            self.log_loss(step_type, "loss_relation_ce_super", loss_rel_ce)

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = sparse_ce_ssc_loss(logits, target, class_weight)
            loss += loss_ssc
            self.log_loss(step_type, "loss_ssc", loss_ssc)

        if self.sem_scal_loss:
            loss_sem_scal = sparse_sem_scal_loss(logits, target)
            loss += loss_sem_scal
            self.log_loss(step_type, "loss_sem_scal", loss_sem_scal)

        if self.geo_scal_loss:
            loss_geo_scal = sparse_geo_scal_loss(logits, target)
            loss += loss_geo_scal
            self.log_loss(step_type, "loss_geo_scal", loss_geo_scal)

        y_true = batch["target"].cpu().numpy()
        y_pred_sparse = logits.detach().argmax(dim=1)
        y_pred = self.dense_grid_from_sparse(
            out_dict["query_coords"],
            y_pred_sparse,
            tuple(batch["target"].shape[1:]),
            batch["target"].shape[0],
        ).cpu().numpy()
        metric.add_batch(y_pred, y_true, nonempty=self.visible_eval_mask(batch))

        self.log_loss(step_type, "loss", loss)
        return loss


MONOSCENE_MODELS = {
    "dense": DenseMonoScene,
    "sparse": SparseMonoScene,
}


def get_monoscene_model_class(model):
    return MONOSCENE_MODELS[model]
