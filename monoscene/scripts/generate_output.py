from pytorch_lightning import Trainer
from monoscene.models.monoscene import get_monoscene_model_class
from monoscene.data.NYU.nyu_dm import NYUDataModule
from monoscene.data.semantic_kitti.kitti_dm import KittiDataModule
from monoscene.data.kitti_360.kitti_360_dm import Kitti360DataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle


def autocast_dtype(precision):
    return torch.float16 if str(precision).startswith("16") else torch.bfloat16


@hydra.main(version_base=None, config_path="../config", config_name="monoscene.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    # Setup dataloader
    if config.dataset == "kitti" or config.dataset == "kitti_360":
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)

        if config.dataset == "kitti":
            data_module = KittiDataModule(
                root=config.kitti_root,
                preprocess_root=config.kitti_preprocess_root,
                frustum_size=config.frustum_size,
                batch_size=int(config.batch_size / config.n_gpus),
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            data_loader = data_module.val_dataloader()
            # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
        else:
            data_module = Kitti360DataModule(
                root=config.kitti_360_root,
                sequences=[config.kitti_360_sequence],
                n_scans=2000,
                batch_size=1,
                num_workers=3,
            )
            data_module.setup()
            data_loader = data_module.dataloader()

    elif config.dataset == "NYU":
        project_scale = 1
        feature = 200
        full_scene_size = (60, 36, 60)
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            sparse=config.model == "sparse",
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    else:
        print("dataset not support")

    # Load pretrained models
    if config.dataset == "NYU":
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "monoscene_nyu.ckpt"
        )
    else:
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "monoscene_kitti.ckpt"
        )

    model_cls = get_monoscene_model_class(config.model)
    model = model_cls.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
        use_visible_mask=config.use_visible_mask,
        context_heads=config.context_heads,
        context_depth=config.context_depth,
        context_dropout=config.context_dropout,
        weights_only=False,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch["img"] = batch["img"].cuda()
            with torch.autocast("cuda", dtype=autocast_dtype(config.precision)):
                pred = model(batch)
            dense_pred = "ssc_logit" in pred
            if dense_pred:
                y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = pred["ssc_logit_sparse"].argmax(dim=1).detach().cpu()

            for i in range(len(batch["img"])):
                if dense_pred:
                    out_dict = {"y_pred": y_pred[i].astype(np.uint16)}
                else:
                    query_coords = pred["query_coords"].detach().cpu()
                    query_mask = (query_coords[:, 0] == i).detach().cpu()
                    out_dict = {
                        "y_pred_sparse": y_pred[query_mask].numpy().astype(np.uint16),
                        "query_coords": query_coords[query_mask, 1:].numpy(),
                        "target_sparse": batch["sparse_target"][i].detach().cpu().numpy(),
                    }

                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                    if "observed_mask" in batch and batch["observed_mask"]:
                        out_dict["observed_mask"] = (
                            batch["observed_mask"][i].detach().cpu().numpy()
                        )
                    if dense_pred and "query_coords" in pred:
                        query_coords = pred["query_coords"]
                        query_coords = query_coords[query_coords[:, 0] == i, 1:]
                        out_dict["query_coords"] = query_coords.detach().cpu().numpy()
                else:
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                    out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )

                os.makedirs(write_path, exist_ok=True)
                with open(filepath, "wb") as handle:
                    pickle.dump(out_dict, handle)
                    print("wrote to", filepath)


if __name__ == "__main__":
    main()
