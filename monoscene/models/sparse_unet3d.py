import spconv.pytorch as spconv
import torch.nn.functional as F
from torch import nn

from monoscene.models.CRP3D import CPMegaVoxels


class SparseBatchNorm(nn.Module):
    def __init__(self, channels, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, momentum=momentum)

    def forward(self, x):
        return x.replace_feature(self.bn(x.features))


class SparseReLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.replace_feature(F.relu(x.features, inplace=self.inplace))


class SparseBottleneck(nn.Module):
    def __init__(self, channels, bn_momentum, dilation=1, indice_key="proc"):
        super().__init__()
        mid_channels = max(channels // 4, 1)

        self.conv1 = spconv.SubMConv3d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = SparseBatchNorm(mid_channels, momentum=bn_momentum)
        self.relu1 = SparseReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
            indice_key=f"{indice_key}_d{dilation}",
        )
        self.bn2 = SparseBatchNorm(mid_channels, momentum=bn_momentum)
        self.relu2 = SparseReLU(inplace=True)

        self.conv3 = spconv.SubMConv3d(mid_channels, channels, kernel_size=1, bias=False)
        self.bn3 = SparseBatchNorm(channels, momentum=bn_momentum)
        self.relu3 = SparseReLU(inplace=True)

    def forward(self, x):
        identity = x.features

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = x.replace_feature(x.features + identity)
        return self.relu3(x)


class SparseProcess(nn.Module):
    def __init__(self, channels, bn_momentum, dilations, indice_prefix):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SparseBottleneck(
                    channels,
                    bn_momentum=bn_momentum,
                    dilation=dilation,
                    indice_key=f"{indice_prefix}_{dilation}",
                )
                for dilation in dilations
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class SparseDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum, indice_key):
        super().__init__()
        self.main = nn.Sequential(
            spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
                indice_key=indice_key,
            ),
            SparseBatchNorm(out_channels, momentum=bn_momentum),
            SparseReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class SparseUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum, indice_key):
        super().__init__()
        self.main = nn.Sequential(
            spconv.SparseInverseConv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                indice_key=indice_key,
                bias=False,
            ),
            SparseBatchNorm(out_channels, momentum=bn_momentum),
            SparseReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class SparseSegmentationHead(nn.Module):
    def __init__(self, channels, n_classes, dilations, bn_momentum):
        super().__init__()
        self.conv0 = nn.Sequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            SparseBatchNorm(channels, momentum=bn_momentum),
            SparseReLU(inplace=True),
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    spconv.SubMConv3d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                        indice_key=f"ssc_head_{dilation}",
                    ),
                    SparseBatchNorm(channels, momentum=bn_momentum),
                    SparseReLU(inplace=True),
                    spconv.SubMConv3d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                        indice_key=f"ssc_head_{dilation}",
                    ),
                    SparseBatchNorm(channels, momentum=bn_momentum),
                )
                for dilation in dilations
            ]
        )
        self.relu = SparseReLU(inplace=True)
        self.classifier = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = self.conv0(x)
        y = self.conv_list[0](x)
        for block in self.conv_list[1:]:
            y = y.replace_feature(y.features + block(x).features)
        x = self.relu(y.replace_feature(y.features + x.features))
        return self.classifier(x.features)


class DenseContextPriorAdapter(nn.Module):
    def __init__(self, feature, size, n_relations, bn_momentum):
        super().__init__()
        self.context_prior = CPMegaVoxels(
            feature,
            size,
            n_relations=n_relations,
            bn_momentum=bn_momentum,
        )
        self.size = size

    def forward(self, x):
        coords = x.indices.long()
        b, x_idx, y_idx, z_idx = coords.T
        dense = x.features.new_zeros((x.batch_size, x.features.shape[1], *self.size))
        dense[b, :, x_idx, y_idx, z_idx] = x.features

        ret = self.context_prior(dense)
        dense_features = ret["x"]
        sampled = dense_features[b, :, x_idx, y_idx, z_idx]

        return {"P_logits": ret["P_logits"], "x": x.replace_feature(sampled)}


class SparseUNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        feature,
        full_scene_size,
        n_relations=4,
        context_prior=True,
        bn_momentum=0.1,
    ):
        super().__init__()
        self.feature_1_4 = feature
        self.feature_1_8 = feature * 2
        self.feature_1_16 = feature * 4
        self.context_prior = context_prior

        self.process_1_4 = nn.Sequential(
            SparseProcess(self.feature_1_4, bn_momentum, [1, 2, 3], "sparse_l1"),
            SparseDownsample(
                self.feature_1_4,
                self.feature_1_8,
                bn_momentum,
                indice_key="sparse_down_1_4_1_8",
            ),
        )
        self.process_1_8 = nn.Sequential(
            SparseProcess(self.feature_1_8, bn_momentum, [1, 2, 3], "sparse_l2"),
            SparseDownsample(
                self.feature_1_8,
                self.feature_1_16,
                bn_momentum,
                indice_key="sparse_down_1_8_1_16",
            ),
        )
        self.up_1_16_1_8 = SparseUpsample(
            self.feature_1_16,
            self.feature_1_8,
            bn_momentum,
            indice_key="sparse_down_1_8_1_16",
        )
        self.up_1_8_1_4 = SparseUpsample(
            self.feature_1_8,
            self.feature_1_4,
            bn_momentum,
            indice_key="sparse_down_1_4_1_8",
        )
        self.ssc_head = SparseSegmentationHead(
            self.feature_1_4,
            class_num,
            [1, 2, 3],
            bn_momentum,
        )

        if context_prior:
            size_1_16 = tuple(int((dim + 3) // 4) for dim in full_scene_size)
            self.cp_block = DenseContextPriorAdapter(
                self.feature_1_16,
                size_1_16,
                n_relations=n_relations,
                bn_momentum=bn_momentum,
            )

    def forward(self, x):
        ret = {}

        x_1_4 = x
        x_1_8 = self.process_1_4(x_1_4)
        x_1_16 = self.process_1_8(x_1_8)

        if self.context_prior:
            cp_ret = self.cp_block(x_1_16)
            x_1_16 = cp_ret["x"]
            ret["P_logits"] = cp_ret["P_logits"]

        x_up_1_8 = self.up_1_16_1_8(x_1_16)
        x_up_1_8 = x_up_1_8.replace_feature(x_up_1_8.features + x_1_8.features)

        x_up_1_4 = self.up_1_8_1_4(x_up_1_8)
        x_up_1_4 = x_up_1_4.replace_feature(x_up_1_4.features + x_1_4.features)

        ret["ssc_logit_sparse"] = self.ssc_head(x_up_1_4)
        ret["query_coords"] = x_up_1_4.indices
        return ret
