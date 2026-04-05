import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
from monoscene.data.NYU.params import (
    NYU_CAM_K,
    NYU_VOXEL_SIZE,
    NYU_IMG_W,
    NYU_IMG_H,
    NYU_SCENE_SIZE,
)
import pickle
import torch.nn.functional as F


class NYUDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
    ):
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.n_classes = 12
        self.root = os.path.join(root, "NYU" + split)
        self.preprocess_root = preprocess_root
        self.base_dir = os.path.join(preprocess_root, "base", "NYU" + split)
        self.fliplr = fliplr

        self.voxel_size = NYU_VOXEL_SIZE
        self.scene_size = NYU_SCENE_SIZE
        self.img_W = NYU_IMG_W
        self.img_H = NYU_IMG_H
        self.cam_k = NYU_CAM_K

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.scan_names = glob.glob(os.path.join(self.root, "*.bin"))

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        filepath = os.path.join(self.base_dir, name + ".pkl")

        with open(filepath, "rb") as handle:
            data = pickle.load(handle)

        cam_pose = data["cam_pose"]
        T_world_2_cam = np.linalg.inv(cam_pose)
        vox_origin = data["voxel_origin"]
        data["cam_k"] = self.cam_k
        target = data[
            "target_1_4"
        ]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        data["target"] = target
        target_1_4 = data["target_1_16"]

        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
        data["CP_mega_matrix"] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_world_2_cam,
            self.cam_k,
            vox_origin,
            self.voxel_size,
            self.img_W,
            self.img_H,
            self.scene_size,
        )
        
        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_W,
            self.img_H,
            dataset="NYU",
            n_classes=12,
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.asarray(img, dtype=np.float32) / 255.0

        # randomly fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][:, 0] = (
                img.shape[1] - 1 - data["projected_pix_1"][:, 0]
            )

        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)

        return data

    def __len__(self):
        return len(self.scan_names)
