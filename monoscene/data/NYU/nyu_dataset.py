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
        sparse=False,
    ):
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.sparse = sparse
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
        target = data["target_1_4"]
        data["target"] = target

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
            projected_pix[:, 0] = img.shape[1] - 1 - projected_pix[:, 0]

        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)
        data["surface_mask"] = data.get("surface_mask", data["visible_mask_1_4"])
        data["observed_mask"] = data.get("observed_mask", data["surface_mask"])

        if self.sparse:
            return self.sparse_data(data, projected_pix, fov_mask)

        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask

        target_1_16 = data["target_1_16"]
        data["CP_mega_matrix"] = compute_CP_mega_matrix(
            target_1_16, is_binary=self.n_relations == 2
        )

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
        data.pop("surface_coords", None)
        data.pop("observed_coords", None)
        data.pop("halo_mask", None)
        data.pop("observed_halo_size", None)

        return data

    def sparse_data(self, data, projected_pix, fov_mask):
        coords = data.get("observed_coords")
        coords = (
            np.argwhere(data["observed_mask"]).astype(np.int32)
            if coords is None
            else coords
        )
        coords = coords.astype(np.int32, copy=False)
        projected_pix, fov_mask = self.projection_volumes(projected_pix, fov_mask)
        x, y, z = coords.T

        return {
            "cam_pose": data["cam_pose"],
            "voxel_origin": data["voxel_origin"],
            "cam_k": data["cam_k"],
            "name": data["name"],
            "img": data["img"],
            "sparse_coords": coords,
            "sparse_projected_pix_1": projected_pix[x, y, z],
            "sparse_fov_mask_1": fov_mask[x, y, z],
            "sparse_target": data["target"][x, y, z],
        }

    def projection_volumes(self, projected_pix, fov_mask):
        grid_shape = np.ceil(np.array(self.scene_size) / self.voxel_size).astype(int)
        projected_pix = projected_pix.reshape(*grid_shape, 2)
        fov_mask = fov_mask.reshape(*grid_shape)
        projected_pix = np.moveaxis(projected_pix, [0, 1, 2], [0, 2, 1])
        fov_mask = np.moveaxis(fov_mask, [0, 1, 2], [0, 2, 1])
        return projected_pix, fov_mask

    def __len__(self):
        return len(self.scan_names)
