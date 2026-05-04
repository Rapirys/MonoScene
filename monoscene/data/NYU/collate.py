import torch


def collate_fn(batch):
    imgs = []
    names = []
    cam_poses = []

    vox_origins = []
    cam_ks = []

    optional = {}

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        cam_poses.append(torch.from_numpy(input_dict["cam_pose"]).float())
        vox_origins.append(torch.from_numpy(input_dict["voxel_origin"]).double())

        names.append(input_dict["name"])

        img = input_dict["img"]
        imgs.append(img)

        for key, value in input_dict.items():
            if key in {"cam_k", "cam_pose", "voxel_origin", "name", "img"}:
                continue
            key = "CP_mega_matrices" if key == "CP_mega_matrix" else key
            optional.setdefault(key, []).append(torch.from_numpy(value))

    ret_data = {
        "cam_pose": torch.stack(cam_poses),
        "cam_k": torch.stack(cam_ks),
        "vox_origin": torch.stack(vox_origins),
        "name": names,
        "img": torch.stack(imgs),
    }

    for key, values in optional.items():
        ret_data[key] = torch.stack(values) if key == "target" else values
    if "sparse_coords" in ret_data:
        sparse_coords = ret_data["sparse_coords"]
        ret_data["sparse_coords"] = torch.cat([
            torch.cat((coords.new_full((coords.shape[0], 1), idx), coords), dim=1)
            for idx, coords in enumerate(sparse_coords)
        ])
        for key in ("sparse_projected_pix_1", "sparse_fov_mask_1", "sparse_target"):
            ret_data[key] = torch.cat(ret_data[key])
    return ret_data
