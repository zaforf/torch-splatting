import os
import json
import math

import numpy as np
import torch
import imageio.v2 as imageio


def load_nerf_synthetic(folder, split="train", resize_factor=1.0, white_bkgd=True):
    """
    Load a NeRF synthetic (Blender) dataset split.

    Returns a dict with:
        rgb:    [N, H, W, 3] float32 in [0, 1]
        alpha:  [N, H, W]   float32 in [0, 1]
        camera: [N, 34]     float32 layout: [H, W, intrinsic_4x4 (16), c2w_4x4 (16)]
    """
    with open(os.path.join(folder, f"transforms_{split}.json")) as f:
        meta = json.load(f)

    rgbs, alphas, cameras = [], [], []

    for frame in meta["frames"]:
        # image
        img_path = os.path.join(folder, frame["file_path"] + ".png")
        img = imageio.imread(img_path).astype(np.float32) / 255.0  # [H, W, 4]

        H, W = img.shape[:2]
        if resize_factor != 1.0:
            import torch.nn.functional as F
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, scale_factor=resize_factor, mode="bilinear", align_corners=False)
            img = t.squeeze(0).permute(1, 2, 0).numpy()
            H, W = img.shape[:2]

        rgb   = img[..., :3]
        alpha = img[..., 3]

        if white_bkgd:
            rgb = rgb * alpha[..., None] + (1.0 - alpha[..., None])

        # camera — focal from horizontal FoV, images are square so fx == fy
        focal = (W / 2.0) / math.tan(meta["camera_angle_x"] / 2.0)

        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = W / 2.0
        intrinsic[1, 2] = H / 2.0

        # Blender/OpenGL (Y-up, Z-back) -> OpenCV/COLMAP (Y-down, Z-forward)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:, 1:3] *= -1

        # flat layout expected by camera_utils.parse_camera: [H, W, K(16), c2w(16)]
        cam = np.concatenate([
            [float(H), float(W)],
            intrinsic.flatten(),
            c2w.flatten(),
        ]).astype(np.float32)

        rgbs.append(rgb)
        alphas.append(alpha)
        cameras.append(cam)

    return {
        "rgb":    torch.from_numpy(np.stack(rgbs,    axis=0)),
        "alpha":  torch.from_numpy(np.stack(alphas,  axis=0)),
        "camera": torch.from_numpy(np.stack(cameras, axis=0)),
    }
