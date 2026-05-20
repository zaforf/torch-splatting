import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import load_nerf_synthetic
from camera import to_viewpoint_camera
from model import GSModel
from renderer import render


def show_views(model, data, indices=(0, 10, 20, 30), device="cuda"):
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
    for row, i in enumerate(indices):
        cam = to_viewpoint_camera(data["camera"][i])
        with torch.no_grad():
            img = render(cam, model).clamp(0, 1).cpu().numpy()
        gt = data["rgb"][i].cpu().numpy()
        axes[row][0].imshow(img);  axes[row][0].set_title(f"rendered [{i}]"); axes[row][0].axis("off")
        axes[row][1].imshow(gt);   axes[row][1].set_title(f"gt [{i}]");       axes[row][1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--data",   default="lego")
    parser.add_argument("--resize", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_nerf_synthetic(args.data, split="train", resize_factor=args.resize)
    data = {k: v.to(device) for k, v in data.items()}

    model = GSModel()
    model.load_ply(args.ply, device=device)
    print(f"Loaded {len(model._xyz):,} gaussians, sh_deg={model.sh_deg}")

    show_views(model, data, device=device)
