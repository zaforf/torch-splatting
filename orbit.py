import math
import argparse

import torch
import numpy as np
import imageio

from data import load_nerf_synthetic
from camera import Camera, to_viewpoint_camera
from model import GSModel
from renderer import render


def orbit_c2w(theta_deg, phi_deg, radius):
    """c2w for a camera orbiting the origin (OpenCV convention: z=forward, y=down)."""
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    pos = torch.tensor([
        radius * math.cos(p) * math.cos(t),
        radius * math.cos(p) * math.sin(t),
        radius * math.sin(p),
    ], dtype=torch.float32)

    cam_z = -pos / pos.norm()                              # points toward origin
    world_up = torch.tensor([0., 0., 1.] if abs(cam_z[2].item()) < 0.99
                             else [0., 1., 0.], dtype=torch.float32)
    cam_x = torch.linalg.cross(cam_z, world_up);  cam_x = cam_x / cam_x.norm()
    cam_y = torch.linalg.cross(cam_z, cam_x);     cam_y = cam_y / cam_y.norm()

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = cam_x   # right
    c2w[:3, 1] = cam_y   # down
    c2w[:3, 2] = cam_z   # forward
    c2w[:3, 3] = pos
    return c2w


def render_orbit(model, ref_cam, n_frames=60, phi=30.0, radius=4.0, device="cuda"):
    intrinsic = torch.zeros(4, 4, device=device)
    intrinsic[0, 0] = ref_cam.focal_x
    intrinsic[1, 1] = ref_cam.focal_y

    frames = []
    for i in range(n_frames):
        theta = 360.0 * i / n_frames
        c2w = orbit_c2w(theta, phi, radius).to(device)
        cam = Camera(ref_cam.image_width, ref_cam.image_height, intrinsic, c2w)
        with torch.no_grad():
            img = render(cam, model).clamp(0, 1).cpu().numpy()
        frames.append((img * 255).astype(np.uint8))
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--data",   default="lego")
    parser.add_argument("--resize", type=float, default=0.5)
    parser.add_argument("--frames", type=int,   default=60)
    parser.add_argument("--phi",    type=float, default=30.0,
                        help="elevation above horizontal in degrees")
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--fps",    type=int,   default=15)
    parser.add_argument("--out",    default="orbit.gif")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_nerf_synthetic(args.data, split="train", resize_factor=args.resize)
    ref_cam = to_viewpoint_camera(data["camera"][0].to(device))

    model = GSModel()
    model.load_ply(args.ply, device=device)
    print(f"Loaded {len(model._xyz):,} gaussians, sh_deg={model.sh_deg}")

    frames = render_orbit(model, ref_cam, args.frames, args.phi, args.radius, device)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"saved {args.out}")
