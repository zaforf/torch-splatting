import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import load_nerf_synthetic
from camera import to_viewpoint_camera
from model import GSModel, inverse_sigmoid
from renderer import render, build_rotation
from loss_utils import l1_loss, ssim


def psnr(pred, gt):
    return -10 * torch.log10((pred - gt).pow(2).mean())


# ---------------------------------------------------------------------------
# optimizer setup
# ---------------------------------------------------------------------------

def make_optimizer(model):
    # named groups so we can identify params during densification
    return torch.optim.Adam([
        {"params": [model._xyz],           "lr": 1.6e-4, "name": "xyz"},
        {"params": [model._features_dc],   "lr": 2.5e-3, "name": "features_dc"},
        {"params": [model._features_rest], "lr": 2.5e-4, "name": "features_rest"},
        {"params": [model._opacity],       "lr": 5e-2,   "name": "opacity"},
        {"params": [model._scaling],       "lr": 5e-3,   "name": "scaling"},
        {"params": [model._rotation],      "lr": 1e-3,   "name": "rotation"},
    ], eps=1e-15)


PARAM_ATTR = {
    "xyz": "_xyz", "features_dc": "_features_dc",
    "features_rest": "_features_rest", "opacity": "_opacity",
    "scaling": "_scaling", "rotation": "_rotation",
}


def replace_params(optimizer, model, new_tensors, keep_mask, n_new):
    """
    Swap all parameter tensors in model + optimizer in one shot.
    Adam state for kept gaussians is preserved; n_new new entries get zero state.
    """
    for group in optimizer.param_groups:
        name  = group["name"]
        old_p = group["params"][0]
        new_p = nn.Parameter(new_tensors[name])

        old_st = optimizer.state.pop(old_p, {})
        new_st = {}
        for key in ("exp_avg", "exp_avg_sq"):
            if key in old_st:
                kept  = old_st[key][keep_mask]
                zeros = torch.zeros(n_new, *kept.shape[1:], device=kept.device)
                new_st[key] = torch.cat([kept, zeros])
        if "step" in old_st:
            new_st["step"] = old_st["step"]

        group["params"][0] = new_p
        optimizer.state[new_p] = new_st
        setattr(model, PARAM_ATTR[name], new_p)


# ---------------------------------------------------------------------------
# densification  (paper §5.2 — clone, split, prune)
# ---------------------------------------------------------------------------

def densify_and_prune(model, optimizer, grad_accum, grad_count, scene_extent=2.0, grad_threshold=2e-4):
    avg_grad = (grad_accum / grad_count.clamp(min=1)).squeeze(-1)  # [N]

    is_large   = model.get_scaling.max(dim=1).values >= 0.01 * scene_extent
    needs_d    = avg_grad > grad_threshold
    clone_mask = needs_d & ~is_large   # under-reconstructed small gaussians -> duplicate
    split_mask = needs_d &  is_large   # over-reconstructed large gaussians  -> split into 2
    prune_mask = (model.get_opacity.squeeze() < 5e-3) | \
                 (model.get_scaling.max(dim=1).values > scene_extent)
    keep_mask  = ~prune_mask

    # positions and scales for split gaussians
    n_split, split_xyz, split_scaling = split_mask.sum().item(), None, None
    if n_split:
        scales    = model.get_scaling[split_mask]
        offsets   = torch.randn(2*n_split, 3, device=scales.device) * scales.repeat(2, 1)
        R         = build_rotation(model.get_rotation[split_mask].repeat(2, 1))
        split_xyz = (R @ offsets.unsqueeze(-1)).squeeze(-1) + model._xyz[split_mask].repeat(2, 1)
        split_scaling = torch.log(scales.repeat(2, 1) / 1.6)

    def build_new(name, attr):
        src   = getattr(model, attr).data
        parts = [src[keep_mask]]
        if clone_mask.any():
            parts.append(src[clone_mask])
        if n_split:
            if   name == "xyz":     parts.append(split_xyz.detach())
            elif name == "scaling": parts.append(split_scaling.detach())
            else:
                reps = (2,) + (1,) * (src.dim() - 1)
                parts.append(src[split_mask].repeat(reps))
        return torch.cat(parts)

    new_tensors = {n: build_new(n, a) for n, a in PARAM_ATTR.items()}
    n_new = int(clone_mask.sum() + 2 * n_split)
    replace_params(optimizer, model, new_tensors, keep_mask, n_new)

    N = len(model._xyz)
    return (torch.zeros(N, 1, device=model._xyz.device),
            torch.zeros(N, 1, device=model._xyz.device, dtype=torch.long))


def reset_opacity(model, optimizer):
    """
    Clamp opacity back near zero every 3k steps.
    Prevents gaussians from becoming permanently opaque and blocking densification.
    """
    with torch.no_grad():
        model._opacity.data.clamp_(max=inverse_sigmoid(torch.tensor(0.01)).item())
    # zero out opacity momentum so the optimizer doesn't immediately undo the reset
    for group in optimizer.param_groups:
        if group["name"] == "opacity":
            st = optimizer.state.get(group["params"][0], {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key].zero_()


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="lego")
    parser.add_argument("--scene_type", default="blender", choices=["blender", "mip360", "llff"])
    parser.add_argument("--steps",       type=int,   default=30_000)
    parser.add_argument("--n_gaussians", type=int,   default=50_000)
    parser.add_argument("--sh_degree",   type=int,   default=3)
    parser.add_argument("--resize",      type=float, default=0.5)
    parser.add_argument("--white_bkgd",  action="store_true", default=True)
    parser.add_argument("--no_white_bkgd",     dest="white_bkgd", action="store_false")
    parser.add_argument("--densify_grad_thresh", type=float, default=2e-4)
    parser.add_argument("--densify_until",       type=int,   default=15_000)
    parser.add_argument("--scene_extent",        type=float, default=2.0)
    parser.add_argument("--out",                 default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join("output", os.path.basename(args.data.rstrip("/")))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    if args.scene_type == "mip360":
        from data import load_mip360
        data = load_mip360(args.data, split="train", resize_factor=args.resize)
    elif args.scene_type == "llff":
        from data import load_llff
        data = load_llff(args.data, split="train", resize_factor=args.resize)
    else:
        data = load_nerf_synthetic(args.data, split="train", resize_factor=args.resize)
    data   = {k: v.to(device) for k, v in data.items()}
    N_cams = len(data["rgb"])

    model = GSModel(sh_deg=args.sh_degree)  # allocate full SH capacity
    model.init_from_random(args.n_gaussians, device=device)
    model.sh_deg = 0                        # start evaluating at degree 0
    opt   = make_optimizer(model)

    # xyz LR decays exponentially: 1.6e-4 -> 1.6e-6 over training
    xyz_lr_decay = (1.6e-6 / 1.6e-4) ** (1.0 / args.steps)

    # grad accumulators for densification signal (use xyz.grad norm as proxy)
    grad_accum = torch.zeros(args.n_gaussians, 1, device=device)
    grad_count = torch.zeros(args.n_gaussians, 1, device=device, dtype=torch.long)

    pbar = tqdm(range(1, args.steps + 1))
    for step in pbar:
        i      = np.random.randint(N_cams)
        cam    = to_viewpoint_camera(data["camera"][i])
        gt_rgb = data["rgb"][i]

        rendered = render(cam, model, white_bkgd=args.white_bkgd)

        loss = (0.8 * l1_loss(rendered, gt_rgb)
              + 0.2 * (1.0 - ssim(rendered.permute(2,0,1).unsqueeze(0),
                                   gt_rgb.permute(2,0,1).unsqueeze(0))))

        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            # accumulate gradient signal for densification
            if model._xyz.grad is not None and len(model._xyz) == len(grad_accum):
                grad_accum += model._xyz.grad.norm(dim=-1, keepdim=True)
                grad_count += 1

        opt.step()

        # decay xyz LR each step
        opt.param_groups[0]["lr"] *= xyz_lr_decay

        # progressive SH: add one degree per 1000 steps up to max
        model.sh_deg = min(args.sh_degree, step // 1000)

        # densification: clone/split/prune between steps 500–15000
        if 500 <= step <= args.densify_until and step % 100 == 0:
            grad_accum, grad_count = densify_and_prune(
                model, opt, grad_accum, grad_count,
                grad_threshold=args.densify_grad_thresh,
                scene_extent=args.scene_extent)

        if step % 500 == 0:
            p = psnr(rendered.detach(), gt_rgb).item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{p:.2f}", n={len(model._xyz)})

        if step % 5_000 == 0 or step == args.steps:
            model.save_ply(os.path.join(args.out, f"splats_{step:06d}.ply"))

        # opacity reset every 3000 steps — must come after save_ply
        if step % 3_000 == 0:
            reset_opacity(model, opt)


if __name__ == "__main__":
    main()
