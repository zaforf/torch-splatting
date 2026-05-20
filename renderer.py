import math

import torch
import torch.nn.functional as F

from sh_utils import eval_sh
from camera import Camera
from model import GSModel


def homogenous(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def projection_ndc(points, viewmatrix, projmatrix):
    points = homogenous(points)
    p_view = points @ viewmatrix          # world -> camera space
    p_clip = p_view @ projmatrix          # camera -> clip space
    p_proj = p_clip / (p_clip[..., -1:] + 1e-6)  # perspective divide -> NDC
    in_mask = p_view[..., 2] >= 0.2      # cull points behind camera
    return p_proj, p_view, in_mask


def get_colors(means3D, shs, sh_deg, camera: Camera):
    rays_d = means3D - camera.camera_center
    rays_d = F.normalize(rays_d, dim=-1)
    # eval_sh expects [N, 3, K], get_features returns [N, K, 3]
    colors = eval_sh(sh_deg, shs.permute(0, 2, 1), rays_d)
    return (colors + 0.5).clamp(min=0.0)


def build_rotation(q):
    # q: [N, 4] as [w, x, y, z]
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
        2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x),
        2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def build_cov3d(scales, quats):
    R = build_rotation(quats)           # [N, 3, 3]
    S = torch.diag_embed(scales)        # [N, 3, 3]
    L = R @ S                           # [N, 3, 3]
    return L @ L.transpose(1, 2)        # R S S^T R^T  ->  [N, 3, 3]


def build_cov2d(means3D, cov3d, viewmatrix, focal_x, focal_y, fov_x, fov_y):
    # camera-space positions
    t  = means3D @ viewmatrix[:3, :3] + viewmatrix[3:4, :3]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    # clamp to just outside frustum edges to avoid cov blowup
    tx = tx.clamp(-math.tan(fov_x * 0.5) * 1.3 * tz, math.tan(fov_x * 0.5) * 1.3 * tz)
    ty = ty.clamp(-math.tan(fov_y * 0.5) * 1.3 * tz, math.tan(fov_y * 0.5) * 1.3 * tz)

    # jacobian of perspective projection (EWA splatting Eq. 29)
    J = torch.zeros(len(means3D), 3, 3, device=means3D.device)
    J[:, 0, 0] =  focal_x / tz
    J[:, 0, 2] = -focal_x * tx / (tz * tz)
    J[:, 1, 1] =  focal_y / tz
    J[:, 1, 2] = -focal_y * ty / (tz * tz)

    W = viewmatrix[:3, :3].T            # world->cam rotation
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)

    # low-pass filter (Eq. 32) — avoids aliasing from undersampled gaussians
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    return cov2d[:, :2, :2]             # [N, 2, 2]


def get_radius(cov2d):
    # radius of the 3-sigma bounding circle, from the larger eigenvalue
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    l1  = mid + (mid**2 - det).clamp(min=0.1).sqrt()
    return (3.0 * l1.sqrt()).ceil()     # [N]


def rasterize(means2D, cov2d, colors, opacity, depths, W, H, white_bkgd=True):
    TILE = 64
    dev  = means2D.device

    # analytical 2x2 inverse — avoids linalg.inv per tile
    det    = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp(min=1e-6)
    conics = torch.stack([
         cov2d[:, 1, 1] / det,
        -cov2d[:, 0, 1] / det,
         cov2d[:, 0, 0] / det,
    ], dim=-1)  # [N, 3] as (a, b, c) where Σ^{-1} = [[a,b],[b,c]]

    # global depth sort once — avoids argsort per tile
    order   = torch.argsort(depths)
    means2D = means2D[order]
    conics  = conics[order]
    colors  = colors[order]
    opacity = opacity[order]

    radii = get_radius(cov2d[order])
    lo    = means2D - radii[:, None]
    hi    = means2D + radii[:, None]

    pix = torch.stack(torch.meshgrid(
        torch.arange(W, device=dev, dtype=torch.float32),
        torch.arange(H, device=dev, dtype=torch.float32),
        indexing='xy'), dim=-1)         # [H, W, 2]

    out = torch.ones(H, W, 3, device=dev) if white_bkgd else torch.zeros(H, W, 3, device=dev)

    for tile_y in range(0, H, TILE):
        for tile_x in range(0, W, TILE):
            ty = min(TILE, H - tile_y)
            tx = min(TILE, W - tile_x)

            in_tile = (
                (lo[:, 0] < tile_x + tx) & (hi[:, 0] >= tile_x) &
                (lo[:, 1] < tile_y + ty) & (hi[:, 1] >= tile_y)
            )
            if not in_tile.any():
                continue

            # already depth-sorted globally — no argsort needed
            g_xy  = means2D[in_tile]    # [P, 2]
            g_con = conics[in_tile]     # [P, 3]
            g_col = colors[in_tile]     # [P, 3]
            g_opa = opacity[in_tile]    # [P, 1]

            tile_pix = pix[tile_y:tile_y+ty, tile_x:tile_x+tx].reshape(-1, 2)
            d  = tile_pix[:, None, :] - g_xy[None, :, :]
            du, dv = d[..., 0], d[..., 1]

            q = g_con[None,:,0]*du*du + 2*g_con[None,:,1]*du*dv + g_con[None,:,2]*dv*dv

            alpha = (torch.exp(-0.5 * q).unsqueeze(-1) * g_opa[None]).clamp(max=0.99)
            T     = torch.cat([torch.ones_like(alpha[:,:1]), 1 - alpha[:,:-1]], dim=1).cumprod(1)

            tile_color = (T * alpha * g_col[None]).sum(1)
            if white_bkgd:
                tile_color += (1 - (alpha * T).sum(1))

            out[tile_y:tile_y+ty, tile_x:tile_x+tx] = tile_color.reshape(ty, tx, 3)

    return out


def render(camera: Camera, model: GSModel, white_bkgd=True):
    means3D   = model.get_xyz
    shs       = model.get_features
    scales    = model.get_scaling
    rotations = model.get_rotation
    opacity   = model.get_opacity

    means_ndc, means_view, in_mask = projection_ndc(
        means3D, camera.world_view_transform, camera.projection_matrix)

    # apply frustum mask to everything from here on
    means3D   = means3D[in_mask]
    means_ndc = means_ndc[in_mask]
    depths    = means_view[in_mask, 2]
    shs       = shs[in_mask]
    scales    = scales[in_mask]
    rotations = rotations[in_mask]
    opacity   = opacity[in_mask]

    colors = get_colors(means3D, shs, model.sh_deg, camera)

    cov3d  = build_cov3d(scales, rotations)
    cov2d  = build_cov2d(means3D, cov3d,
                         camera.world_view_transform,
                         camera.focal_x, camera.focal_y,
                         camera.FoVx, camera.FoVy)

    W, H = camera.image_width, camera.image_height
    means2D = torch.stack([
        (means_ndc[:, 0] + 1) * W * 0.5 - 0.5,
        (means_ndc[:, 1] + 1) * H * 0.5 - 0.5,
    ], dim=-1)

    return rasterize(means2D, cov2d, colors, opacity, depths, W, H, white_bkgd)
