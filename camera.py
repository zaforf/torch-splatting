import math

import torch
import torch.nn as nn


def parse_camera(params):
    """Unpack the flat [N, 34] camera tensor from data.py."""
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape(-1, 4, 4)
    c2w = params[:, 18:34].reshape(-1, 4, 4)
    return H, W, intrinsics, c2w


def to_viewpoint_camera(cam_params):
    """Convert a single [34] camera tensor into a Camera object."""
    H, W, K, c2w = parse_camera(cam_params.unsqueeze(0))
    return Camera(int(W[0]), int(H[0]), K[0], c2w[0])


class Camera(nn.Module):
    def __init__(self, width, height, intrinsic, c2w, znear=0.01, zfar=100.0):
        super().__init__()
        self.image_width  = width
        self.image_height = height
        self.znear = znear
        self.zfar  = zfar

        self.focal_x = intrinsic[0, 0]
        self.focal_y = intrinsic[1, 1]
        self.FoVx = focal2fov(self.focal_x, width)
        self.FoVy = focal2fov(self.focal_y, height)

        # world_view_transform: w2c transposed (row-major, right-multiply convention)
        w2c = torch.linalg.inv(c2w)
        self.world_view_transform = w2c.T

        self.projection_matrix = _projection_matrix(znear, zfar, self.FoVx, self.FoVy).T.to(c2w)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        # camera origin in world space
        self.camera_center = c2w[:3, 3]


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def _projection_matrix(znear, zfar, fovX, fovY):
    t = math.tan(fovY / 2) * znear
    r = math.tan(fovX / 2) * znear
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * znear / (2 * r)
    P[1, 1] = 2 * znear / (2 * t)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
