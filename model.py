import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GSModel(nn.Module):

    def __init__(self, sh_deg: int = 3):
        super(GSModel, self).__init__()
        self.sh_deg = sh_deg

        self._xyz          = torch.empty(0)
        self._features_dc  = torch.empty(0)  # [N, 1, 3]  — degree-0 SH (one per RGB channel)
        self._features_rest = torch.empty(0) # [N, K-1, 3] — degrees 1-3
        self._scaling      = torch.empty(0)  # log scale, so exp() keeps it positive
        self._rotation     = torch.empty(0)  # quaternion [w,x,y,z], normalized before use
        self._opacity      = torch.empty(0)  # pre-sigmoid, so sigmoid() keeps it in (0,1)

    def init_from_random(self, n: int, device="cuda"):
        # scatter points randomly inside a sphere of radius 1.5
        xyz = F.normalize(torch.randn(n, 3), dim=-1)
        xyz = xyz * torch.rand(n, 1).pow(1/3) * 1.5
        xyz = xyz.to(device)

        K = (self.sh_deg + 1) ** 2
        features_dc   = torch.zeros(n, 1, 3, device=device)
        features_rest = torch.zeros(n, K - 1, 3, device=device)

        # small isotropic blobs to start
        scaling  = torch.full((n, 3), -3.0, device=device)  # exp(-3) ≈ 0.05
        # identity rotation: w=1, x=y=z=0
        rotation = torch.zeros(n, 4, device=device)
        rotation[:, 0] = 1.0
        # low opacity so early renders don't immediately saturate
        opacity  = inverse_sigmoid(torch.full((n, 1), 0.1, device=device))

        self._xyz          = nn.Parameter(xyz)
        self._features_dc  = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling      = nn.Parameter(scaling)
        self._rotation     = nn.Parameter(rotation)
        self._opacity      = nn.Parameter(opacity)

    # --- activations applied on the way out ---

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)  # [N, K, 3]

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    # --- saving ---

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement

        xyz      = self._xyz.detach().cpu().numpy()
        f_dc     = self._features_dc.detach().transpose(1, 2).flatten(1).cpu().numpy()
        f_rest   = self._features_rest.detach().transpose(1, 2).flatten(1).cpu().numpy()
        opacity  = self._opacity.detach().cpu().numpy()
        scaling  = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attrs  = ['x','y','z','nx','ny','nz']
        attrs += [f'f_dc_{i}'   for i in range(f_dc.shape[1])]
        attrs += [f'f_rest_{i}' for i in range(f_rest.shape[1])]
        attrs += ['opacity']
        attrs += [f'scale_{i}'  for i in range(3)]
        attrs += [f'rot_{i}'    for i in range(4)]

        normals = np.zeros_like(xyz)
        data    = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scaling, rotation], axis=1)
        el = PlyElement.describe(
            np.array([tuple(r) for r in data], dtype=[(a, 'f4') for a in attrs]),
            'vertex')
        PlyData([el]).write(path)
