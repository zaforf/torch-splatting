from math import exp

import torch
import torch.nn.functional as F


def l1_loss(pred, gt):
    return (pred - gt).abs().mean()


def ssim(img1, img2, window_size=11):
    """
    SSIM for images shaped [1, C, H, W].
    Call as: ssim(rendered.permute(2,0,1).unsqueeze(0), gt.permute(2,0,1).unsqueeze(0))
    """
    C = img1.size(1)
    window = _make_window(window_size, C).to(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

    sig1 = F.conv2d(img1*img1, window, padding=window_size//2, groups=C) - mu1_sq
    sig2 = F.conv2d(img2*img2, window, padding=window_size//2, groups=C) - mu2_sq
    sig12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=C) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + c1) * (2*sig12 + c2)) / ((mu1_sq+mu2_sq+c1) * (sig1+sig2+c2))
    return ssim_map.mean()


def _make_window(window_size, channels):
    g = torch.tensor([exp(-(x - window_size//2)**2 / (2*1.5**2)) for x in range(window_size)])
    g = g / g.sum()
    w = g.outer(g).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    return w.expand(channels, 1, window_size, window_size).contiguous()
