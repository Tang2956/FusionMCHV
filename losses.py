# Loss functions -- Section 3.5.3 of FusionMCHV (Eqs. 38-41).
# All terms are computed in the HVSM (human visual) space: local layers for
# gradient/intensity, global layers for structure.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hvsm import hvsm_transform


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    # Sobel gradient magnitude (differentiable), Eq. (40).
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                      device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32,
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), kx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def gaussian_window(ws: int = 11, sigma: float = 1.5, device='cpu') -> torch.Tensor:
    g = torch.exp(-(torch.arange(ws, dtype=torch.float32, device=device) - ws // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    w = g[:, None] @ g[None, :]
    return w.expand(1, 1, ws, ws).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Standard SSIM (Eq. 41), images in [0,1]-like range.
    ws, C = 11, a.shape[1]
    w = gaussian_window(ws, device=a.device).expand(C, 1, ws, ws)
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    va = F.conv2d(a * a, w) - mu_a ** 2
    vb = F.conv2d(b * b, w) - mu_b ** 2
    vab = F.conv2d(a * b, w) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return m.mean()


class FusionLoss(nn.Module):
    # L_total = l1*L_int + l2*L_grad + l3*L_SSIM  (Eq. 38; l=(5,10,1) per Sec. 4.1d)
    def __init__(self, l1=5.0, l2=10.0, l3=1.0, gamma=0.1, alpha=1e-4):
        super().__init__()
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.gamma, self.alpha = gamma, alpha

    def hvsm_batch(self, t: torch.Tensor):
        # tensor (B,1,H,W) in [0,1] -> (I_L, I_G) tensors on same device
        out_L, out_G = [], []
        for i in range(t.shape[0]):
            arr = t[i, 0].detach().cpu().numpy().astype(np.float64)
            L, G = hvsm_transform(arr, gamma=self.gamma, alpha=self.alpha)
            out_L.append(torch.from_numpy(L).float())
            out_G.append(torch.from_numpy(G).float())
        dev = t.device
        return (torch.stack(out_L).unsqueeze(1).to(dev),
                torch.stack(out_G).unsqueeze(1).to(dev))

    def forward(self, ir, vis, fused):
        ir_L, ir_G = self.hvsm_batch(ir)
        vis_L, vis_G = self.hvsm_batch(vis)
        f_L, f_G = self.hvsm_batch(fused)
        # HVSM is a non-differentiable CPU transform; re-inject gradients with a
        # straight-through estimator (value = transformed, d()/dfused = identity).
        f_L = f_L + (fused - fused.detach())
        f_G = f_G + (fused - fused.detach())

        # Eq. (39): intensity loss on BOTH local and global layers
        l_int = (torch.mean(torch.abs(f_L - torch.maximum(ir_L, vis_L))) +
                 torch.mean(torch.abs(f_G - torch.maximum(ir_G, vis_G))))
        # Eq. (40): gradient loss on local layers only
        l_grad = torch.mean(torch.abs(sobel_mag(f_L) -
                                      torch.maximum(sobel_mag(ir_L), sobel_mag(vis_L))))
        # Eq. (41): structural loss on global layers
        l_ssim = (1 - ssim(f_G, ir_G)) / 2 + (1 - ssim(f_G, vis_G)) / 2
        total = self.l1 * l_int + self.l2 * l_grad + self.l3 * l_ssim
        return total, {'int': l_int.item(), 'grad': l_grad.item(), 'ssim': l_ssim.item()}


if __name__ == '__main__':
    ir = torch.rand(2, 1, 64, 64)
    vis = torch.rand(2, 1, 64, 64)
    fused = torch.rand(2, 1, 64, 64)
    loss, parts = FusionLoss()(ir, vis, fused)
    print('loss %.4f' % loss.item(), parts)