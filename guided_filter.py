"""Cross-Guided Filter (CGF) decomposition -- Section 3.2 of FusionMCHV."""
import torch
import torch.nn.functional as F


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean filter with window radius r (implemented with avg_pool)."""
    k = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode='reflect')
    return F.avg_pool2d(x, k, stride=1)


def guided_filter(I: torch.Tensor, P: torch.Tensor, r: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """Classic guided filter (He et al.). I: guidance, P: source. Both (B,1,H,W) in [0,1]."""
    mean_I = box_filter(I, r)
    mean_P = box_filter(P, r)
    corr_I = box_filter(I * I, r)
    corr_IP = box_filter(I * P, r)
    var_I = corr_I - mean_I * mean_I
    cov_IP = corr_IP - mean_I * mean_P
    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    return mean_a * I + mean_b


def cross_guided_decompose(ir: torch.Tensor, vis: torch.Tensor,
                           r: int = 4, eps: float = 1e-3):
    """CGF: IR and VIS guide each other's decomposition (Eqs. 9-12).

    Returns: ir_base, ir_detail, vis_base, vis_detail  (each B,1,H,W)
    """
    ir_base = guided_filter(vis, ir, r, eps)    # VIS guides IR  (Eq. 9)
    vis_base = guided_filter(ir, vis, r, eps)   # IR  guides VIS (Eq. 11)
    return ir_base, ir - ir_base, vis_base, vis - vis_base