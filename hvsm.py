"""Human Visual Spatial Model (HVSM) -- Section 3.5.2 of FusionMCHV.

FARTV: frequency-aware relative total variation smoothing (Eqs. 29-36),
then the contrast-based transform (Eq. 37) producing local/global layers.
Solved on CPU with sparse linear systems (per image), intended for small
patches (e.g. 64x64) during training.
"""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized
from scipy.ndimage import uniform_filter


def _fwd_diff(n):
    return diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format='csr')


def fartv_smooth(img: np.ndarray, lam: float = 0.05, win: int = 9,
                 niter: int = 5, eps: float = 1e-4, freq_aware: bool = True) -> np.ndarray:
    """One level of FARTV smoothing. img: (H,W) float64 in [0,1]."""
    H, W = img.shape
    v = img.astype(np.float64).ravel()
    DY = kron(_fwd_diff(H), eye(W, format='csr'), format='csr')   # (H-1)*W x HW
    DX = kron(eye(H, format='csr'), _fwd_diff(W), format='csr')   # H*(W-1) x HW

    S = v.copy()
    for _ in range(niter):
        gy = (DY @ S).reshape(H - 1, W)
        gx = (DX @ S).reshape(H, W - 1)
        # windowed total variation D_* and window variation L_* (box window as in RTV practice)
        Dx = uniform_filter(np.abs(gx), size=win)
        Dy = uniform_filter(np.abs(gy), size=win)
        Lx = np.abs(uniform_filter(gx, size=win))
        Ly = np.abs(uniform_filter(gy, size=win))

        # spatial-frequency-aware weighting term  (Eq. 33)
        if freq_aware:
            fx = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
            fy = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
            f = np.sqrt(fx ** 2 + fy ** 2)
            ratio = (f + eps) / (f.mean() + eps)
        else:
            ratio = np.ones_like(Dx)

        wx, ux = 1.0 / (Lx + eps), Dx / (Lx + eps)
        wy, uy = 1.0 / (Ly + eps), Dy / (Ly + eps)
        Lmat = (DY.T @ diags((wy * uy).ravel()) @ DY +
                DX.T @ diags((wx * ux).ravel()) @ DX)
        A = lam * diags(ratio.ravel()) @ Lmat + eye(H * W, format='csr')
        solve = factorized(A.tocsc())
        S = solve(v)
    return S.reshape(H, W)


def hvsm_transform(img: np.ndarray, gamma: float = 0.1, alpha: float = 1e-4,
                   lam1: float = 0.05, lam2: float = 0.05, win: int = 9,
                   niter: int = 5) -> tuple:
    """Eq. (36)+(37): decompose into I1, I2 then build local/global layers.

    Returns (I_L, I_G) with the same shape as img.
    """
    img = img.astype(np.float64)
    I1 = fartv_smooth(img, lam=lam1, win=win, niter=niter)
    I2 = fartv_smooth(I1, lam=lam2, win=win, niter=niter)
    m0, m1, m2 = np.log10(img.mean() + 1e-12), np.log10(I1.mean() + 1e-12), np.log10(I2.mean() + 1e-12)
    I_L = (m0 / m1) * (img - I1) / np.power(I1 + alpha, gamma)
    I_G = (m1 / m2) * (I1 - I2) / np.power(I2 + alpha, gamma)
    return I_L, I_G