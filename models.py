# FusionMCHV network -- Section 3 of the paper.
# Structure follows Fig. 3 exactly:
#   CGF pre-decomposition
#   -> base branch : Conv1x1 -> GRDense -> CCMM        (Sec. 3.3.2, Figs. 9-10)
#   -> detail branch: Conv1x1 -> DenseBlock -> MDECM   (Sec. 3.3.1, Figs. 5-8)
#   -> concat + D_Conv fusion -> Res reconstruction with Sigmoid (Sec. 3.4)
# NOTE (reasonable assumptions, not specified in the paper):
#   * channel width C = 16 for every branch
#   * guided-filter radius r = 4, eps = 1e-3
#   * CCMM local window k = 16 (chosen in Sec. 4.2.1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter import cross_guided_decompose


# ---------------------------------------------------------------- Sobel
class SobelDir(nn.Module):
    # Fixed Sobel operator in one of 4 directions (0/45/90/135 deg).
    KERNELS = {
        0:   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        45:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        90:  [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        135: [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    }

    def __init__(self, theta: int, ch: int):
        super().__init__()
        w = torch.tensor(self.KERNELS[theta], dtype=torch.float32)
        w = w.view(1, 1, 3, 3).repeat(ch, 1, 1, 1)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False, groups=ch)
        self.conv.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------- blocks
class DenseBlock(nn.Module):
    # Dense block of Fig. 5: 3 x (Conv3x3 + LReLU) with dense connections.
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(2 * ch, ch, 3, padding=1)
        self.c3 = nn.Conv2d(3 * ch, ch, 3, padding=1)
        self.out = nn.Conv2d(3 * ch, ch, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        return self.out(torch.cat([x1, x2, x3], 1))


class GRDense(nn.Module):
    # Gradient residual dense block of Fig. 9 (base branch extractor).
    def __init__(self, ch: int):
        super().__init__()
        self.dense = DenseBlock(ch)
        self.d1 = nn.Conv2d(ch, ch, 1)
        self.sobel = SobelDir(0, ch)
        self.g1 = nn.Conv2d(ch, ch, 1)
        self.fuse = nn.Conv2d(2 * ch, ch, 1)

    def forward(self, x):
        d = self.d1(self.dense(x))
        g = self.g1(self.sobel(x))
        return self.fuse(torch.cat([x, d + g], 1))


class DConv(nn.Module):
    # Two convolutional units (Conv3x3+BN+ReLU) used for fusion, Sec. 3.4.
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


# ---------------------------------------------------------------- CCMM
def local_pearson(x: torch.Tensor, tmpl: torch.Tensor, k: int, chunk: int = 32) -> torch.Tensor:
    # Local Pearson correlation between a global saliency template (B,C,1,1)
    # and every k x k window of x. Returns (B, k*k, H, W)  -- Eqs. (21)-(22).
    B, C, H, W = x.shape
    # total pad k-1 so that unfold with kernel k yields exactly H x W windows
    if k % 2 == 0:
        pl, pr, pt, pb = k // 2 - 1, k // 2, k // 2 - 1, k // 2
    else:
        pl = pr = pt = pb = k // 2
    xp = F.pad(x, (pl, pr, pt, pb), mode='reflect')
    t = tmpl - tmpl.mean(dim=1, keepdim=True)          # centered template
    var_t = t.pow(2).sum(dim=1).clamp_min(1e-12)       # (B,1,1)
    t = t.squeeze(-1).squeeze(-1)                      # (B,C)
    out = x.new_zeros(B, k * k, H, W)
    for h0 in range(0, H, chunk):
        h1 = min(h0 + chunk, H)
        pw = xp[:, :, h0:h0 + (h1 - h0) + k - 1, :]
        patches = pw.unfold(2, k, 1).unfold(3, k, 1)   # B,C,ch,W,k,k
        patches = patches.reshape(B, C, h1 - h0, W, k * k)
        xc = patches - patches.mean(dim=1, keepdim=True)
        var_x = xc.pow(2).sum(dim=1).clamp_min(1e-12)  # B,ch,W,k^2
        cov = torch.einsum('bcwhn,bc->bwhn', xc, t)
        G = cov / (var_x.sqrt() * var_t.sqrt().view(B, 1, 1, 1) + 1e-8)
        out[:, :, h0:h1, :] = G.permute(0, 3, 1, 2)
    return out


class CCMM(nn.Module):
    # Cross-correlation metric module, Fig. 10 / Sec. 3.3.2 (k=16, Sec. 4.2.1).
    def __init__(self, ch: int, k: int = 16):
        super().__init__()
        self.k = k
        self.red_ir = nn.Conv2d(k * k, ch, 1)      # G_ib   -> I_ib
        self.red_vis = nn.Conv2d(k * k, ch, 1)
        self.crs_ir = nn.Conv2d(k * k, ch, 1)      # G_cb   -> W_cb
        self.crs_vis = nn.Conv2d(k * k, ch, 1)

    def forward(self, f_ir, f_vis):
        # local saliency templates (Eq. 21)
        t_ir = F.adaptive_max_pool2d(f_ir, 1) + F.adaptive_avg_pool2d(f_ir, 1)
        t_vis = F.adaptive_max_pool2d(f_vis, 1) + F.adaptive_avg_pool2d(f_vis, 1)
        # auto-correlation maps (Eq. 22)
        g_ir = local_pearson(f_ir, t_ir, self.k)     # B,k^2,H,W
        g_vis = local_pearson(f_vis, t_vis, self.k)
        i_ir = self.red_ir(g_ir)                     # I_ib (C x H x W)
        i_vis = self.red_vis(g_vis)
        # local probability representations (global max + sigmoid)
        p_ir = torch.sigmoid(F.adaptive_max_pool2d(i_ir, 1))
        p_vis = torch.sigmoid(F.adaptive_max_pool2d(i_vis, 1))
        # cross-modal correlation (Eqs. 23-24)
        gc_ir = local_pearson(i_vis, p_ir, self.k)   # B,k^2,H,W
        gc_vis = local_pearson(i_ir, p_vis, self.k)
        w_ir = torch.sigmoid(self.crs_ir(gc_ir))     # W_cb (C x H x W)
        w_vis = torch.sigmoid(self.crs_vis(gc_vis))
        return f_ir * w_ir, f_vis * w_vis            # Eq. (25)


# ---------------------------------------------------------------- CEAM / MDECM
class CEAM(nn.Module):
    # Cross-modal edge attention module, Fig. 7 / Eqs. (14)-(17).
    # CEAM(a, b): channel attention of a x spatial probability of b -> weight for a.
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.PReLU(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.s1 = nn.Conv2d(1, 1, 3, padding=1)  # spatial map is channel-mean (1 ch)
        self.w = nn.Conv2d(2 * ch, ch, 1)

    def forward(self, a, b):
        # channel branch: adaptive max pool over space -> Conv3-PReLU-Conv3 (Eq. 14)
        m = a.amax(dim=(2, 3), keepdim=True)                 # B,C,1,1
        f_ca = self.c2(self.act(self.c1(m)))                 # B,C,1,1
        # spatial branch: channel-mean -> Conv3 -> Sigmoid (Eq. 15)
        f_sa = torch.sigmoid(self.s1(a.mean(dim=1, keepdim=True)))   # B,1,H,W
        f_sb = torch.sigmoid(self.s1(b.mean(dim=1, keepdim=True)))
        # decoupled weights: cross-inhibition + intrinsic (Eqs. 16-17)
        w = self.w(torch.cat([f_ca * (1.0 - f_sb), f_ca * f_sa], dim=1))
        return a * w


class MDECM(nn.Module):
    # Multi-directional edge correlation module, Fig. 6 / Sec. 3.3.1.
    DIRS = (0, 45, 90, 135)

    def __init__(self, ch: int):
        super().__init__()
        self.integ_ir = nn.Conv2d(ch, ch, 3, padding=1)
        self.integ_vis = nn.Conv2d(ch, ch, 3, padding=1)
        self.dir_ir = nn.ModuleList([SobelDir(t, ch) for t in self.DIRS])
        self.dir_vis = nn.ModuleList([SobelDir(t, ch) for t in self.DIRS])
        self.w1 = nn.Conv2d(4 * ch, ch, 1)
        self.w2 = nn.Conv2d(ch, 4, 1)
        self.ceams = nn.ModuleList([CEAM(ch) for _ in self.DIRS])

    def mesem(self, x, integ, dirs):
        # Multi-directional edge self-enhancement (MESEM, Fig. 6 left/right).
        x = F.leaky_relu(integ(x), 0.2, inplace=True)
        edges = [d(x) for d in dirs]                          # 4 x (B,C,H,W)
        w = torch.sigmoid(self.w2(F.leaky_relu(self.w1(torch.cat(edges, 1)), 0.2, inplace=True)))
        return [edges[i] * w[:, i:i + 1] for i in range(4)]

    def forward(self, f_ir, f_vis):
        e_ir = self.mesem(f_ir, self.integ_ir, self.dir_ir)
        e_vis = self.mesem(f_vis, self.integ_vis, self.dir_vis)
        c_ir = sum(self.ceams[i](e_ir[i], e_vis[i]) for i in range(4))    # Eq. (18) sum
        c_vis = sum(self.ceams[i](e_vis[i], e_ir[i]) for i in range(4))
        return c_ir, c_vis


# ---------------------------------------------------------------- full model
class FusionMCHV(nn.Module):
    def __init__(self, ch: int = 16, gf_r: int = 4, gf_eps: float = 1e-3, k: int = 16):
        super().__init__()
        self.gf_r, self.gf_eps = gf_r, gf_eps
        # base branch
        self.in_b = nn.Conv2d(1, ch, 1)
        self.grd_ir, self.grd_vis = GRDense(ch), GRDense(ch)
        self.ccmm = CCMM(ch, k)
        # detail branch
        self.in_d_ir = nn.Conv2d(1, ch, 1)
        self.in_d_vis = nn.Conv2d(1, ch, 1)
        self.dns_ir, self.dns_vis = DenseBlock(ch), DenseBlock(ch)
        self.mdecm = MDECM(ch)
        # fusion + reconstruction (Sec. 3.4)
        self.dconv_b = DConv(2 * ch, ch)
        self.dconv_d = DConv(2 * ch, ch)
        self.res = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # 1) CGF pre-decomposition (Sec. 3.2)
        ir_b, ir_d, vis_b, vis_d = cross_guided_decompose(ir, vis, self.gf_r, self.gf_eps)
        # 2) base branch (Sec. 3.3.2): independent -> CCMM -> sum  (Eqs. 26-27)
        f_ib_ir = self.grd_ir(self.in_b(ir_b))
        f_ib_vis = self.grd_vis(self.in_b(vis_b))
        f_cb_ir, f_cb_vis = self.ccmm(f_ib_ir, f_ib_vis)
        f_b_ir, f_b_vis = f_ib_ir + f_cb_ir, f_ib_vis + f_cb_vis
        # 3) detail branch (Sec. 3.3.1): independent -> MDECM -> sum (Eqs. 18-19)
        f_id_ir = self.dns_ir(self.in_d_ir(ir_d))
        f_id_vis = self.dns_vis(self.in_d_vis(vis_d))
        f_cd_ir, f_cd_vis = self.mdecm(f_id_ir, f_id_vis)
        f_d_ir, f_d_vis = f_id_ir + f_cd_ir, f_id_vis + f_cd_vis
        # 4) fusion + reconstruction (Eq. 28)
        f_b = self.dconv_b(torch.cat([f_b_ir, f_b_vis], 1))
        f_d = self.dconv_d(torch.cat([f_d_ir, f_d_vis], 1))
        return self.res(torch.cat([f_b, f_d], 1))


if __name__ == '__main__':
    m = FusionMCHV()
    x = torch.rand(2, 1, 96, 96)
    y = m(x, x)
    print('output:', y.shape, 'params:', sum(p.numel() for p in m.parameters()))