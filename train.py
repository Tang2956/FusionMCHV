# Training script for FusionMCHV (config per Sec. 4.1d):
#   Adam(beta1=0.9, beta2=0.999), lr=5e-4, batch=4, epochs=50,
#   loss weights (5, 10, 1). Trained on MSRS (1083 pairs), 640x480, [0,1].
# NOTE: paper does not state the training patch size; HVSM is solved on CPU
# with sparse linear systems, so the default patch is 64x64. Increase it
# (e.g. 128) if your CPU memory/time budget allows.
import os, argparse, time, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from models import FusionMCHV
from losses import FusionLoss


class PairedDataset(Dataset):
    # MSRS-style folder: root/ir/*.png and root/vis/*.png with identical names.
    def __init__(self, root, patch=64, train=True):
        self.ir_dir = os.path.join(root, 'ir')
        self.vis_dir = os.path.join(root, 'vis')
        self.names = sorted(os.listdir(self.ir_dir))
        self.patch, self.train = patch, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        ir = np.asarray(Image.open(os.path.join(self.ir_dir, n)).convert('L'), np.float32) / 255.
        vis = np.asarray(Image.open(os.path.join(self.vis_dir, n)).convert('L'), np.float32) / 255.
        if self.train and self.patch:
            h, w = ir.shape
            ps = self.patch
            i = random.randint(0, h - ps)
            j = random.randint(0, w - ps)
            ir, vis = ir[i:i + ps, j:j + ps], vis[i:i + ps, j:j + ps]
        return (torch.from_numpy(ir)[None], torch.from_numpy(vis)[None], n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train', help='folder with ir/ and vis/ subfolders')
    ap.add_argument('--out', default='runs/fusionmchv')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = PairedDataset(args.data, patch=args.patch, train=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = FusionMCHV().to(args.device)
    crit = FusionLoss(l1=5.0, l2=10.0, l3=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for ir, vis, _ in dl:
            ir, vis = ir.to(args.device), vis.to(args.device)
            fused = model(ir, vis)
            loss, parts = crit(ir, vis, fused)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f'epoch {ep:3d}/{args.epochs}  avg_loss {tot/len(dl):.4f}  '
              f'int {parts["int"]:.4f} grad {parts["grad"]:.4f} ssim {parts["ssim"]:.4f}  {time.time()-t0:.1f}s')
        torch.save(model.state_dict(), os.path.join(args.out, 'ckpt_last.pth'))
    torch.save(model.state_dict(), os.path.join(args.out, 'ckpt_final.pth'))


if __name__ == '__main__':
    main()