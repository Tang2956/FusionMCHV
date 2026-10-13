# Test/inference script: fuse registered IR/VIS pairs with a trained checkpoint.
import os, argparse
import numpy as np
import torch
from PIL import Image
from models import FusionMCHV


def to_tensor(p):
    return torch.from_numpy(np.asarray(p.convert('L'), np.float32) / 255.)[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/fusionmchv/ckpt_final.pth')
    ap.add_argument('--ir', default='data/test/ir')
    ap.add_argument('--vis', default='data/test/vis')
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = FusionMCHV().to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    for n in sorted(os.listdir(args.ir)):
        ir = to_tensor(Image.open(os.path.join(args.ir, n)))
        vis = to_tensor(Image.open(os.path.join(args.vis, n)))
        ir, vis = ir.to(args.device), vis.to(args.device)
        h, w = ir.shape[-2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8   # pad to multiple of 8
        with torch.no_grad():
            fused = model(torch.nn.functional.pad(ir, (0, pw, 0, ph), mode='replicate'),
                          torch.nn.functional.pad(vis, (0, pw, 0, ph), mode='replicate'))
        out = (fused[..., :h, :w].clamp(0, 1).cpu().numpy()[0, 0] * 255).astype(np.uint8)
        Image.fromarray(out).save(os.path.join(args.out, n))
        print('fused ->', n)


if __name__ == '__main__':
    main()