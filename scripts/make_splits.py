import argparse, random
from pathlib import Path
import shutil

# 将 YOLO 数据根下的 train 全量划分为 train/val/test（可复现）

def split(root, val=0.1, test=0.1, seed=2025):
    root = Path(root)
    imgs = sorted((root/'images/train').glob('*'))
    random.Random(seed).shuffle(imgs)
    n = len(imgs); nv = int(n*val); nt = int(n*test)
    parts = {'train': imgs[nv+nt:], 'val': imgs[:nv], 'test': imgs[nv:nv+nt]}
    for p, lst in parts.items():
        (root/f'images/{p}').mkdir(parents=True, exist_ok=True)
        (root/f'labels/{p}').mkdir(parents=True, exist_ok=True)
        for im in lst:
            lbl = root/'labels/train'/(im.stem+'.txt')
            shutil.copy(im, root/f'images/{p}'/im.name)
            if lbl.exists():
                shutil.copy(lbl, root/f'labels/{p}'/(im.stem+'.txt'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=2025)
    args = ap.parse_args()
    split(args.root, args.val, args.test, args.seed)