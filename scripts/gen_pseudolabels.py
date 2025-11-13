import argparse, os, glob
from pathlib import Path
import torch
from unsupwaste.pseudo.gdino_runner import GDRunner
from unsupwaste.pseudo.sam2_runner import SAM2Runner
from unsupwaste.pseudo.yolo_writer import write_yolo_det

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--cfg', default='configs/pseudo.yaml')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gd = GDRunner.from_cfg(args.cfg, device=device)
    sam = SAM2Runner(device=device)

    out_dir = Path(args.out); (out_dir/"labels").mkdir(parents=True, exist_ok=True); (out_dir/"images").mkdir(parents=True, exist_ok=True)

    imgs = sorted(sum([glob.glob(os.path.join(args.images, ext)) for ext in ('*.jpg','*.jpeg','*.png','*.bmp')], []))
    for im in imgs:
        dets = gd.detect(im)
        dets = sam.refine(im, dets) # 当前为直传，后续可启用掩码细化
        write_yolo_det(im, dets, out_dir)

if __name__ == '__main__':
    main()