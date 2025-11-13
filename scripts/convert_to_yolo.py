import argparse, json
from pathlib import Path
import shutil
from tqdm import tqdm

# 将 COCO 标注（如 TACO）转换为 YOLO 检测 txt

def coco_to_yolo(coco_json, raw_img_root, out_dir):
    out = Path(out_dir)
    (out/'images/train').mkdir(parents=True, exist_ok=True)
    (out/'labels/train').mkdir(parents=True, exist_ok=True)
    id2file = {im['id']: im['file_name'] for im in coco_json['images']}
    id2im = {im['id']: im for im in coco_json['images']}
    for ann in tqdm(coco_json['annotations']):
        img_file = id2file[ann['image_id']]
        src = Path(raw_img_root)/img_file
        dst_img = out/'images/train'/img_file
        dst_lbl = out/'labels/train'/(Path(img_file).stem+'.txt')
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy(src, dst_img)
        x, y, w, h = ann['bbox']
        W, H = id2im[ann['image_id']]['width'], id2im[ann['image_id']]['height']
        cx, cy, bw, bh = (x+w/2)/W, (y+h/2)/H, w/W, h/H
        cls = 0 # 简化：统一映射为 "waste"，或根据类别名定制
        with open(dst_lbl, 'a', encoding='utf-8') as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='包含 COCO 标注文件 taco.json 的目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw', default='data/raw/taco/images')
    args = ap.parse_args()
    coco_path = Path(args.source)/'taco.json'
    coco_json = json.load(open(coco_path,'r',encoding='utf-8'))
    coco_to_yolo(coco_json, args.raw, args.out)