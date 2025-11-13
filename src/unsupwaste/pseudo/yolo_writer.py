from pathlib import Path
import shutil
from PIL import Image
import yaml

# 将检测框写为 YOLO txt；类别按 configs/classes.yaml 顺序映射

def _load_classes():
    with open('configs/classes.yaml','r',encoding='utf-8') as f:
        return yaml.safe_load(f)['names']

CLASSES = _load_classes()

def write_yolo_det(image_path, dets, out_dir):
    out_dir = Path(out_dir)
    out_img = out_dir/'images'/Path(image_path).name
    out_lbl = out_dir/'labels'/(Path(image_path).stem+'.txt')
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    shutil.copy(image_path, out_img)

    with open(out_lbl, 'w', encoding='utf-8') as f:
        for d in dets:
            x1,y1,x2,y2 = d['xyxy']
            cx, cy = ((x1+x2)/2)/W, ((y1+y2)/2)/H
            bw, bh = (x2-x1)/W, (y2-y1)/H
            name = d.get('text','other_waste')
            cls = CLASSES.index(name) if name in CLASSES else CLASSES.index('other_waste')
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")