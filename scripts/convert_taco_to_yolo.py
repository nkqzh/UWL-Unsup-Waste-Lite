# scripts/convert_taco_to_yolo.py
"""
将完整 TACO 数据集 (batch_1 ~ batch_15) 转换为 YOLO 检测格式。

目录结构：
    external/TACO/data/
        batch_1/
        batch_2/
        ...
        batch_15/
        annotations.json

annotations.json 里的 file_name 一般形如:
    "file_name": "batch_1/000001.jpg"

输出：
    datasets/data/taco_yolo/
        images/train/batch_x/*.jpg
        images/val/batch_x/*.jpg
        images/test/batch_x/*.jpg
        labels/train/batch_x/*.txt
        labels/val/batch_x/*.txt
        labels/test/batch_x/*.txt
"""

import json
import random
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

def coco_bbox_to_yolo_bbox(bbox, img_w, img_h):
    """
    COCO bbox: [x, y, w, h] (左上角 + 宽高, 像素)
    转为 YOLO: (cx, cy, w, h) (相对坐标, 0~1)
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return cx, cy, bw, bh

def main():
    random.seed(0)

    project_root = Path(__file__).resolve().parents[1]
    taco_root = project_root / "external" / "TACO" / "data"
    ann_path = taco_root / "annotations.json"

    if not ann_path.is_file():
        raise FileNotFoundError(f"找不到 annotations.json: {ann_path}")

    print(f"[TACO] 使用标注文件: {ann_path}")
    print(f"[TACO] 数据根目录: {taco_root}")

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    imgid2info = {img["id"]: img for img in images}
    imgid2annos = defaultdict(list)
    for a in annotations:
        imgid2annos[a["image_id"]].append(a)

    total_imgs = len(images)
    imgs_with_ann = sum(1 for img in images if imgid2annos[img["id"]])
    print(f"[TACO] 总图片数: {total_imgs}")
    print(f"[TACO] 至少含 1 个标注的图片数: {imgs_with_ann}")
    print(f"[TACO] 总标注框数: {len(annotations)}")

    # 使用全部 1500 张图片
    all_img_ids = list(imgid2info.keys())
    random.shuffle(all_img_ids)

    n_total = len(all_img_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ids = all_img_ids[:n_train]
    val_ids = all_img_ids[n_train:n_train + n_val]
    test_ids = all_img_ids[n_train + n_val:]

    print(f"[Split] train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)} (共 {n_total})")

    # 类别映射
    catid2name = {c["id"]: c["name"] for c in categories}
    sorted_cat_ids = sorted(catid2name.keys())
    catid2yoloid = {cid: i for i, cid in enumerate(sorted_cat_ids)}
    print(f"[TACO] 类别数: {len(sorted_cat_ids)}")

    out_root = project_root / "datasets" / "data" / "taco_yolo"

    def process_split(split_name, img_ids):
        print(f"\n[Convert] 处理 {split_name} 集, 图片数: {len(img_ids)}")
        for img_id in tqdm(img_ids):
            info = imgid2info[img_id]
            file_name = info["file_name"]  # e.g. "batch_1/000001.jpg"
            w, h = info["width"], info["height"]

            src_img_path = taco_root / file_name
            if not src_img_path.is_file():
                print(f"  ⚠ 找不到图片文件: {src_img_path}, 跳过该图片")
                continue

            # 保留 batch_x 子目录结构
            dst_img_path = out_root / "images" / split_name / file_name
            dst_img_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img_path.exists():
                dst_img_path.write_bytes(src_img_path.read_bytes())

            # label 也保留同样的子目录结构，保证一一对应
            dst_label_path = out_root / "labels" / split_name / file_name
            dst_label_path = dst_label_path.with_suffix(".txt")
            dst_label_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            for a in imgid2annos.get(img_id, []):
                cat_id = a["category_id"]
                if cat_id not in catid2yoloid:
                    continue
                cls = catid2yoloid[cat_id]
                bbox = a["bbox"]
                cx, cy, bw, bh = coco_bbox_to_yolo_bbox(bbox, w, h)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # 有标注写框，没有标注写空文件
            dst_label_path.write_text("\n".join(lines), encoding="utf-8")

    process_split("train", train_ids)
    process_split("val", val_ids)
    process_split("test", test_ids)

    print("\n✅ TACO → YOLO 转换完成。输出目录:")
    print(f"   {out_root.resolve()}")
    print("   注意：images/* 目录中包含 batch_x 子文件夹，YOLO 会递归读取。")

if __name__ == "__main__":
    main()
