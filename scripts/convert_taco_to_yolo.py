# scripts/convert_taco_to_yolo.py
"""
将 TACO 的 COCO 标注转换为 YOLO 检测格式（单类：waste）。
默认输入：
    external/TACO/data/annotations.json
    external/TACO/data/batch_x/*.jpg （官方下载脚本的结构）
默认输出：
    data/taco_yolo/images/{train,val,test}
    data/taco_yolo/labels/{train,val,test}
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_w, img_h):
    """COCO: [x_min, y_min, w, h] -> YOLO: [cx, cy, w, h] (normalized)."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    return [
        cx / img_w,
        cy / img_h,
        w / img_w,
        h / img_h,
    ]

def build_image_index(coco):
    # id -> image info
    img_idx: Dict[int, Dict] = {}
    for img in coco["images"]:
        img_idx[img["id"]] = img
    return img_idx

def build_ann_index(coco):
    # image_id -> list of annotations
    ann_idx: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        ann_idx.setdefault(img_id, []).append(ann)
    return ann_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taco-root",
        type=str,
        default="external/TACO",
        help="TACO 仓库根目录（包含 data/annotations.json）",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/taco_yolo",
        help="输出 YOLO 数据集根目录",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="train,val,test 比例，逗号分隔",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于划分数据集）",
    )
    args = parser.parse_args()

    taco_root = Path(args.taco_root)
    coco_ann_path = taco_root / "data" / "annotations.json"
    # 🔑 关键修改：图片根目录就是 data，本身包含 batch_1/... 这些子目录
    images_root = taco_root / "data"

    if not coco_ann_path.exists():
        raise FileNotFoundError(
            f"找不到 COCO 标注文件: {coco_ann_path}\n"
            "请先运行: python scripts/get_taco_dataset.py"
        )
    if not images_root.exists():
        raise FileNotFoundError(
            f"找不到图片根目录: {images_root}\n"
            "请先运行: python scripts/get_taco_dataset.py"
        )

    out_root = Path(args.out_root)
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[convert_taco_to_yolo] 读取 COCO 标注: {coco_ann_path}")
    coco = json.loads(coco_ann_path.read_text(encoding="utf-8"))

    img_idx = build_image_index(coco)
    ann_idx = build_ann_index(coco)

    img_ids = list(img_idx.keys())
    random.seed(args.seed)
    random.shuffle(img_ids)

    r_train, r_val, r_test = [float(x) for x in args.split_ratio.split(",")]
    assert abs(r_train + r_val + r_test - 1.0) < 1e-6, "split-ratio 之和必须为 1"

    n = len(img_ids)
    n_train = int(n * r_train)
    n_val = int(n * r_val)

    train_ids = img_ids[:n_train]
    val_ids = img_ids[n_train : n_train + n_val]
    test_ids = img_ids[n_train + n_val :]

    def get_split_name(img_id):
        if img_id in train_ids:
            return "train"
        elif img_id in val_ids:
            return "val"
        else:
            return "test"

    print(f"[convert_taco_to_yolo] 总图片数: {n}")
    print(f"  train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    # 单类别：waste -> class_id = 0
    class_id = 0

    num_no_ann = 0
    for img_id in tqdm(img_ids, desc="Converting TACO to YOLO"):
        img_info = img_idx[img_id]
        file_name = img_info["file_name"]  # 例如 "batch_1/00001.jpg"
        width, height = img_info["width"], img_info["height"]

        anns = ann_idx.get(img_id, [])
        split = get_split_name(img_id)

        # 🔑 关键修改：直接在 images_root 下拼接 file_name
        src_img_path = images_root / file_name
        if not src_img_path.exists():
            tqdm.write(f"WARNING: 图片不存在，跳过: {src_img_path}")
            continue

        dst_img_path = out_root / "images" / split / src_img_path.name
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img_path, dst_img_path)

        label_name = src_img_path.with_suffix(".txt").name
        dst_label_path = out_root / "labels" / split / label_name

        if not anns:
            num_no_ann += 1
            dst_label_path.touch()
            continue

        yolo_lines = []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, w, h] in pixels
            cx, cy, bw, bh = coco_to_yolo_bbox(bbox, width, height)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dst_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    print()
    print("✅ COCO -> YOLO 转换完成！")
    print(f"   输出目录: {out_root}")
    print(f"   其中无标注图片数量（仅空 txt）: {num_no_ann}")

if __name__ == "__main__":
    main()
