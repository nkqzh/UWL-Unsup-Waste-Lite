# scripts/build_pseudo_yolo_from_clusters.py
"""
Stage C: 使用 Stage A'(regions_yolo.jsonl) + Stage B(cluster_labels_k6.json)
构建一个新的伪标签 YOLO 检测数据集 (k=6 clusters)。

输出目录结构（会自动创建）:
    datasets/data/taco_unsup_yolo/
        images/train/*.jpg
        labels/train/*.txt
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """将像素坐标 (xyxy) 转为 YOLO 格式 (cx, cy, w, h)，归一化到 [0,1]。"""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        type=str,
        default="artifacts/taco_unsup/regions_yolo.jsonl",
        help="Stage A' 生成的候选框文件",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default="artifacts/taco_unsup/clip_clusters/cluster_labels_k6.json",
        help="Stage B 生成的聚类结果文件",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="datasets/data/taco_unsup_yolo",
        help="输出伪标签 YOLO 数据集根目录",
    )
    args = parser.parse_args()

    regions_path = Path(args.regions)
    clusters_path = Path(args.clusters)
    out_root = Path(args.out_root)

    images_out_dir = out_root / "images" / "train"
    labels_out_dir = out_root / "labels" / "train"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 regions_yolo.jsonl
    records = []
    with regions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[Stage C] 读取 region 行数: {len(records)}")

    # cluster_labels_k6.json 一行对应一个框，顺序与裁剪顺序一致
    cluster_lines = []
    with clusters_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cluster_lines.append(json.loads(line))

    print(f"[Stage C] 读取 cluster 行数(=框数): {len(cluster_lines)}")

    # 做一个迭代器，用于逐个为 bbox 取 cluster id
    cluster_iter = iter(cluster_lines)
    total_boxes = 0

    for rec in tqdm(records, desc="[Stage C] Building YOLO dataset"):
        img_path = Path(rec["image"])
        if not img_path.is_file():
            # 若路径是相对路径，尝试从项目根目录补全
            alt = Path(".") / img_path
            if alt.is_file():
                img_path = alt
            else:
                print(f"  ⚠ 找不到图片: {img_path}, 跳过")
                continue

        # 打开图像获取宽高
        with Image.open(img_path) as im:
            w, h = im.size

        # 拷贝图像到新数据集目录（只拷贝一次）
        dst_img_path = images_out_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy(img_path, dst_img_path)

        # 对应的 label 文件
        dst_label_path = labels_out_dir / (img_path.stem + ".txt")
        yolo_lines = []

        for det in rec.get("detections", []):
            xyxy = det["xyxy"]
            x1, y1, x2, y2 = xyxy

            try:
                cluster_item = next(cluster_iter)
            except StopIteration:
                # 正常情况下不会发生，如果发生说明计数不对
                print("  ⚠ cluster_iter 提前耗尽，可能是 Stage B / C 顺序不一致")
                break

            cluster_id = int(cluster_item["cluster"])  # 0 ~ k-1

            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

            # YOLO 标签格式: <class> <cx> <cy> <w> <h>
            yolo_lines.append(f"{cluster_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_boxes += 1

        # 写 label 文件（如果没有框，可以选择不写或写空文件，YOLO 都能处理）
        if yolo_lines:
            with dst_label_path.open("w", encoding="utf-8") as fw:
                fw.write("\n".join(yolo_lines))

    print(f"\n✅ Stage C 完成。输出根目录: {out_root.resolve()}")
    print(f"   共写入伪标签框数: {total_boxes}")

if __name__ == "__main__":
    main()
