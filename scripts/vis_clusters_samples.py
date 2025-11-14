# scripts/vis_clusters_samples.py
"""
从伪标签数据集中，按 cluster_x 抽样若干张图片，方便人工观察每一类的大致语义。
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def load_regions(regions_path):
    records = []
    with open(regions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_clusters(cluster_path):
    clusters = []
    with open(cluster_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        default="artifacts/taco_unsup/regions_yolo.jsonl",
    )
    parser.add_argument(
        "--clusters",
        default="artifacts/taco_unsup/clip_clusters/cluster_labels_k10.json",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/taco_unsup/cluster_vis",
    )
    parser.add_argument(
        "--samples-per-cluster",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 regions & clusters
    regions = load_regions(args.regions)
    cluster_items = load_clusters(args.clusters)

    # 构造 per-cluster 的 patch 列表（为简单起见，用 image + bbox）
    cluster_patches = defaultdict(list)
    cluster_iter = iter(cluster_items)

    for rec in regions:
        img_path = rec["image"]
        for det in rec.get("detections", []):
            try:
                citem = next(cluster_iter)
            except StopIteration:
                break
            cid = citem["cluster"]
            cluster_patches[cid].append((img_path, det["xyxy"]))

    # 每个 cluster 抽样
    for cid, plist in cluster_patches.items():
        random.shuffle(plist)
        sample = plist[: args.samples_per_cluster]

        cluster_dir = out_dir / f"cluster_{cid}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, xyxy) in enumerate(sample):
            p = Path(img_path)
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            crop = im.crop((x1, y1, x2, y2))

            save_path = cluster_dir / f"{p.stem}_c{cid}_{idx}.jpg"
            crop.save(save_path)

    print(f"✅ 抽样可视化完成，结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
