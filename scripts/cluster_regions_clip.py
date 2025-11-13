# scripts/cluster_regions_clip.py
"""
Stage B: 用 CLIP 对 YOLO Teacher 的候选区域进行视觉聚类
输入:
    artifacts/taco_unsup/regions_yolo.jsonl
输出:
    artifacts/taco_unsup/clip_clusters/*.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import clip

def crop_image(img_path, xyxy):
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = map(int, xyxy)
    return img.crop((x1, y1, x2, y2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        type=str,
        default="artifacts/taco_unsup/regions_yolo.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/taco_unsup/clip_clusters",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    regions_path = Path(args.regions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Stage B] 加载 CLIP 模型 (ViT-B/16)...")
    device = args.device
    clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

    # 读取所有 region proposals
    records = []
    with regions_path.open("r", encoding="utf-8") as fr:
        for line in fr:
            records.append(json.loads(line.strip()))

    print(f"[Stage B] 读取 region proposals 数量: {len(records)} 行")

    crops = []
    img_paths = []
    for rec in records:
        img_path = rec["image"]
        for det in rec["detections"]:
            xyxy = det["xyxy"]
            crops.append(crop_image(img_path, xyxy))
            img_paths.append(img_path)

    print(f"[Stage B] 裁剪得到 patch 数量: {len(crops)}")

    # 提取 CLIP embedding
    all_embeds = []
    print("[Stage B] 提取 CLIP 特征...")
    for img in tqdm(crops):
        img_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_embeds.append(feat.cpu().numpy()[0])

    all_embeds = np.stack(all_embeds)
    print(f"[Stage B] Embedding shape: {all_embeds.shape}")

    # PCA 降维
    print("[Stage B] PCA 降维到 50 维...")
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(all_embeds)

    # 自动评估 K
    Ks = [6, 8, 10, 12]
    best_k = None
    best_score = -1
    scores = {}

    print("[Stage B] 搜索最佳聚类 K...")
    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        scores[k] = score
        print(f"  K={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n[Stage B] 最佳 K = {best_k} (silhouette={best_score:.4f})")

    # 用最佳 K 重新聚类
    kmeans = KMeans(n_clusters=best_k, random_state=0)
    labels = kmeans.fit_predict(reduced)

    # 保存聚类结果
    out_json = out_dir / f"cluster_labels_k{best_k}.json"
    with out_json.open("w", encoding="utf-8") as fw:
        for img_path, label in zip(img_paths, labels):
            fw.write(json.dumps({"image": img_path, "cluster": int(label)}) + "\n")

    print(f"[Stage B] 聚类结果保存到: {out_json}")

if __name__ == "__main__":
    main()
