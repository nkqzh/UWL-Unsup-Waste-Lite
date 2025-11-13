# scripts/gen_regions_gdino.py
"""
Stage A: 使用 GroundingDINO 生成 TACO 的候选垃圾区域（高召回）。
输出: artifacts/taco_unsup/regions_gdino.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image

# === 关键：使用官方 pip 版本 groundingdino==0.1.0 → 无需编译扩展 ===
from groundingdino.util.inference import Model

PROMPTS = (
    "trash. garbage. waste. plastic bottle. metal can. glass bottle. "
    "plastic bag. cardboard. paper. wrapper. packaging. food container."
)

BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.25

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/taco_yolo/images/train",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/taco_unsup/regions_gdino.jsonl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Stage A] 使用 GroundingDINO 处理目录: {img_dir}")
    print(f"[Stage A] 输出: {out_path}")
    print(f"[Stage A] 设备: {args.device}")

    # === GroundingDINO 模型 ===
    print("[Stage A] 加载 GroundingDINO-T (pip 版本)...")
    model = Model(
        model_config_path="src/models/gdino/groundingdino_swinT.cfg.py",
        model_checkpoint_path="src/models/gdino/groundingdino_swinT.pth",
    )
    model.to(device=args.device)

    img_paths = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    print(f"[Stage A] 共发现图像 {len(img_paths)} 张")

    with out_path.open("w", encoding="utf-8") as fw:
        for p in tqdm(img_paths, desc="Detecting"):
            # load
            image = Image.open(p).convert("RGB")

            # detect
            boxes, logits, phrases = model.predict_with_caption(
                image=image,
                caption=PROMPTS,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            dets = []
            W, H = image.size
            for (x1, y1, x2, y2), score, phrase in zip(boxes, logits, phrases):
                dets.append(
                    {
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score),
                        "label": phrase,
                    }
                )

            fw.write(json.dumps({"image": str(p), "detections": dets}, ensure_ascii=False) + "\n")

    print("\n✅ Stage A（GroundingDINO） 完成:", out_path)

if __name__ == "__main__":
    main()
