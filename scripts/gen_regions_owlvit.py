# scripts/gen_regions_owlvit.py
"""
Stage A: 使用 OWL-ViT 在无标注图像上挖掘候选垃圾区域。

输入:
    data/taco_yolo/images/train/*.jpg
输出:
    artifacts/taco_unsup/regions_owlvit.jsonl
    （JSON Lines，每行是一张图片的检测结果）
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor


# 一组针对垃圾场景精心设计的文本提示
PROMPTS = [
    "trash",
    "garbage",
    "waste",
    "rubbish",
    "plastic waste",
    "metal waste",
    "paper waste",
    "plastic bottle",
    "glass bottle",
    "metal can",
    "aluminum can",
    "plastic bag",
    "paper cup",
    "food container",
    "wrapping",
    "packaging",
]

# 检测得分阈值（可以后续在实验里调）
BOX_THRESHOLD = 0.2  # 先设低一点，宁可多一些候选，再靠聚类筛


def detect_one_image(img_path: Path, model, processor, device: str):
    """对单张图像运行 OWL-ViT，返回一组 {xyxy, score, label} 字典。"""
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    w, h = image.size

    # 注意：OWL-ViT 的 text 需要是 per-image 的 list[list[str]]
    inputs = processor(
        text=[PROMPTS],
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = OwlViTForObjectDetection.forward(model, **inputs)

    # 这里用 post_process_object_detection（会有 FutureWarning，但可以用）
    # 若你后面想改成 post_process_grounded_object_detection 也行
    processed = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=torch.tensor([[h, w]]).to(device),
        threshold=BOX_THRESHOLD,
    )[0]

    boxes = processed["boxes"]      # [N, 4], xyxy，像素坐标
    scores = processed["scores"]    # [N]
    labels = processed["labels"]    # [N]，索引到 PROMPTS

    dets = []
    for box, score, label_id in zip(boxes, scores, labels):
        score = float(score.item())
        label_id = int(label_id.item())
        if not (0 <= label_id < len(PROMPTS)):
            continue

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        # 简单做一下合法性裁剪
        x1 = max(0.0, min(x1, w - 1.0))
        x2 = max(0.0, min(x2, w * 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        y2 = max(0.0, min(y2, h * 1.0))
        if x2 <= x1 or y2 <= y1:
            continue

        dets.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "score": score,
                "label": PROMPTS[label_id],
            }
        )

    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/taco_yolo/images/train",
        help="输入图像目录（默认用 TACO YOLO 的 train 子集）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/taco_unsup/regions_owlvit.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='推理设备，如 "cuda:0" 或 "cpu"',
    )
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"找不到图像目录: {img_dir.resolve()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print("[Stage A] 警告: CUDA 不可用，自动切换到 CPU。")
        device = "cpu"

    print(f"[Stage A] 使用 OWL-ViT 处理图像目录: {img_dir}")
    print(f"[Stage A] 输出 JSONL: {out_path}")
    print(f"[Stage A] 使用设备: {device}")
    print(f"[Stage A] 使用 PROMPTS: {PROMPTS}")
    print(f"[Stage A] BOX_THRESHOLD = {BOX_THRESHOLD}")

    # ⚠️ 模型加载一次即可
    print("[Stage A] 加载 OWL-ViT 模型: google/owlvit-base-patch32 ...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()

    image_paths = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    print(f"[Stage A] 共发现图像 {len(image_paths)} 张。")

    with out_path.open("w", encoding="utf-8") as f_out:
        for img_path in tqdm(image_paths, desc="Detecting regions"):
            dets = detect_one_image(img_path, model, processor, device)
            record = {
                "image": str(img_path.as_posix()),
                "detections": dets,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n✅ Stage A 完成。候选区域已保存到:", out_path.resolve())


if __name__ == "__main__":
    main()
