# scripts/gen_regions_yolo.py
"""
Stage A': 使用已训练好的 YOLO11n baseline 作为 Teacher，
在无标签图像上生成候选框（region proposals）。

输入:
    data/taco_yolo/images/train/*.jpg  （你也可以换成自己收集的垃圾图片目录）
输出:
    artifacts/taco_unsup/regions_yolo.jsonl
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/taco_yolo/images/train",
        help="要生成伪标签的图像目录",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/uwl_taco_sup/yolo11n2/weights/best.pt",
        help="Teacher 模型权重路径（你的监督 YOLO11n）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/taco_unsup/regions_yolo.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="YOLO 检测置信度阈值（可调）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='推理设备，如 "0" 或 "cpu"',
    )
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"找不到图像目录: {img_dir.resolve()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Stage A'] 使用 YOLO Teacher 处理图像目录: {img_dir}")
    print(f"[Stage A'] Teacher 权重: {args.weights}")
    print(f"[Stage A'] 输出 JSONL: {out_path}")
    print(f"[Stage A'] 置信度阈值 conf = {args.conf}")

    # 加载 YOLO Teacher
    model = YOLO(args.weights)

    # 收集图片路径
    image_paths = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    print(f"[Stage A'] 共发现图像 {len(image_paths)} 张。")

    with out_path.open("w", encoding="utf-8") as f_out:
        for img_path in tqdm(image_paths, desc="Detecting with YOLO"):
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                device=args.device,
                verbose=False,
            )

            # Ultralytics 的结果格式：一个列表，每个元素是一个 Results
            if not results:
                dets = []
            else:
                r = results[0]
                dets = []
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                        score = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        dets.append(
                            {
                                "xyxy": [float(v) for v in xyxy],
                                "score": score,
                                "label": cls_id,  # 这里是原始 YOLO 类别，不用于后续聚类，只做参考
                            }
                        )

            record = {
                "image": str(img_path.as_posix()),
                "detections": dets,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n✅ Stage A' 完成。候选区域已保存到:", out_path.resolve())


if __name__ == "__main__":
    main()
