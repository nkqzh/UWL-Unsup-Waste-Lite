# scripts/eval_sup_taco.py
"""
对训练好的 YOLO 模型在 TACO YOLO 数据集上做评测。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/uwl_taco_sup/yolo11n/weights/best.pt",
        help="训练好的权重路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/taco_yolo.yaml",
        help="YOLO 数据配置文件",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="评测使用哪个 split（train/val/test）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='设备，如 "0" 或 "cpu"',
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    model = YOLO(str(weights_path))

    print(f"[eval_sup_taco] 在 {args.split} 集上评测：")

    # Ultralytics 使用 data 配置里的 test/val，不过我们可以通过 overrides 改写 split
    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        imgsz=640,
        batch=16,
    )

    print()
    print("✅ 评测完成。主要指标：")
    # metrics 是 BoxResults 对象，里面有很多字段，这里打印常用几个
    try:
        print(f"  mAP50:      {metrics.box.map50:.4f}")
        print(f"  mAP50-95:   {metrics.box.map:.4f}")
        print(f"  precision:  {metrics.box.mp:.4f}")
        print(f"  recall:     {metrics.box.mr:.4f}")
    except Exception:
        print("  原始 metrics 对象：", metrics)

if __name__ == "__main__":
    main()