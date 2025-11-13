# scripts/train_sup_taco.py
"""
在 TACO YOLO 数据集上训练一个监督 YOLOv11n baseline。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="configs/taco_yolo.yaml",
        help="YOLO 数据配置文件路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸（正方形）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='设备，如 "0", "0,1" 或 "cpu"',
    )
    args = parser.parse_args()

    project = "runs/uwl_taco_sup"
    name = "yolo11n"

    # 这里直接用官方预训练的 yolo11n.pt
    model = YOLO("yolo11n.pt")

    print("[train_sup_taco] 开始训练 YOLOv11n 监督 baseline...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project,
        name=name,
        workers=4,
        cache=True,
        val=True,
        patience=20,  # 早停
        cos_lr=True,
    )

    save_dir = Path(project) / name
    print()
    print("✅ 训练完成。模型和日志保存在：", save_dir.resolve())
    print("   最优权重：", (save_dir / "weights" / "best.pt").resolve())

if __name__ == "__main__":
    main()