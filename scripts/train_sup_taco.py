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
    name = "yolo11s"

    # 这里直接用官方预训练的 yolo11n.pt
    model = YOLO("yolo11s.pt")

    print("[train_sup_taco] 开始训练 YOLOv11s 监督 baseline...")

    results = model.train(
        data="configs/taco_yolo.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs/uwl_taco_sup",
        name="yolo11s",
        device=0,
        cache=True,
        seed=0,
        deterministic=True,
    )

    # Ultralytics 在 train 之后会把 save_dir 挂在 trainer 上
    save_dir = Path(model.trainer.save_dir)
    best_ckpt = save_dir / "weights" / "best.pt"
    last_ckpt = save_dir / "weights" / "last.pt"

    print(f"\n✅ 训练完成。模型和日志保存在： {save_dir}")
    print(f"   最优权重： {best_ckpt}")
    print(f"   最新权重： {last_ckpt}")

if __name__ == "__main__":
    main()