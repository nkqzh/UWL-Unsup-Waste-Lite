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
        default=300,
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
    name = "yolo11s-taco"

    # 这里直接用官方预训练的 yolo11s.pt
    model = YOLO("yolo11s.pt")

    print("[train_sup_taco] 开始训练 YOLOv11s 监督 baseline...")

    results = model.train(
        data="configs/taco_yolo.yaml",
        epochs=200,
        patience=50,
        batch=16,
        imgsz=640,
        cache=True,
        device=0,
        workers=4,
        project="runs/uwl_taco_sup",
        name="yolo11s_full",
        seed=0,
        deterministic=True,

        # 优化器 & lr
        optimizer="auto",
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.4,
        close_mosaic=10,
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