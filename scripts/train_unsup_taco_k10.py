# scripts/train_unsup_taco_k10.py
"""
使用伪标签数据集 (k=10 clusters) 训练轻量 YOLO 学生模型
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="configs/taco_unsup_k10.yaml",
        help="YOLO 数据集配置文件",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",  # 或 'yolo11s.pt'，视显存情况
        help="初始模型（预训练权重）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/uwl_taco_unsup",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11s_k10",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )
    args = parser.parse_args()

    print("[Train-Unsup] 加载学生模型:", args.model)
    model = YOLO(args.model)

    print("[Train-Unsup] 开始训练 (k=10 clusters)...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        cache=True,
        seed=0,
        deterministic=True,
    )

    print("\n✅ 无监督学生模型训练完成。"
          f"\n   输出目录: {args.project}/{args.name}")

if __name__ == "__main__":
    main()
