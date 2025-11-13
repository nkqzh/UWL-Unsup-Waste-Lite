# scripts/eval_sup_taco.py
"""
Evaluate supervised YOLO11n on TACO YOLO dataset.
用来在 val / test 集上重复评测，方便论文记录指标。
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    # 1) 权重路径（可以改成命令行参数，这里先写死一个示例）
    ckpt = Path("runs/uwl_taco_sup/yolo11n2/weights/best.pt")

    if not ckpt.is_file():
        raise FileNotFoundError(f"找不到权重文件: {ckpt.resolve()}")

    # 2) 加载模型
    model = YOLO(str(ckpt))

    # 3) 在 val 集上评测
    metrics_val = model.val(
        data="configs/taco_yolo.yaml",
        split="val",        # 也可以改成 "test"
        imgsz=640,
        batch=16,
        device=0,
        plots=True,         # 生成 PR 曲线等
        save_json=False,
    )

    print("\n🧪 Validation metrics:")
    print(f"  mAP50      = {metrics_val.box.map50:.4f}")
    print(f"  mAP50-95   = {metrics_val.box.map:.4f}")
    print(f"  save_dir   = {metrics_val.save_dir}")

    # 4) 若想在 test 集上再评测一次，可以取消下面注释：
    # metrics_test = model.val(
    #     data="configs/taco_yolo.yaml",
    #     split="test",
    #     imgsz=640,
    #     batch=16,
    #     device=0,
    #     plots=True,
    # )
    # print("\n🧪 Test metrics:")
    # print(f"  mAP50      = {metrics_test.box.map50:.4f}")
    # print(f"  mAP50-95   = {metrics_test.box.map:.4f}")
    # print(f"  save_dir   = {metrics_test.save_dir}")


if __name__ == "__main__":
    main()
