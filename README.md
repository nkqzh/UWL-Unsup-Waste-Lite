# UnsupWaste-Lite


> 无监督垃圾识别/分拣：自监督 + 开放词汇伪标 → 轻量学生检测器训练 → RPi5 ONNX 部署


## 快速开始（10 分钟）


### 1. 创建环境（CUDA 12.x 或 CPU 均可）
conda env create -f environment.yml
conda activate unsupwaste


# 可选：安装 CUDA 对应的 PyTorch（若需 GPU 训练）
# 参考 https://pytorch.org/get-started/locally/

### 2. 下载数据（默认 TACO + TrashNet + MJU-Waste）
bash scripts/download_taco.sh # 首选官方脚本；失败则提示用 Kaggle 备选
bash scripts/download_trashnet.sh
bash scripts/download_mju.sh

### 3. 转换为 YOLO 数据格式并划分
python scripts/convert_to_yolo.py --source data/raw --out data/yolo/taco
python scripts/make_splits.py --root data/yolo/taco --val 0.1 --test 0.1 --seed 2025

### 4. 生成开放词汇伪标（可选，增强/无监督主线）
python scripts/gen_pseudolabels.py \
--images data/yolo/taco/images/train \
--cfg configs/pseudo.yaml \
--out data/yolo/taco_pseudo

### 5. 训练 YOLO11 学生模型
# 使用伪标或原标（TACO 自带标注），二选一：
# A) 伪标：
python scripts/train_yolo.py \
--data configs/dataset_taco.yaml \
--model yolo11n.pt \
--epochs 100 --imgsz 640 --project runs/yolo11_taco_pseudo \
--train-dir data/yolo/taco_pseudo
# B) 原标：
python scripts/train_yolo.py \
--data configs/dataset_taco.yaml \
--model yolo11n.pt \
--epochs 100 --imgsz 640 --project runs/yolo11_taco

### 6. 评测与可视化
python scripts/eval_yolo.py --run runs/yolo11_taco/weights/best.pt --data configs/dataset_taco.yaml

### 7. 导出 ONNX 并在 RPi5 推理
python scripts/export_onnx.py --weights runs/yolo11_taco/weights/best.pt --opset 12 --dynamic
python scripts/infer_onnx.py --onnx runs/yolo11_taco/weights/best.onnx --source demo.jpg