# UWL: Unsupervised Waste Lite  
轻量级无监督垃圾检测框架（CLIP + Teacher + Student）

---

## 📌 1. 简介

UWL 是一个 **不依赖人工标注** 的垃圾检测框架，只需要输入无标签图像即可自动生成伪标签、聚类类别，并训练一个轻量、高性能的学生检测模型（YOLO11n）。

核心流程：

1. Teacher（YOLO11n） → 候选框  
2. CLIP ViT-B/16 → 特征聚类（自动选择 K）  
3. 伪标签构建 → YOLO11n 学生模型训练  

最终得到一个 **2.58M 参数、6.3GFLOPs、可实时部署** 的垃圾检测器。

---

## 📌 2. 环境安装

```bash
conda create -n uwl python=3.10 -y
conda activate uwl

pip install -r requirements.txt
pip install ultralytics
pip install ftfy regex tqdm scikit-learn
pip install git+https://github.com/openai/CLIP.git
如果需要 GroundingDINO，可按以下方式安装（Windows 避免 C++ 编译）：
pip install groundingdino-py
```

📌 3. 数据集准备

运行：

python scripts/get_taco_dataset.py


转换为 YOLO 格式：

python scripts/convert_taco_to_yolo.py


文件结构（自动生成）：

datasets/
  data/
    taco_yolo/
      images/
      labels/

📌 4. 无监督完整流程（A → B → C）
🔹 Stage A — 生成候选区域
python scripts/gen_regions_yolo.py


输出：

artifacts/taco_unsup/regions_yolo.jsonl

🔹 Stage B — CLIP 聚类
python scripts/cluster_clip.py


输出：

artifacts/taco_unsup/clip_clusters/cluster_labels_k6.json

🔹 Stage C — 构建伪标签 YOLO 数据集
python scripts/build_pseudo_yolo_from_clusters.py


输出：

datasets/data/taco_unsup_yolo/

📌 5. 训练无监督学生模型
python scripts/train_unsup_taco_k6.py


训练结果保存在：

runs/uwl_taco_unsup/yolo11n_k6/

📌 6. 可视化聚类样本
python scripts/vis_clusters_samples.py


输出示例：

cluster_0: 多为透明瓶
cluster_3: 多为大纸箱
cluster_5: 多为塑料袋

📌 7. 性能评估 & 实验复现（E1~E4）
监督 baseline：
python scripts/train_sup_taco.py

K 消融实验：
python scripts/cluster_clip.py --k-list 4 6 8

Teacher vs Student：
python scripts/vis_students_vs_teacher.py

📌 8. 部署（ONNX）
python scripts/export_onnx.py


可用于：

树莓派 5

Jetson Orin / Nano

移动端 NPU

📌 9. 项目结构
UnsupWaste-Lite/
  configs/
  scripts/
  src/
  artifacts/
  datasets/
  runs/

📌 10. 引用格式（论文可直接使用）
@misc{UWL2025,
  title={UWL: Unsupervised Waste Lite},
  author={Your Name},
  year={2025},
  note={Lightweight Unsupervised Waste Detector},
}

📌 11. License

MIT License.

如果你使用了本项目或论文内容，请在文中注明来源（即可）。