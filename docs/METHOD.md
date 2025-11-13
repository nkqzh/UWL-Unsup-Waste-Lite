# Method

本研究提出一个 **完全无监督的轻量级垃圾目标检测框架 UWL（Unsupervised Waste Lite）**。  
该框架摆脱对人工标注的依赖，通过“视觉区域挖掘 → 视觉聚类 → 学生检测模型训练”三阶段流程，在垃圾场景中自动构建语义类别并学习到稳健的检测能力。

整体结构如图所示（论文图 1 建议绘制如下三段流程）：  

Teacher (Detector, e.g., YOLO11)
↓ 生成候选框
CLIP Embedding + K-means
↓ 构建伪标签
Lightweight Student (YOLO11n)


本节将详细介绍三个阶段。

---

# 1. Stage A — Region Mining via Weak Teacher

在无监督环境中，原始图像没有目标位置与类别，因此我们首先借助 **弱监督 Teacher 检测器**（例如 YOLO11n 在 TACO 的少量标注上训练得到）来生成候选区域。

给定原始图像 \(I\)，Teacher 模型输出一组候选框：

\[
B = \{ b_i = (x_1, y_1, x_2, y_2, s_i) \mid i=1,2,\dots,N \}
\]

其中 \(s_i\) 为置信度。我们对所有图像执行：

\[
\mathcal{D}_\text{region} = \bigcup_{I \in \mathcal{D}} \{ (I, b_i) \}
\]

得到一个大型候选区域集合。

---

# 2. Stage B — CLIP Embedding & Unsupervised Visual Clustering

## 2.1 Patch Feature Extraction

对于每个候选区域 patch \(P_i\)，使用 CLIP ViT-B/16 提取 embedding：

\[
z_i = f_\text{CLIP}(P_i) \in \mathbb{R}^{512}
\]

所有 patch 的 embedding 组成矩阵：

\[
Z = [z_1, z_2, \dots, z_n]^T \in \mathbb{R}^{n \times 512}
\]

## 2.2 PCA 降维

为了降低噪声与计算复杂度，对向量执行 PCA：

\[
\tilde{z}_i = W^T (z_i - \mu), \quad \tilde{z}_i \in \mathbb{R}^{d}
\]

实验中设置 \(d = 50\)。

## 2.3 自动选择最佳聚类数 K

采用 K-means 聚类：

\[
c_i = \operatorname*{arg\,min}_k \| \tilde{z}_i - \mu_k \|
\]

选择使 silhouette score 最大的 K：

\[
K^* = \operatorname*{arg\,max}_K \text{Silhouette}(K)
\]

实验中自动选出 \(K=6\)，代表 6 类语义相近的垃圾子类。

---

# 3. Stage C — Pseudo-label Construction & Student Training

## 3.1 伪标签形式化

对于候选框 \(b_i\) 对应的 cluster id \(c_i\)，我们构建 YOLO 格式伪标签：

\[
y_i = (c_i, x, y, w, h)
\]

其中 \(x,y,w,h\) 为归一化坐标。

最终形成无监督伪标签数据集：

\[
\mathcal{D}_\text{pseudo} = \{ (I_j, Y_j) \}
\]

## 3.2 轻量级学生检测器训练

选择 YOLO11n 作为学生模型：

- 参数量仅 2.58M
- FLOPs 6.3G
- 可直接导出 ONNX 用于边缘设备（树莓派 5、Jetson Nano）

训练目标是拟合伪标签：

\[
\theta^* = \operatorname*{arg\,min}_\theta \mathcal{L}_\text{YOLO}(f_\theta(I), Y)
\]

模型在伪标签集上获得：

- mAP50 ≈ 0.856  
- mAP50-95 ≈ 0.729  

说明聚类出来的 6 类语义结构具有较高一致性。

---

# 总结

本方法实现了一个 **端到端完全无监督垃圾检测流程**，不依赖人工标注，只利用 Teacher 的弱监督区域，结合 CLIP 聚类，学到具有语义区分能力的轻量 Student 检测器。方法简单、可扩展、且适合真实落地部署。
