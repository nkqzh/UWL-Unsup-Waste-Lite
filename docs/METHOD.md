docs/METHOD.md（方法学摘要）

总体框架：DINOv2/MAE 自监督特征 → GroundingDINO(+OWL-ViT) 文本驱动检测 → SAM2 细化掩码 → 高质量伪标池 → 轻量学生（YOLO11/RT-DETRv2）监督训练 → 迭代自训练（难例回流）→ 导出 ONNX → RPi5 实时推理。

损失：学生检测器沿用原生损失（分类/回归/IoU），蒸馏分支可选（教师为 DINOv2 特征或 OVD logits 对齐）。

复杂度：选用 yolo11n/s 或 rtdetrv2-r18 以达成端侧 30+ FPS 目标（640 输入，RPi5 CPU 参考），INT8 量化后进一步降延迟。

鲁棒性：文本提示扩展、跨数据集混合训练、强增广（color jitter/blur/cutout/mosaic）、一致性训练（不同视图 KL）。

完整公式（信息论一致性、蒸馏 KL、伪标一致性筛选）已在该文档内给出，可直接纳入论文正文。