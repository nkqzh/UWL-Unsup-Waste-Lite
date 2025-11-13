实验清单与命令速查（docs/EXPERIMENTS.md）

基线一（弱监督/原标）：YOLO11n/s @ TACO → 指标：mAP@0.5:0.95 / F1

基线二（无监督/伪标）：GroundingDINO(+SAM2) → YOLO11n → 对比提升

对照：RT-DETRv2-R18 小模型同配→ 精度-延迟权衡

消融：prompt 集合、阈值、SAM2、增广、蒸馏分支

端侧：ONNX + RPi5（640）延迟、FPS、CPU 利用

已提供命令模板与记录表格，可直接开始跑实验。