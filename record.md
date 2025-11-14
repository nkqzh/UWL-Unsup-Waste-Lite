(uwl) PS U:\Desktop\QUSTPaper\UnsupWaste-Lite> python scripts/train_sup_taco.py

EarlyStopping: Training stopped early as no improvement observed in last 50 epochs. Best results observed at epoch 43, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

93 epochs completed in 0.466 hours.
Optimizer stripped from runs\uwl_taco_sup\yolo11s_full\weights\last.pt, 19.2MB
Optimizer stripped from runs\uwl_taco_sup\yolo11s_full\weights\best.pt, 19.2MB

Validating runs\uwl_taco_sup\yolo11s_full\weights\best.pt...
Ultralytics 8.3.50 🚀 Python-3.10.19 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
YOLO11s summary (fused): 238 layers, 9,436,020 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 8/8 [00:01<00:00,  4.91it/s]
                   all        225        708      0.287      0.243      0.188      0.147
                 cls_0          4          9      0.416      0.713      0.718      0.689
                 cls_1          1          1          0          0          0          0
                 cls_2          1          1          1          0          0          0
                 cls_3          1          1          0          0          0          0
                 cls_4          5          6          0          0     0.0162     0.0147
                 cls_5         36         47      0.362      0.447      0.349      0.266
                 cls_6         15         19      0.277      0.211     0.0972     0.0632
                 cls_7         32         34      0.343        0.4      0.327      0.259
                 cls_8         12         21      0.251     0.0952     0.0721     0.0453
                 cls_9          4         11          1          0          0          0
                cls_10          1          2          0          0     0.0155    0.00155
                cls_11          3          3      0.507      0.667      0.677      0.627
                cls_12         30         44      0.387      0.523      0.442      0.371
                cls_13          1          3          0          0          0          0
                cls_14         10         13       0.14      0.154       0.15      0.103
                cls_16          4          4      0.143       0.25      0.259      0.208
                cls_17          3          4      0.382          1      0.616      0.446
                cls_18          3          3     0.0758      0.333      0.355      0.345
                cls_19          1          1          1          0          0          0
                cls_20         12         16      0.352        0.5      0.405      0.317
                cls_21         17         17      0.192      0.294      0.216      0.147
                cls_22          1          2          0          0     0.0129    0.00257
                cls_25          2          2          1          0          0          0
                cls_27          7          8      0.135      0.625     0.0955     0.0769
                cls_29         20         25      0.141        0.2     0.0828     0.0719
                cls_30          1          1          0          0    0.00544    0.00489
                cls_31          8         10      0.483        0.1     0.0577     0.0577
                cls_32          1          1          0          0          0          0
                cls_33         12         16      0.168      0.188      0.142      0.117
                cls_34          3          3      0.245      0.762      0.599      0.479
                cls_36         43         61      0.221      0.262      0.238      0.158
                cls_37          1          1      0.685          1      0.995      0.796
                cls_38          1          1      0.162          1      0.249     0.0458
                cls_39         28         39      0.183      0.462      0.232      0.168
                cls_40          8         10      0.194        0.6      0.305      0.268
                cls_42          5          5     0.0345     0.0966     0.0796     0.0673
                cls_43          4          4          0          0          0          0
                cls_44          1          1          1          0          0          0
                cls_45          5          5      0.112      0.134     0.0622     0.0575
                cls_49         10         12     0.0109    0.00455     0.0739     0.0542
                cls_50          9         15     0.0711     0.0667     0.0201     0.0106
                cls_51          6          6      0.028    0.00934      0.379      0.218
                cls_52          3          5          0          0    0.00606    0.00303
                cls_53          2          3      0.935      0.333      0.501      0.451
                cls_54          1          1          1          0     0.0249     0.0174
                cls_55         17         22      0.235      0.364      0.268      0.205
                cls_56          1          1          0          0          0          0
                cls_57          6          8      0.099      0.125     0.0817     0.0647
                cls_58         38         78     0.0965      0.128     0.0511     0.0247
                cls_59         35        102      0.294      0.098      0.101     0.0464
Speed: 0.1ms preprocess, 2.9ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to runs\uwl_taco_sup\yolo11s_full

✅ 训练完成。模型和日志保存在： runs\uwl_taco_sup\yolo11s_full
   最优权重： runs\uwl_taco_sup\yolo11s_full\weights\best.pt
   最新权重： runs\uwl_taco_sup\yolo11s_full\weights\last.pt

(uwl) PS U:\Desktop\QUSTPaper\UnsupWaste-Lite> python scripts/gen_regions_yolo.py
[Stage A'] 使用 YOLO Teacher 处理图像目录: datasets\data\taco_yolo\images\train
[Stage A'] Teacher 权重: runs/uwl_taco_sup/yolo11s_full/weights/best.pt
[Stage A'] 输出 JSONL: artifacts\taco_unsup\regions_yolo.jsonl
[Stage A'] 置信度阈值 conf = 0.2
[Stage A'] 共发现图像 1050 张。

(uwl) PS U:\Desktop\QUSTPaper\UnsupWaste-Lite> python scripts/cluster_regions_clip.py
[Stage B] 加载 CLIP 模型 (ViT-B/16)...
[Stage B] 读取 region proposals 数量: 1050 行
[Stage B] 裁剪得到 patch 数量: 3134
[Stage B] 提取 CLIP 特征...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3134/3134 [00:32<00:00, 95.47it/s] 
[Stage B] Embedding shape: (3134, 512)
[Stage B] PCA 降维到 50 维...
[Stage B] 搜索最佳聚类 K...
  K=6, silhouette=0.0883
  K=8, silhouette=0.0773
  K=10, silhouette=0.0944
  K=12, silhouette=0.0883

[Stage B] 最佳 K = 10 (silhouette=0.0944)
[Stage B] 聚类结果保存到: artifacts\taco_unsup\clip_clusters\cluster_labels_k10.json


(uwl) PS U:\Desktop\QUSTPaper\UnsupWaste-Lite> python scripts/build_pseudo_yolo_from_clusters.py
[Stage C] 读取 region 行数: 1050
[Stage C] 读取 cluster 行数(=框数): 3134
[Stage C] Building YOLO dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1050/1050 [00:01<00:00, 785.31it/s]

✅ Stage C 完成。输出根目录: U:\Desktop\QUSTPaper\UnsupWaste-Lite\datasets\data\taco_unsup_yolo
   共写入伪标签框数: 3134


(uwl) PS U:\Desktop\QUSTPaper\UnsupWaste-Lite> python scripts/train_unsup_taco_k10.py
[Train-Unsup] 加载学生模型: yolo11s.pt
[Train-Unsup] 开始训练 (k=10 clusters)...
YOLO11s summary (fused): 238 layers, 9,416,670 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:01<00:00,  4.68it/s]
                   all        210        567      0.907      0.559      0.613      0.518
             cluster_0         63        101      0.923      0.208      0.269      0.209
             cluster_1         24         32      0.942      0.812      0.833      0.777
             cluster_2         61         93      0.886        0.5      0.578      0.434
             cluster_3         45         55      0.916      0.594      0.631      0.533
             cluster_4         63         98      0.836      0.365      0.425      0.322
             cluster_5         34         40      0.925      0.616       0.72       0.59
             cluster_6         21         28      0.925      0.536       0.55      0.463
             cluster_7         50         60      0.851      0.617      0.699      0.546
             cluster_9         30         60      0.962      0.783       0.81      0.788
Speed: 0.1ms preprocess, 2.7ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to runs\uwl_taco_unsup\yolo11s_k102


(uwl) PS U:\Desktop\QUSTPaper\UnsupWaste-Lite> python scripts/vis_clusters_samples.py
✅ 抽样可视化完成，结果保存在: artifacts\taco_unsup\cluster_vis
