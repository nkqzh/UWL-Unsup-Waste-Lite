docs/SYSTEM.md（RPi5 端侧部署要点）

Raspberry Pi 5（Cortex-A76 2.4GHz，8–16GB RAM 推荐）上安装 64-bit OS。

Python 3.10 + pip install onnxruntime（若官方轮子不可用，参考社区预编译 wheel 或源码编译指引）。

摄像头：CSI/MIPI 或 USB 摄像头，OpenCV 读取；GPIO 控制红外/云台（可用 pigpio）。

推理：infer_onnx.py --onnx best.onnx --source /dev/video0；预热 10 帧后统计 FPS。

若需 INT8：PC 端 QAT/静态量化导出 best-int8.onnx，端侧直接加载。