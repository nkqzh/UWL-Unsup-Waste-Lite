# scripts/get_taco_dataset.py
"""
自动下载 TACO 数据集：
1. 将 pedropro/TACO 仓库 clone 到 external/TACO
2. 运行其 download.py，将图片+annotations.json 下载到 external/TACO/data
"""

import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None):
    print(f"[get_taco_dataset] >> {' '.join(cmd)}  (cwd={cwd})")
    subprocess.check_call(cmd, cwd=cwd)

def main():
    root = Path(__file__).resolve().parents[1]
    external_dir = root / "external"
    external_dir.mkdir(exist_ok=True)

    taco_dir = external_dir / "TACO"
    if not taco_dir.exists():
        print("[get_taco_dataset] Cloning pedropro/TACO...")
        run(["git", "clone", "https://github.com/pedropro/TACO", str(taco_dir)])
    else:
        print("[get_taco_dataset] external/TACO 已存在，跳过 clone")

    # 安装 TACO 所需依赖（只要最基本的）
    req_file = taco_dir / "requirements.txt"
    if req_file.exists():
        print("[get_taco_dataset] Installing TACO requirements (this may take a while)...")
        run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        print("[get_taco_dataset] WARNING: requirements.txt 不存在，跳过依赖安装")

    # 调用 TACO 的 download.py 下载数据
    print("[get_taco_dataset] Downloading TACO images & annotations...")
    run([sys.executable, "download.py"], cwd=taco_dir)

    print()
    print("✅ TACO 数据下载完成。默认数据位置：external/TACO/data")
    print("   你现在可以运行 scripts/convert_taco_to_yolo.py 将其转换成 YOLO 格式。")

if __name__ == "__main__":
    main()
