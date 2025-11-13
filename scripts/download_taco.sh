#!/usr/bin/env bash
set -e
mkdir -p data/raw/taco && cd data/raw/taco
# 1) 克隆官方仓库（含注释与下载脚本）
if [ ! -d TACO ]; then git clone https://github.com/pedropro/TACO.git; fi
cd TACO
# 尝试使用官方脚本（若失败请参考 README 手动下载）
python3 scripts/download.py || echo "TACO 官方下载脚本失败，请参考 docs/REFERENCES.md 使用 Kaggle 备选。"
cd ../
# 2) Kaggle 备选（需配置 kaggle.json）
# kaggle datasets download -d kneroma/tacotrashdataset -p . && unzip -q tacotrashdataset.zip