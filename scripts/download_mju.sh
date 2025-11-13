#!/usr/bin/env bash
set -e
mkdir -p data/raw/mju && cd data/raw/mju
# 官方数据脚本或镜像链接（如失效，请根据 README 手动下载再解压至此目录）
if [ ! -d mju-waste ]; then git clone https://github.com/realwecan/mju-waste.git || true; fi
# 也可参考 datasetninja 页面获取直链（可能随时间变动）