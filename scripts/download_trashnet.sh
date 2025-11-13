#!/usr/bin/env bash
set -e
mkdir -p data/raw/trashnet && cd data/raw/trashnet
# GitHub 原仓（分类任务原图，可自标或配合伪标使用）
if [ ! -d trashnet ]; then git clone https://github.com/garythung/trashnet.git; fi
# Kaggle 版本（可含检测标注）：
# kaggle datasets download -d feyzazkefe/trashnet -p . && unzip -q trashnet.zip