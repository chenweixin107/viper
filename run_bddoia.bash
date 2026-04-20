#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/weixinchen/viper"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/data/common/weixinchen/llm_pc/datasets/BDDOIA}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" create_data_oia.py --task BDDOIA --split test --dataset_root "${DATASET_ROOT}"

CONFIG_NAMES=BDDOIA_test "${PYTHON_BIN}" main_batch.py
