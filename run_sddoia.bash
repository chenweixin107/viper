#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/weixinchen/viper"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" create_data_oia.py --split test
"${PYTHON_BIN}" create_data_oia.py --split ood

CONFIG_NAMES=SDDOIA_test "${PYTHON_BIN}" main_batch.py
CONFIG_NAMES=SDDOIA_ood "${PYTHON_BIN}" main_batch.py
