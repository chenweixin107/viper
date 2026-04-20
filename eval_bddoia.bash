#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/weixinchen/viper"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

latest_result() {
    local split_dir="$1"
    ls -1t "${split_dir}"/results_*.csv 2>/dev/null | head -n 1
}

TEST_RESULT="${TEST_RESULT:-$(latest_result "${ROOT_DIR}/results/BDDOIA/test")}"

if [[ -z "${TEST_RESULT}" ]]; then
    echo "No test result CSV found under ${ROOT_DIR}/results/BDDOIA/test" >&2
    exit 1
fi

"${PYTHON_BIN}" compute_acc_oia.py --result_path "${TEST_RESULT}" --split test
