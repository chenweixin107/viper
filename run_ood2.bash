#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

seed=1
model_size="7B"

OOD_DATASETS=(
  "KandLogic_3obj_unseen_colors"
  "KandLogic_3obj_unseen_shapes"
  "MNLogic_XOR_3digit_red"
  "MNLogic_XOR_3digit_rot15"
  "MNMath_Add_3digit_red"
  "MNMath_Add_3digit_rot15"
)

for dataset_name in "${OOD_DATASETS[@]}"; do
  echo "================================"
  echo "Preparing ${dataset_name}"
  echo "================================"

  python create_data.py \
    --data_name "${dataset_name}" \
    --output_csv "${SCRIPT_DIR}/data/${dataset_name}.csv"

  mkdir -p "${SCRIPT_DIR}/results_ood/${dataset_name}"

  echo "Running ${dataset_name}"
  CONFIG_NAMES="${dataset_name}" python main_batch.py \
    > "${SCRIPT_DIR}/results_ood/${dataset_name}/codellama_log_${model_size}_seed_${seed}.txt" 2>&1
done

echo "All OOD experiments completed!"
