#!/usr/bin/env bash
set -euo pipefail

# Ensure unbuffered Python output
export PYTHONUNBUFFERED=1

echo "=== [1/2] Starting Model Training ==="
python train_all.py --config_file configs/task_config_list_xlstm_lp_4tasks.yaml

echo "=== [2/2] Running Benchmark Evaluation ==="
python benchscore.py logs_lightning best_xlstm_lp

echo "=== Pipeline Completed Successfully ==="