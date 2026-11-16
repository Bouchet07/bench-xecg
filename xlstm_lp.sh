#!/usr/bin/env bash
set -euo pipefail

if [ $# -ge 1 ] && [ -n "${1:-}" ]; then
    CONFIG_FILE="configs/only_${1}_xlstm.yaml"
else
    CONFIG_FILE="configs/task_config_list_xlstm_lp_4tasks.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found." >&2
    exit 1
fi

echo "=== [1/2] Starting Model Training ($CONFIG_FILE) ==="
python train_all.py --config_file "$CONFIG_FILE"

echo "=== [2/2] Running Benchmark Evaluation ==="
python benchscore.py logs_lightning best_xlstm_lp

echo "=== Pipeline Completed Successfully ==="