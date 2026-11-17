#!/usr/bin/env bash
set -euo pipefail

PACKED_SOURCE="${PACKED_SOURCE:-/mnt/cephfs/home/bio/dbouchet/datasets/packed_100hz}"
RAW_SOURCE="${RAW_SOURCE:-/mnt/cephfs/home/bio/dbouchet/datasets}"
MODEL=""
TASK="all"
STAGE_DATA=1
NUM_RUNS=""

usage() {
    cat <<'EOF'
Usage:
  run_task.sh --model <mine|xlstm> [--task <task|all>] [--num-runs N] [--no-stage]
  run_task.sh <mine|xlstm> [task|all]

Models:
  mine    Uses configs/only_<task>.yaml (or task_config_list_lp_my_model.yaml) -> benchscore: my_model_lp
  xlstm   Uses configs/only_<task>_xlstm.yaml (or task_config_list_xlstm_lp_4tasks.yaml) -> benchscore: best_xlstm_lp

Tasks:
  age, apnea, cpsc2018, exercise, lab, mit_bih, ppg_af, ptb_xl, r_peaks, survival, all (default: all)
EOF
}

# Parse flags or positional args (<model> [task])
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)
            MODEL="${2:-}"; shift 2 ;;
        -t|--task)
            TASK="${2:-}"; shift 2 ;;
        -n|--num-runs)
            NUM_RUNS="${2:-}"; shift 2 ;;
        --no-stage)
            STAGE_DATA=0; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
            elif [ "$TASK" = "all" ]; then
                TASK="$1"
            else
                echo "Unexpected argument: $1" >&2; usage; exit 1
            fi
            shift ;;
    esac
done

case "$MODEL" in
    mine)
        EXPERIMENT_NAME="my_model_lp"
        if [ "$TASK" = "all" ]; then
            CONFIG_FILE="configs/task_config_list_lp_my_model.yaml"
        else
            CONFIG_FILE="configs/only_${TASK}.yaml"
        fi
        ;;
    xlstm)
        EXPERIMENT_NAME="best_xlstm_lp"
        if [ "$TASK" = "all" ]; then
            CONFIG_FILE="configs/task_config_list_xlstm_lp_4tasks.yaml"
        else
            CONFIG_FILE="configs/only_${TASK}_xlstm.yaml"
        fi
        ;;
    *)
        echo "Error: --model must be 'mine' or 'xlstm' (got '${MODEL:-}')." >&2
        usage
        exit 1
        ;;
esac

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found." >&2
    exit 1
fi

# Determine which packed and raw datasets to stage into /dev/shm for this task
NEEDED_DATASETS=()
NEEDED_RAW_DATASETS=()
case "$TASK" in
    survival)            NEEDED_DATASETS=("code-15" "mimic-iv-ecg") ;;
    age)                 NEEDED_DATASETS=("code-15" "mimic-iv-ecg" "ptb-xl" "cpsc_2018") ;;
    lab)                 NEEDED_DATASETS=("mimic-iv-ecg") ;;
    ptb_xl)              NEEDED_DATASETS=("ptb-xl") ;;
    cpsc2018)            NEEDED_DATASETS=("cpsc_2018") ;;
    apnea)               NEEDED_RAW_DATASETS=("apnea-ecg-database-1.0.0") ;;
    exercise)            NEEDED_RAW_DATASETS=("ECG_high_intensity_exercise") ;;
    mit_bih|r_peaks)     NEEDED_RAW_DATASETS=("mitdb") ;;
    ppg_af)              NEEDED_RAW_DATASETS=("deepbeat") ;;
    all)                 NEEDED_DATASETS=("code-15" "mimic-iv-ecg" "ptb-xl" "cpsc_2018")
                         NEEDED_RAW_DATASETS=("apnea-ecg-database-1.0.0" "ECG_high_intensity_exercise" "mitdb" "deepbeat") ;;
    *)                   NEEDED_DATASETS=("code-15" "mimic-iv-ecg" "ptb-xl" "cpsc_2018") ;;
esac

STAGE_DIR=""
cleanup() {
    if [ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR" ]; then
        echo "=== Cleaning up $STAGE_DIR ==="
        rm -rf "$STAGE_DIR"
    fi
}
trap cleanup EXIT

AVAILABLE_PACKED=()
AVAILABLE_RAW=()
REQUIRED_KB=0

if [ "${#NEEDED_DATASETS[@]}" -gt 0 ] && [ -d "$PACKED_SOURCE" ]; then
    for ds in "${NEEDED_DATASETS[@]}"; do
        if [ -f "$PACKED_SOURCE/$ds/signals.bin" ] && [ -f "$PACKED_SOURCE/$ds/index.npz" ]; then
            AVAILABLE_PACKED+=("$ds")
            ds_kb=$(du -sk "$PACKED_SOURCE/$ds" | awk '{print $1}')
            REQUIRED_KB=$((REQUIRED_KB + ds_kb))
        fi
    done
fi

if [ "${#NEEDED_RAW_DATASETS[@]}" -gt 0 ] && [ -d "$RAW_SOURCE" ]; then
    for ds in "${NEEDED_RAW_DATASETS[@]}"; do
        if [ -d "$RAW_SOURCE/$ds" ]; then
            AVAILABLE_RAW+=("$ds")
            ds_kb=$(du -skL "$RAW_SOURCE/$ds" | awk '{print $1}')
            REQUIRED_KB=$((REQUIRED_KB + ds_kb))
        fi
    done
fi

if [ "$STAGE_DATA" -eq 1 ] && [ $(( ${#AVAILABLE_PACKED[@]} + ${#AVAILABLE_RAW[@]} )) -gt 0 ]; then
    SHM_FREE_KB=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
    # Require 2 GB headroom in /dev/shm
    if [ "$SHM_FREE_KB" -gt $((REQUIRED_KB + 2097152)) ]; then
        STAGE_DIR="/dev/shm/bench_data_${USER:-user}_$$"
        if [ "${#AVAILABLE_PACKED[@]}" -gt 0 ]; then
            mkdir -p "$STAGE_DIR/packed"
            echo "=== Staging packed datasets (${AVAILABLE_PACKED[*]}) to $STAGE_DIR/packed ==="
            for ds in "${AVAILABLE_PACKED[@]}"; do
                cp -r "$PACKED_SOURCE/$ds" "$STAGE_DIR/packed/"
            done
            export BENCH_PACKED_DIR="$STAGE_DIR/packed"
        fi
        if [ "${#AVAILABLE_RAW[@]}" -gt 0 ]; then
            mkdir -p "$STAGE_DIR/raw"
            echo "=== Staging raw datasets (${AVAILABLE_RAW[*]}) to $STAGE_DIR/raw ==="
            for ds in "${AVAILABLE_RAW[@]}"; do
                cp -rL "$RAW_SOURCE/$ds" "$STAGE_DIR/raw/"
            done
            export BENCH_RAW_DIR="$STAGE_DIR/raw"
        fi
    else
        echo "=== /dev/shm free (${SHM_FREE_KB} KB) < required (${REQUIRED_KB} KB + 2 GB); reading from CephFS ==="
        if [ "${#AVAILABLE_PACKED[@]}" -gt 0 ]; then
            export BENCH_PACKED_DIR="$PACKED_SOURCE"
        fi
    fi
elif [ "${#AVAILABLE_PACKED[@]}" -gt 0 ]; then
    export BENCH_PACKED_DIR="$PACKED_SOURCE"
fi

echo "=== [1/2] Starting Model Training ($CONFIG_FILE, model=$MODEL, task=$TASK) ==="
TRAIN_ARGS=(--config_file "$CONFIG_FILE")
if [ -n "$NUM_RUNS" ]; then
    TRAIN_ARGS+=(--num_runs "$NUM_RUNS")
fi
python train_all.py "${TRAIN_ARGS[@]}"

echo "=== [2/2] Running Benchmark Evaluation ($EXPERIMENT_NAME) ==="
python benchscore.py logs_lightning "$EXPERIMENT_NAME"

echo "=== Pipeline Completed Successfully ==="
