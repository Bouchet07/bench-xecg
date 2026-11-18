#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAW_SOURCE="${RAW_SOURCE:-${BENCH_DATA_ROOT:-$HOME/datasets}}"
PACKED_SOURCE="${PACKED_SOURCE:-$RAW_SOURCE/packed_100hz}"

MODEL=""
TASK="all"
CONFIG_FILE=""
EXPERIMENT_NAME=""
STAGE_DATA=1
SKIP_SCORE=0
WANDB_LOG=0
NUM_RUNS=""

usage() {
    cat <<'EOF'
Usage:
  ./run_task.sh --model <mine|xlstm> [--task <task[,task2,...]|all>] [options]
  ./run_task.sh --config <task_list.yaml> [--task <task[,task2,...]|all>] [--exp <name>] [options]
  ./run_task.sh <mine|xlstm> [task[,task2,...]|all]

Models (presets):
  mine    configs/task_config_list_lp_my_model.yaml (or configs/only_<task>.yaml) -> exp: my_model_lp
  xlstm   configs/task_config_list_lp.yaml (or configs/only_<task>_xlstm.yaml)    -> exp: best_xlstm_lp

Tasks:
  age, apnea (sleep_apnea), cpsc2018, exercise, lab (lab_test),
  mit_bih, ppg_af, ptb_xl, r_peaks, survival, all (default: all)
  Multiple tasks can be comma-separated, e.g. --task ptb_xl,cpsc2018,lab

Options:
  -m, --model <name>      Model preset: mine | xlstm
  -c, --config <path>     Custom task-list YAML (overrides --model config)
  -e, --exp <name>        Experiment name for benchscore.py (auto-detected if omitted with --config)
  -t, --task <spec>       Task(s) to run (default: all)
  -n, --num-runs <N>      Number of runs per task (forwarded to train_all.py)
  --wandb                 Enable Weights & Biases logging (--wandb_log)
  --no-stage              Read datasets directly from disk instead of staging to /dev/shm
  --skip-score            Skip running benchscore.py after training
  -h, --help              Show this help message

Environment overrides:
  BENCH_DATA_ROOT         Base datasets directory (default: $HOME/datasets)
  RAW_SOURCE              Raw datasets directory (default: $BENCH_DATA_ROOT)
  PACKED_SOURCE           Packed 100 Hz memmap directory (default: $RAW_SOURCE/packed_100hz)
EOF
}

# Parse flags or positional args (<model> [task])
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)
            MODEL="${2:-}"; shift 2 ;;
        -c|--config)
            CONFIG_FILE="${2:-}"; shift 2 ;;
        -e|--exp|--experiment)
            EXPERIMENT_NAME="${2:-}"; shift 2 ;;
        -t|--task|--tasks)
            TASK="${2:-}"; shift 2 ;;
        -n|--num-runs)
            NUM_RUNS="${2:-}"; shift 2 ;;
        --wandb)
            WANDB_LOG=1; shift ;;
        --no-stage)
            STAGE_DATA=0; shift ;;
        --skip-score)
            SKIP_SCORE=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)
            if [ -z "$MODEL" ] && [ -z "$CONFIG_FILE" ]; then
                MODEL="$1"
            elif [ "$TASK" = "all" ]; then
                TASK="$1"
            else
                echo "Unexpected argument: $1" >&2; usage; exit 1
            fi
            shift ;;
    esac
done

# Split comma-separated tasks into TASK_LIST
IFS=',' read -r -a RAW_TASK_LIST <<< "$TASK"
TASK_LIST=()
for t in "${RAW_TASK_LIST[@]}"; do
    t_trimmed="$(echo "$t" | xargs)"
    [ -n "$t_trimmed" ] && TASK_LIST+=("$t_trimmed")
done
if [ "${#TASK_LIST[@]}" -eq 0 ]; then
    TASK_LIST=("all")
fi

IS_SINGLE_TASK=0
if [ "${#TASK_LIST[@]}" -eq 1 ] && [ "${TASK_LIST[0]}" != "all" ]; then
    IS_SINGLE_TASK=1
fi

PASS_TASK_FILTER=0
if [ -n "$CONFIG_FILE" ]; then
    if [ "${TASK_LIST[0]}" != "all" ]; then
        PASS_TASK_FILTER=1
    fi
else
    case "$MODEL" in
        mine)
            EXPERIMENT_NAME="${EXPERIMENT_NAME:-my_model_lp}"
            if [ "$IS_SINGLE_TASK" -eq 1 ] && [ -f "configs/only_${TASK_LIST[0]}.yaml" ]; then
                CONFIG_FILE="configs/only_${TASK_LIST[0]}.yaml"
            else
                CONFIG_FILE="configs/task_config_list_lp_my_model.yaml"
                if [ "${TASK_LIST[0]}" != "all" ]; then
                    PASS_TASK_FILTER=1
                fi
            fi
            ;;
        xlstm)
            EXPERIMENT_NAME="${EXPERIMENT_NAME:-best_xlstm_lp}"
            if [ "$IS_SINGLE_TASK" -eq 1 ] && [ -f "configs/only_${TASK_LIST[0]}_xlstm.yaml" ]; then
                CONFIG_FILE="configs/only_${TASK_LIST[0]}_xlstm.yaml"
            else
                CONFIG_FILE="configs/task_config_list_lp.yaml"
                if [ "${TASK_LIST[0]}" != "all" ]; then
                    PASS_TASK_FILTER=1
                fi
            fi
            ;;
        *)
            echo "Error: Specify --model <mine|xlstm> or --config <task_list.yaml>." >&2
            usage
            exit 1
            ;;
    esac
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found." >&2
    exit 1
fi

# Auto-detect experiment name from first task config's wandb_group if not specified
if [ -z "$EXPERIMENT_NAME" ] && [ "$SKIP_SCORE" -eq 0 ]; then
    EXPERIMENT_NAME="$(python - "$CONFIG_FILE" <<'PY'
import sys, yaml
try:
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f) or {}
    for path in cfg.values():
        if not path:
            continue
        with open(path) as tf:
            tcfg = yaml.safe_load(tf) or {}
        grp = tcfg.get("wandb_group")
        if grp:
            print(grp)
            break
except Exception:
    pass
PY
)"
fi

# Helper to append unique elements to an array
add_unique() {
    local arr_name="$1"
    shift
    local item existing found
    for item in "$@"; do
        found=0
        eval 'for existing in "${'"$arr_name"'[@]+"${'"$arr_name"'[@]}"}"; do
            if [ "$existing" = "$item" ]; then
                found=1
                break
            fi
        done'
        if [ "$found" -eq 0 ]; then
            eval "$arr_name"'+=("$item")'
        fi
    done
}

# Determine which packed and raw datasets to stage into /dev/shm
NEEDED_DATASETS=()
NEEDED_RAW_DATASETS=()
for t in "${TASK_LIST[@]}"; do
    case "$t" in
        survival)               add_unique NEEDED_DATASETS "code-15" "mimic-iv-ecg" ;;
        age)                    add_unique NEEDED_DATASETS "code-15" "mimic-iv-ecg" "ptb-xl" "cpsc_2018" ;;
        lab|lab_test)           add_unique NEEDED_DATASETS "mimic-iv-ecg" ;;
        ptb_xl)                 add_unique NEEDED_DATASETS "ptb-xl" ;;
        cpsc2018)               add_unique NEEDED_DATASETS "cpsc_2018" ;;
        apnea|sleep_apnea)      add_unique NEEDED_RAW_DATASETS "apnea-ecg-database-1.0.0" ;;
        exercise)               add_unique NEEDED_RAW_DATASETS "ECG_high_intensity_exercise" ;;
        mit_bih|r_peaks)        add_unique NEEDED_RAW_DATASETS "mitdb" ;;
        ppg_af)                 add_unique NEEDED_RAW_DATASETS "deepbeat" ;;
        all)
            add_unique NEEDED_DATASETS "code-15" "mimic-iv-ecg" "ptb-xl" "cpsc_2018"
            add_unique NEEDED_RAW_DATASETS "apnea-ecg-database-1.0.0" "ECG_high_intensity_exercise" "mitdb" "deepbeat"
            ;;
        *)
            echo "Error: Unknown task '$t'." >&2
            usage
            exit 1
            ;;
    esac
done

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
        echo "=== /dev/shm free (${SHM_FREE_KB} KB) < required (${REQUIRED_KB} KB + 2 GB); reading directly from storage ==="
        if [ "${#AVAILABLE_PACKED[@]}" -gt 0 ]; then
            export BENCH_PACKED_DIR="$PACKED_SOURCE"
        fi
    fi
elif [ "${#AVAILABLE_PACKED[@]}" -gt 0 ]; then
    export BENCH_PACKED_DIR="$PACKED_SOURCE"
fi

echo "=== [1/2] Starting Model Training (config=$CONFIG_FILE, task=$TASK) ==="
TRAIN_ARGS=(--config_file "$CONFIG_FILE")
if [ "$PASS_TASK_FILTER" -eq 1 ]; then
    TRAIN_ARGS+=(--tasks "${TASK_LIST[@]}")
fi
if [ -n "$NUM_RUNS" ]; then
    TRAIN_ARGS+=(--num_runs "$NUM_RUNS")
fi
if [ "$WANDB_LOG" -eq 1 ]; then
    TRAIN_ARGS+=(--wandb_log)
fi
python train_all.py "${TRAIN_ARGS[@]}"

if [ "$SKIP_SCORE" -eq 0 ] && [ -n "$EXPERIMENT_NAME" ]; then
    echo "=== [2/2] Running Benchmark Evaluation ($EXPERIMENT_NAME) ==="
    python benchscore.py logs_lightning "$EXPERIMENT_NAME"
elif [ "$SKIP_SCORE" -eq 0 ]; then
    echo "=== [2/2] Skipping benchscore.py (could not infer experiment name; pass --exp <name>) ==="
fi

echo "=== Pipeline Completed Successfully ==="
