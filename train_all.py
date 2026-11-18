import argparse
import gc
import importlib
import sys

import yaml

# (task_key, default_config_path, train_module_path, is_r_peaks)
TASKS = [
    ('ptb_xl',      'config_defaults/train_ptb_xl_defaults.yaml',         'run_scripts.train_ptb_xl',         False),
    ('cpsc2018',    'config_defaults/train_cpsc2018_defaults.yaml',       'run_scripts.train_cpsc2018',       False),
    ('mit_bih',     'config_defaults/train_mit_bih_defaults.yaml',        'run_scripts.train_mit_bih',        False),
    ('r_peaks',     'config_defaults/train_mit_bih_defaults.yaml',        'run_scripts.train_mit_bih',        True),
    ('survival',    'config_defaults/train_survival_defaults.yaml',       'run_scripts.train_survival',       False),
    ('age',         'config_defaults/train_age_defaults.yaml',            'run_scripts.train_age',            False),
    ('sleep_apnea', 'config_defaults/train_sleep_apnea_defaults.yaml',    'run_scripts.train_sleep_apnea',    False),
    ('lab_test',    'config_defaults/train_blood_test_defaults.yaml',     'run_scripts.train_lab_mimic',      False),
    ('exercise',    'config_defaults/train_high_intensity_defaults.yaml', 'run_scripts.train_r_peak_intense', True),
    ('ppg_af',      'config_defaults/train_ppg_af_defaults.yaml',         'run_scripts.train_ppg_af',         False),
]

TASK_ALIASES = {
    'apnea': 'sleep_apnea',
    'lab': 'lab_test',
}


def parse_task_filter(raw_tasks: list[str] | None) -> set[str] | None:
    """Parse space- and/or comma-separated task names into canonical task keys."""
    if not raw_tasks:
        return None
    valid_keys = {t[0] for t in TASKS}
    selected: set[str] = set()
    for item in raw_tasks:
        for token in item.split(','):
            tok = token.strip().lower()
            if not tok:
                continue
            if tok == 'all':
                return None
            canonical = TASK_ALIASES.get(tok, tok)
            if canonical not in valid_keys:
                valid_str = ', '.join(sorted(valid_keys))
                alias_str = ', '.join(f'{k}->{v}' for k, v in sorted(TASK_ALIASES.items()))
                raise ValueError(
                    f"Unknown task '{token}'. Valid tasks: {valid_str} (aliases: {alias_str})"
                )
            selected.add(canonical)
    return selected or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a model across BenchECG tasks')
    parser.add_argument(
        '--config_file',
        type=str,
        default='configs/task_config_list_lp.yaml',
        help='Path to the task config list YAML file',
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=None,
        help='Optional task filter (space or comma-separated, e.g. --tasks ptb_xl,cpsc2018 or --tasks apnea lab)',
    )
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to train for each task')
    parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb')
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        selected_tasks = parse_task_filter(args.tasks)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict) or not config:
        print(f"Warning: No tasks defined in '{args.config_file}' (file is empty or all entries are commented out).")
        sys.exit(1)

    active_tasks = []
    for task_key, default_cfg, module_path, is_r_peaks in TASKS:
        if selected_tasks is not None and task_key not in selected_tasks:
            continue
        if task_key not in config or not config[task_key]:
            print(f"Skipping {task_key} (not in config list)...")
            continue
        active_tasks.append((task_key, default_cfg, module_path, is_r_peaks, config[task_key]))

    if not active_tasks:
        print(f"Warning: None of the requested tasks matched active entries in '{args.config_file}'.")
        sys.exit(1)

    import torch
    import torch._dynamo
    from bench_xecg.config import parse_config

    torch.set_float32_matmul_precision('medium')
    torch._dynamo.config.cache_size_limit = 64

    for task_key, default_cfg, module_path, is_r_peaks, task_cfg_path in active_tasks:
        train_fn = importlib.import_module(module_path).train
        task_config = parse_config(task_cfg_path, default_cfg)
        task_config.deterministic = False
        if is_r_peaks:
            task_config.r_peaks_detection = True

        for i in range(args.num_runs):
            print(f"=== Running task={task_key} (run {i + 1}/{args.num_runs}) ===")
            train_fn(task_config, wandb=args.wandb_log)
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
