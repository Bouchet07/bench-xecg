import torch
import yaml

from bench_xecg.config import parse_config
from run_scripts.train_ptb_xl import train as train_ptb_xl
from run_scripts.train_cpsc2018 import train as train_cpsc2018
from run_scripts.train_survival import train as train_survival
from run_scripts.train_age import train as train_age
from run_scripts.train_sleep_apnea import train as train_sleep_apnea
from run_scripts.train_lab_mimic import train as train_lab_mimic
from run_scripts.train_mit_bih import train as train_mit_bih
from run_scripts.train_r_peak_intense import train as train_r_peak_intense
from run_scripts.train_ppg_af import train as train_ppg_af

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/task_config_list_lp.yaml', help='Path to the config file')
parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to train for each task')
parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()

    # read all config files task config list
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

TASKS = [
    ('ptb_xl',      'config_defaults/train_ptb_xl_defaults.yaml',         train_ptb_xl,         False),
    ('cpsc2018',    'config_defaults/train_cpsc2018_defaults.yaml',       train_cpsc2018,       False),
    ('mit_bih',     'config_defaults/train_mit_bih_defaults.yaml',        train_mit_bih,        False),
    ('r_peaks',     'config_defaults/train_mit_bih_defaults.yaml',        train_mit_bih,        True),
    ('survival',    'config_defaults/train_survival_defaults.yaml',       train_survival,       False),
    ('age',         'config_defaults/train_age_defaults.yaml',            train_age,            False),
    ('sleep_apnea', 'config_defaults/train_sleep_apnea_defaults.yaml',    train_sleep_apnea,    False),
    ('lab_test',    'config_defaults/train_blood_test_defaults.yaml',     train_lab_mimic,      False),
    ('exercise',    'config_defaults/train_high_intensity_defaults.yaml', train_r_peak_intense, True),
    ('ppg_af',      'config_defaults/train_ppg_af_defaults.yaml',         train_ppg_af,         False),
]

for task_key, default_cfg, train_fn, is_r_peaks in TASKS:
    if task_key not in config or not config[task_key]:
        print(f"Skipping {task_key} (not in config list)...")
        continue
    task_config = parse_config(config[task_key], default_cfg)
    task_config.deterministic = False
    if is_r_peaks:
        task_config.r_peaks_detection = True
    for i in range(args.num_runs):
        train_fn(task_config, wandb=args.wandb_log)