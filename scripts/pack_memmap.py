"""Pack a WFDB ECG dataset into a flat memmap binary (`signals.bin`) + `index.npz`.

Applies the exact preprocessing of `PretrainDataset` / `ECGMIMICDataset`:
1. `wfdb.rdsamp`
2. `np.nan_to_num`
3. Lead mapping (`['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']`, `di->i`, `dii->ii`, `diii->iii`)
4. Optional `nk.ecg_clean`
5. `nk.signal_resample(..., method='FFT')` to `--target-fs` (default 100 Hz)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Optional, Tuple

import neurokit2 as nk
import numpy as np
from tqdm import tqdm
import wfdb

DEFAULT_LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
MAPPINGS = {'di': 'i', 'dii': 'ii', 'diii': 'iii'}


def find_wfdb_records(data_folder: str) -> list[str]:
    """Walk `data_folder` (following symlinks) and return sorted relative record paths without `.hea`."""
    records = []
    for root, _, files in os.walk(data_folder, followlinks=True):
        for file in files:
            if file.endswith('.hea'):
                full_path = os.path.join(root, file[:-4])
                rel_path = os.path.relpath(full_path, data_folder).replace(os.sep, '/')
                records.append(rel_path)
    records.sort()
    return records


def _process_record(args: Tuple[str, str, int, tuple[str, ...], str, bool]) -> Tuple[str, Optional[np.ndarray]]:
    data_folder, rel_record, target_fs, leads, dtype_str, nk_clean = args
    lead_to_idx = {l: i for i, l in enumerate(leads)}
    try:
        signal, info = wfdb.rdsamp(os.path.join(data_folder, rel_record))
        signal = np.nan_to_num(signal)

        mapped = np.zeros((len(signal), len(leads)), dtype=np.float64)
        for col_idx, lead in enumerate(info['sig_name']):
            l = MAPPINGS.get(lead.lower(), lead.lower())
            if l in lead_to_idx:
                col = signal[:, col_idx]
                if nk_clean:
                    col = nk.ecg_clean(col, sampling_rate=info['fs']).copy()
                mapped[:, lead_to_idx[l]] = col

        if target_fs != info['fs']:
            mapped = nk.signal_resample(
                mapped,
                sampling_rate=info['fs'],
                desired_sampling_rate=target_fs,
                method='FFT',
            )

        out = np.ascontiguousarray(mapped, dtype=np.dtype(dtype_str))
        return rel_record, out
    except Exception as exc:
        print(f"[WARN] Failed to process {rel_record}: {exc}", flush=True)
        return rel_record, None


def pack_dataset(
    data_folder: str,
    output_dir: str,
    target_fs: int = 100,
    leads: tuple[str, ...] = tuple(DEFAULT_LEADS),
    dtype_str: str = 'float32',
    nk_clean: bool = False,
    num_workers: int = 16,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_folder} for WFDB .hea records...", flush=True)
    records = find_wfdb_records(data_folder)
    if not records:
        raise FileNotFoundError(f"No .hea records found under {data_folder}")
    print(f"Found {len(records)} records. Packing at {target_fs} Hz ({dtype_str}) -> {out_path}", flush=True)

    bin_path = out_path / 'signals.bin'
    index_path = out_path / 'index.npz'

    saved_records: list[str] = []
    offsets: list[int] = []
    lengths: list[int] = []
    failed = 0
    current_offset = 0

    tasks = (
        (data_folder, rec, target_fs, leads, dtype_str, nk_clean)
        for rec in records
    )

    with open(bin_path, 'wb') as bin_file:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for rec, arr in tqdm(
                executor.map(_process_record, tasks, chunksize=64),
                total=len(records),
                desc=f"Packing {Path(data_folder).name}",
            ):
                if arr is None or arr.shape[0] == 0:
                    failed += 1
                    continue
                bin_file.write(arr.tobytes(order='C'))
                saved_records.append(rec)
                offsets.append(current_offset)
                lengths.append(arr.shape[0])
                current_offset += arr.shape[0]

    np.savez(
        index_path,
        records=np.array(saved_records),
        offsets=np.array(offsets, dtype=np.int64),
        lengths=np.array(lengths, dtype=np.int64),
        dtype=np.array(dtype_str),
        num_leads=np.array(len(leads), dtype=np.int64),
        sampling_freq=np.array(target_fs, dtype=np.int64),
        leads=np.array(leads),
    )

    size_gb = bin_path.stat().st_size / (1024 ** 3)
    print(
        f"Done: saved {len(saved_records)} records ({failed} failed), "
        f"total samples={current_offset}, signals.bin={size_gb:.2f} GB -> {out_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack WFDB ECG dataset into a flat memmap + index.npz")
    parser.add_argument('--data-folder', required=True, help="Root folder containing WFDB .hea/.dat files")
    parser.add_argument('--output-dir', required=True, help="Output directory for signals.bin and index.npz")
    parser.add_argument('--target-fs', type=int, default=100, help="Target sampling frequency in Hz (default: 100)")
    parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32', help="Storage dtype (default: float32)")
    parser.add_argument('--nk-clean', action='store_true', help="Apply neurokit2 ecg_clean before resampling")
    parser.add_argument('--num-workers', type=int, default=16, help="Number of worker processes (default: 16)")
    parser.add_argument(
        '--leads',
        default=','.join(DEFAULT_LEADS),
        help="Comma-separated ordered lead names (default: standard 12 leads)",
    )
    args = parser.parse_args()

    leads = tuple(l.strip().lower() for l in args.leads.split(',') if l.strip())
    pack_dataset(
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        target_fs=args.target_fs,
        leads=leads,
        dtype_str=args.dtype,
        nk_clean=args.nk_clean,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
