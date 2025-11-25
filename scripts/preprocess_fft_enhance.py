#!/usr/bin/env python3
"""Generate FFT-enhanced copies of tabular time-series data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fft_enhancement import FFTEnhancer


def _infer_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    suffix = input_path.suffix or '.csv'
    return input_path.with_name(f"{input_path.stem}_fft{suffix}")


def _select_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> list[str]:
    if columns:
        missing = sorted(set(columns) - set(df.columns))
        if missing:
            raise ValueError(f"Columns not found in dataset: {missing}")
        return list(columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col.lower() != 'time']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Apply FFTEnhancer to numeric columns of a CSV file.')
    parser.add_argument('--input', type=Path, default=Path('dataset/data.csv'), help='Source CSV file.')
    parser.add_argument('--output', type=Path, default=None, help='Output CSV path (defaults to <input>_fft.csv).')
    parser.add_argument('--columns', nargs='*', default=None,
                        help='Columns to enhance. Omit to process all numeric columns except "time".')
    parser.add_argument('--low_freq_ratio', type=float, default=0.25)
    parser.add_argument('--high_freq_ratio', type=float, default=0.7)
    parser.add_argument('--low_freq_boost', type=float, default=1.3)
    parser.add_argument('--mid_freq_boost', type=float, default=1.1)
    parser.add_argument('--high_freq_suppress', type=float, default=0.2)
    parser.add_argument('--cutoff_ratio', type=float, default=0.9)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    output_path = _infer_output_path(input_path, str(args.output) if args.output else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    target_columns = _select_columns(df, args.columns)
    if not target_columns:
        raise ValueError('No columns selected for FFT enhancement.')

    enhancer = FFTEnhancer(
        low_freq_ratio=args.low_freq_ratio,
        high_freq_ratio=args.high_freq_ratio,
        low_freq_boost=args.low_freq_boost,
        mid_freq_boost=args.mid_freq_boost,
        high_freq_suppress=args.high_freq_suppress,
        cutoff_ratio=args.cutoff_ratio,
    )

    print(f"Loaded {len(df)} rows from {input_path}.")
    print(f"Applying FFT enhancement to columns: {target_columns}")

    df_enhanced = df.copy()
    for col in target_columns:
        series = df[col].to_numpy(dtype=np.float64, copy=True)
        series = np.nan_to_num(series)  # 防止 FFT 过程中出现 NaN
        enhanced = enhancer.enhance(series)
        df_enhanced[col] = enhanced

    df_enhanced.to_csv(output_path, index=False)
    print(f"FFT-enhanced data saved to {output_path}")


if __name__ == '__main__':
    main()
