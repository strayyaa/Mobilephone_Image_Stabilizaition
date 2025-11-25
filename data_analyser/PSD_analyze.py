"""Angle dataset PSD analyzer.

Reads the custom angle dataset slice (rows 6000-12000 by default), computes the
power spectral density (PSD) for each of the three gyro-like channels, and
reports a power-law fit quality (R^2) as the requested “贴合度”.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_psd(series: np.ndarray, sample_spacing: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
	"""Return single-sided PSD (via rFFT) for a 1-D signal."""

	centered = series - np.mean(series)
	fft_vals = np.fft.rfft(centered)
	n = centered.size
	psd = (np.abs(fft_vals) ** 2) / n
	freqs = np.fft.rfftfreq(n, d=sample_spacing)
	return freqs, psd


def fit_power_law(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
	"""Fit log10(PSD) ~ slope * log10(freq) + intercept for freq>0."""

	mask = freqs > 0
	if not np.any(mask):
		return {"slope": np.nan, "intercept": np.nan, "alpha": np.nan, "r2": np.nan}

	log_f = np.log10(freqs[mask])
	log_psd = np.log10(psd[mask] + 1e-20)
	slope, intercept = np.polyfit(log_f, log_psd, 1)
	fitted = slope * log_f + intercept
	ss_res = np.sum((log_psd - fitted) ** 2)
	ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
	r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
	return {"slope": float(slope), "intercept": float(intercept), "alpha": float(-slope), "r2": float(r2)}


def run(args: argparse.Namespace) -> None:
	data_path = Path(args.data_path)
	if not data_path.exists():
		raise FileNotFoundError(f"数据文件不存在: {data_path}")

	df = pd.read_csv(data_path)
	segment = df.iloc[args.start_idx:args.end_idx].copy()
	missing = [c for c in args.columns if c not in segment.columns]
	if missing:
		raise ValueError(f"缺少所需的列: {missing}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	spectrum_rows = {}
	summary_records = []
	freqs_ref = None

	for col in args.columns:
		freqs, psd = compute_psd(segment[col].to_numpy(dtype=float), args.sample_spacing)
		fit_stats = fit_power_law(freqs, psd)
		freqs_ref = freqs

		spectrum_rows[col] = psd
		summary_records.append(
			{
				"channel": col,
				"n_samples": psd.size * 2 - 2,  # approximate original length
				"power_law_alpha": fit_stats["alpha"],
				"fit_r2": fit_stats["r2"],
			}
		)

		print(
			f"列 {col}: alpha≈{fit_stats['alpha']:.3f}, R^2={fit_stats['r2']:.4f}, "
			f"频率点 {len(freqs)}"
		)

	spectrum_df = pd.DataFrame({"freq": freqs_ref if freqs_ref is not None else []})
	for col, psd_values in spectrum_rows.items():
		spectrum_df[f"psd_{col}"] = psd_values
	spectrum_csv = output_dir / "angle_psd.csv"
	spectrum_df.to_csv(spectrum_csv, index=False)

	summary_df = pd.DataFrame(summary_records)
	summary_csv = output_dir / "angle_psd_summary.csv"
	summary_df.to_csv(summary_csv, index=False)

	plot_path = Path(args.plot_path)
	if freqs_ref is not None and spectrum_rows:
		plt.figure(figsize=(8, 5))
		mask = freqs_ref > 0
		for col, psd_values in spectrum_rows.items():
			plt.loglog(freqs_ref[mask], psd_values[mask], label=col)
		plt.xlabel("Frequency (Hz)")
		plt.ylabel("PSD")
		plt.title("Angle PSD (log-log)")
		plt.grid(True, which="both", linestyle="--", alpha=0.4)
		plt.legend()
		plot_path.parent.mkdir(parents=True, exist_ok=True)
		plt.tight_layout()
		plt.savefig(plot_path, dpi=300)
		plt.close()
		print(f"已保存PSD图像: {plot_path}")

	print(f"已保存PSD结果: {spectrum_csv}")
	print(f"已保存贴合度摘要: {summary_csv}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Angle PSD analyzer")
	parser.add_argument("--data_path", default="dataset/data.csv", help="CSV 数据路径")
	parser.add_argument("--start_idx", type=int, default=6000, help="起始行 (含)")
	parser.add_argument("--end_idx", type=int, default=12000, help="结束行 (不含)")
	parser.add_argument(
		"--columns",
		nargs="+",
		default=["x_angle_1", "y_angle_1", "z_angle_1"],
		help="需要分析的列",
	)
	parser.add_argument("--sample_spacing", type=float, default=1.0, help="采样间隔")
	parser.add_argument("--output_dir", default="data_analyser/outputs", help="结果输出目录")
	parser.add_argument("--plot_path", default="data_analyser/outputs/angle_psd.png", help="PSD 图像输出路径")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
