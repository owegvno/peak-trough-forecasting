from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.signal import find_peaks

from .config import TARGET_COLUMNS
from .peak_dataset import parse_hourly_csv


@dataclass(frozen=True)
class DatasetWindow:
    seq_days: int
    pred_days: int

    @property
    def seq_len(self) -> int:
        return self.seq_days * 24

    @property
    def pred_len(self) -> int:
        return self.pred_days * 24

    @property
    def total_len(self) -> int:
        return self.seq_len + self.pred_len


def dataset_window_from_name(dataset_name: str) -> DatasetWindow:
    match = re.search(r"_pred(?P<pred>\d+)_seq(?P<seq>\d+)$", dataset_name)
    if not match:
        raise ValueError(f"无法从数据集文件夹名解析窗口参数: {dataset_name}")
    return DatasetWindow(seq_days=int(match.group("seq")), pred_days=int(match.group("pred")))


def sample_csv_for_dataset(
    data_root: str | Path,
    dataset_name: str,
    target_col: str,
    split: str = "训练",
) -> Path:
    window = dataset_window_from_name(dataset_name)
    return (
        Path(data_root)
        / dataset_name
        / target_col
        / f"峰谷预测样本_seq{window.seq_len}_pred{window.pred_len}_{target_col}_{split}.csv"
    )


def turning_csv_for_dataset(data_root: str | Path, dataset_name: str, target_col: str) -> Path:
    return Path(data_root) / dataset_name / "拐点标签" / f"{target_col}拐点.csv"


def read_first_sample(sample_csv: str | Path, target_col: str = "OT") -> dict[str, str]:
    with Path(sample_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row: dict[str, str] | None = None
        for row in reader:
            if first_row is None:
                first_row = row
            if row.get("目标变量") == target_col:
                return row
    if first_row is not None:
        return first_row
    raise ValueError(f"样本文件为空: {sample_csv}")


def sample_input_window_from_sample(
    records: list[dict[str, object]],
    sample_row: dict[str, str],
    seq_len: int,
    pred_len: int = 0,
) -> list[dict[str, object]]:
    forecast_start = datetime.strptime(sample_row["预测起点日期"], "%Y-%m-%d")
    input_start = forecast_start - timedelta(hours=seq_len)
    input_end = forecast_start + timedelta(hours=pred_len)
    window = [
        row
        for row in records
        if isinstance(row["date"], datetime) and input_start <= row["date"] < input_end
    ]
    expected_len = seq_len + pred_len
    if len(window) != expected_len:
        raise ValueError(
            f"样本 {sample_row.get('样本ID', '')} 窗口长度应为 {expected_len}，实际为 {len(window)}"
        )
    return window


def get_series(window: list[dict[str, object]], target_col: str) -> tuple[list[datetime], np.ndarray]:
    dates = [row["date"] for row in window]
    values = np.array([float(row[target_col]) for row in window], dtype=float)
    return dates, values


def setup_chinese_font() -> None:
    font_candidates = [
        Path("/usr/share/fonts/win11/NotoSansSC-Regular.ttf"),
        Path("/usr/share/fonts/win11/msyh.ttc"),
        Path("/usr/share/fonts/win11/msyhbd.ttc"),
        Path("/usr/share/fonts/win11/simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def draw_day_boundaries(ax: plt.Axes, total_len: int, split_at: int | None = None) -> None:
    for x in range(24, total_len + 1, 24):
        ax.axvline(x=x, color="#8f8f8f", linestyle="--", linewidth=0.8, alpha=0.65)
    if split_at is not None and 0 < split_at < total_len:
        ax.axvline(x=split_at, color="#4d4d4d", linestyle="--", linewidth=1.35, alpha=0.9)


def finish_plot(ax: plt.Axes, title: str, total_len: int) -> None:
    ax.set_title(title)
    ax.set_xlabel("点序号")
    ax.set_ylabel("数值")
    ax.set_xlim(1, total_len)
    ax.set_xticks(list(range(24, total_len + 1, 24)))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")


def _float_from_row(row: dict[str, str], key: str) -> float:
    if key not in row or row[key] == "":
        raise KeyError(f"样本缺少字段: {key}")
    return float(row[key])


def _int_from_row(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def daily_peak_trough_points(
    sample_row: dict[str, str],
    target_col: str,
    seq_days: int,
    pred_days: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    peak_x: list[int] = []
    peak_y: list[float] = []
    trough_x: list[int] = []
    trough_y: list[float] = []

    for day in range(1, seq_days + 1):
        offset = (day - 1) * 24
        peak_x.append(offset + _int_from_row(sample_row, f"{target_col}_过去第{day}天_峰值小时") + 1)
        peak_y.append(_float_from_row(sample_row, f"{target_col}_过去第{day}天_最大值"))
        trough_x.append(offset + _int_from_row(sample_row, f"{target_col}_过去第{day}天_谷值小时") + 1)
        trough_y.append(_float_from_row(sample_row, f"{target_col}_过去第{day}天_最小值"))

    for day in range(1, pred_days + 1):
        offset = seq_days * 24 + (day - 1) * 24
        peak_x.append(offset + _int_from_row(sample_row, f"第{day}天_目标峰值小时") + 1)
        peak_y.append(_float_from_row(sample_row, f"第{day}天_目标峰值"))
        trough_x.append(offset + _int_from_row(sample_row, f"第{day}天_目标谷值小时") + 1)
        trough_y.append(_float_from_row(sample_row, f"第{day}天_目标谷值"))

    return (np.array(peak_x), np.array(peak_y)), (np.array(trough_x), np.array(trough_y))


def read_turning_rows(turning_csv: str | Path) -> list[dict[str, str]]:
    with Path(turning_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def turning_points_for_dates(
    turning_rows: list[dict[str, str]],
    dates: list[datetime],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    by_date = {row["日期"]: row for row in turning_rows}
    peak_x: list[int] = []
    peak_y: list[float] = []
    trough_x: list[int] = []
    trough_y: list[float] = []
    for idx, current in enumerate(dates, start=1):
        row = by_date.get(current.strftime("%Y-%m-%d %H:%M"))
        if not row or row.get("有效标记") == "0":
            continue
        turning_type = int(float(row.get("拐点类型", "0")))
        if turning_type == 1:
            peak_x.append(idx)
            peak_y.append(float(row["原始值"]))
        elif turning_type == -1:
            trough_x.append(idx)
            trough_y.append(float(row["原始值"]))
    return (np.array(peak_x), np.array(peak_y)), (np.array(trough_x), np.array(trough_y))


def plot_sample_peaks_troughs(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    sample_csv: str | Path = "数据集/峰谷预测样本_seq96_pred336_训练.csv",
    target_col: str = "OT",
    output_dir: str | Path = "数据集可视化",
    seq_len: int = 96,
    pred_len: int = 0,
    dataset_name: str | None = None,
) -> Path:
    """读取样本 CSV 中第一个指定列样本，画已计算好的每日波峰和波谷。"""
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    sample_row = read_first_sample(sample_csv, target_col)
    window = sample_input_window_from_sample(records, sample_row, seq_len, pred_len)
    _, values = get_series(window, target_col)
    x = np.arange(1, len(values) + 1)
    if pred_len:
        (peak_x, peak_y), (trough_x, trough_y) = daily_peak_trough_points(sample_row, target_col, seq_len // 24, pred_len // 24)
    else:
        peak_indices, _ = find_peaks(values)
        trough_indices, _ = find_peaks(-values)
        peak_x, peak_y = x[peak_indices], values[peak_indices]
        trough_x, trough_y = x[trough_indices], values[trough_indices]

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 7), dpi=160)
    ax.plot(x, values, color="#34495e", linewidth=1.6, label="原始序列")
    ax.scatter(peak_x, peak_y, marker="^", color="#e41a1c", s=50, label="波峰", zorder=3)
    ax.scatter(trough_x, trough_y, marker="v", color="#377eb8", s=50, label="波谷", zorder=3)
    draw_day_boundaries(ax, len(values), split_at=seq_len if pred_len else None)
    title_prefix = f"{dataset_name} " if dataset_name else ""
    finish_plot(ax, f"{title_prefix}{target_col}{sample_row['样本ID']}前{len(values)}个点波峰波谷", len(values))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{target_col}_{sample_row['样本ID']}_波峰波谷.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sample_turning_points(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    sample_csv: str | Path = "数据集/峰谷预测样本_seq96_pred336_训练.csv",
    target_col: str = "OT",
    output_dir: str | Path = "数据集可视化",
    seq_len: int = 96,
    pred_len: int = 0,
    turning_csv: str | Path | None = None,
    dataset_name: str | None = None,
) -> Path:
    """读取样本和已生成拐点标签，画窗口内的峰型/谷型拐点。"""
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    sample_row = read_first_sample(sample_csv, target_col)
    window = sample_input_window_from_sample(records, sample_row, seq_len, pred_len)
    dates, values = get_series(window, target_col)
    x = np.arange(1, len(values) + 1)
    if turning_csv is None:
        peak_turns, _ = find_peaks(values)
        trough_turns, _ = find_peaks(-values)
        peak_x, peak_y = x[peak_turns], values[peak_turns]
        trough_x, trough_y = x[trough_turns], values[trough_turns]
    else:
        (peak_x, peak_y), (trough_x, trough_y) = turning_points_for_dates(read_turning_rows(turning_csv), dates)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 7), dpi=160)
    ax.plot(x, values, color="#34495e", linewidth=1.6, label="原始序列")
    if len(peak_x):
        ax.scatter(peak_x, peak_y, facecolors="none", edgecolors="#ff7f0e", linewidths=1.7, s=82, label="峰值型拐点", zorder=4)
    if len(trough_x):
        ax.scatter(trough_x, trough_y, facecolors="none", edgecolors="#2ca02c", linewidths=1.7, s=82, label="谷值型拐点", zorder=4)
    draw_day_boundaries(ax, len(values), split_at=seq_len if pred_len else None)
    title_prefix = f"{dataset_name} " if dataset_name else ""
    finish_plot(ax, f"{title_prefix}{target_col}{sample_row['样本ID']}前{len(values)}个点拐点", len(values))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{target_col}_{sample_row['样本ID']}_拐点.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dataset_visualizations(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    data_root: str | Path = "数据集",
    dataset_name: str = "ETTH1_pred14_seq4",
    output_root: str | Path = "数据集可视化",
    target_cols: list[str] | None = None,
    split: str = "训练",
) -> dict[str, dict[str, Path]]:
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    window = dataset_window_from_name(dataset_name)
    cols = target_cols or TARGET_COLUMNS
    output_dir = Path(output_root) / dataset_name
    paths: dict[str, dict[str, Path]] = {}
    for col in cols:
        sample_csv = sample_csv_for_dataset(data_root, dataset_name, col, split)
        turning_csv = turning_csv_for_dataset(data_root, dataset_name, col)
        paths[col] = {
            "peaks_troughs": plot_sample_peaks_troughs(
                records,
                sample_csv=sample_csv,
                target_col=col,
                output_dir=output_dir,
                seq_len=window.seq_len,
                pred_len=window.pred_len,
                dataset_name=dataset_name,
            ),
            "turning_points": plot_sample_turning_points(
                records,
                sample_csv=sample_csv,
                target_col=col,
                output_dir=output_dir,
                seq_len=window.seq_len,
                pred_len=window.pred_len,
                turning_csv=turning_csv,
                dataset_name=dataset_name,
            ),
        }
    return paths
