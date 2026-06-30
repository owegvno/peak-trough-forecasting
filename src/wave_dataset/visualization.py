from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.signal import find_peaks

from .config import TARGET_COLUMNS
from .peak_dataset import parse_hourly_csv


DEFAULT_BASELINE_VALUE_PREDICTION_CSV = Path("实验输出/results/baselines/peak_value_baseline_predictions.csv")
DEFAULT_BASELINE_HOUR_PREDICTION_CSV = Path("实验输出/results/baselines/peak_hour_baseline_predictions.csv")
DEFAULT_SELECTED_BASELINE_CSV = Path("实验输出/results/baselines/dataset_with_selected_baselines.csv")
DEFAULT_PEAK_VALUE_BASELINE_NAME = "weighted_mean_last_4"
DEFAULT_PEAK_HOUR_BASELINE_NAME = "mode_last_4"
DEFAULT_PEAK_PLOT_GROUP_NAME = "波峰规则基线"
DEFAULT_SELECTED_BASELINE_PLOT_GROUP_NAME = "逐日最佳基线组合"


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


def read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evenly_sample_sample_ids(
    prediction_rows: list[dict[str, str]],
    split: str = "验证",
    sample_count: int = 6,
) -> list[str]:
    if sample_count <= 0:
        return []

    candidates: dict[str, str] = {}
    for row in prediction_rows:
        if row.get("数据集划分") != split:
            continue
        sample_id = str(row.get("样本ID", ""))
        forecast_start = str(row.get("预测起点日期", ""))
        if not sample_id or not forecast_start:
            continue
        candidates.setdefault(sample_id, forecast_start)

    ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) <= sample_count:
        return [sample_id for sample_id, _ in ordered]

    indices = np.linspace(0, len(ordered) - 1, num=sample_count, dtype=int)
    return [ordered[int(index)][0] for index in indices]


def _optional_filter_matches(row: dict[str, str], column: str, value: str | None) -> bool:
    return value is None or row.get(column) == value


def _prediction_key(row: dict[str, str]) -> tuple[str, str, str, int, str]:
    return (
        str(row.get("样本ID", "")),
        str(row.get("预测起点日期", "")),
        str(row.get("数据集划分", "")),
        int(float(row.get("预测天数", "0"))),
        str(row.get("目标变量", "")),
    )


def merge_baseline_peak_predictions(
    value_prediction_rows: list[dict[str, str]],
    hour_prediction_rows: list[dict[str, str]],
    sample_id: str,
    target_col: str,
    split: str = "验证",
    value_baseline_name: str | None = DEFAULT_PEAK_VALUE_BASELINE_NAME,
    hour_baseline_name: str | None = DEFAULT_PEAK_HOUR_BASELINE_NAME,
    value_baseline_column: str = "baseline_name",
    hour_baseline_column: str = "baseline_name",
    predicted_value_column: str = "baseline_peak_value",
    predicted_hour_column: str = "baseline_peak_hour",
    true_value_column: str = "目标峰值",
    true_hour_column: str = "目标峰值小时",
) -> list[dict[str, Any]]:
    hour_by_key: dict[tuple[str, str, str, int, str], dict[str, str]] = {}
    for row in hour_prediction_rows:
        if row.get("样本ID") != sample_id or row.get("目标变量") != target_col or row.get("数据集划分") != split:
            continue
        if not _optional_filter_matches(row, hour_baseline_column, hour_baseline_name):
            continue
        hour_by_key[_prediction_key(row)] = row

    merged: list[dict[str, Any]] = []
    for value_row in value_prediction_rows:
        if (
            value_row.get("样本ID") != sample_id
            or value_row.get("目标变量") != target_col
            or value_row.get("数据集划分") != split
        ):
            continue
        if not _optional_filter_matches(value_row, value_baseline_column, value_baseline_name):
            continue
        key = _prediction_key(value_row)
        hour_row = hour_by_key.get(key)
        if hour_row is None:
            continue
        merged.append(
            {
                "样本ID": key[0],
                "预测起点日期": key[1],
                "数据集划分": key[2],
                "预测天数": key[3],
                "目标变量": key[4],
                "目标峰值": float(value_row[true_value_column]),
                "目标峰值小时": int(float(hour_row[true_hour_column])),
                "baseline_peak_value": float(value_row[predicted_value_column]),
                "baseline_peak_hour": int(float(hour_row[predicted_hour_column])),
            }
        )
    return sorted(merged, key=lambda row: int(row["预测天数"]))


def prediction_window_from_forecast_start(
    records: list[dict[str, object]],
    forecast_start_text: str,
    pred_len: int,
) -> list[dict[str, object]]:
    forecast_start = datetime.strptime(forecast_start_text, "%Y-%m-%d")
    forecast_end = forecast_start + timedelta(hours=pred_len)
    window = [
        row
        for row in records
        if isinstance(row["date"], datetime) and forecast_start <= row["date"] < forecast_end
    ]
    if len(window) != pred_len:
        raise ValueError(f"预测窗口长度应为 {pred_len}，实际为 {len(window)}")
    return window


def plot_window_from_forecast_start(
    records: list[dict[str, object]],
    forecast_start_text: str,
    history_len: int,
    pred_len: int,
) -> list[dict[str, object]]:
    forecast_start = datetime.strptime(forecast_start_text, "%Y-%m-%d")
    window_start = forecast_start - timedelta(hours=history_len)
    window_end = forecast_start + timedelta(hours=pred_len)
    window = [
        row
        for row in records
        if isinstance(row["date"], datetime) and window_start <= row["date"] < window_end
    ]
    expected_len = history_len + pred_len
    if len(window) != expected_len:
        raise ValueError(f"绘图窗口长度应为 {expected_len}，实际为 {len(window)}")
    return window


def baseline_prediction_plot_dir(
    output_root: str | Path,
    dataset_name: str,
    task_name: str,
    baseline_name: str,
    target_col: str,
) -> Path:
    return Path(output_root) / dataset_name / task_name / baseline_name / target_col


def baseline_peak_plot_dir(
    output_root: str | Path,
    dataset_name: str,
    model_display_name: str,
    plot_group_name: str,
    value_baseline_name: str | None,
    hour_baseline_name: str | None,
    target_col: str,
) -> Path:
    return baseline_prediction_plot_dir(output_root, dataset_name, plot_group_name, model_display_name, target_col)


def baseline_peak_model_display_name(
    value_baseline_name: str | None,
    hour_baseline_name: str | None,
    model_display_name: str | None = None,
) -> str:
    return model_display_name or value_baseline_name or hour_baseline_name or "全部模型"


def unique_prediction_names(rows: list[dict[str, str]], baseline_column: str = "baseline_name") -> list[str]:
    names = sorted({str(row.get(baseline_column, "")) for row in rows if row.get(baseline_column)})
    return names


def _filter_prediction_rows(
    rows: list[dict[str, str]],
    sample_id: str,
    target_col: str,
    split: str,
    baseline_name: str,
    baseline_column: str = "baseline_name",
) -> list[dict[str, str]]:
    filtered = [
        row
        for row in rows
        if row.get("样本ID") == sample_id
        and row.get("目标变量") == target_col
        and row.get("数据集划分") == split
        and row.get(baseline_column) == baseline_name
    ]
    return sorted(filtered, key=lambda row: int(float(row["预测天数"])))


def _true_daily_peak_rows(
    records: list[dict[str, object]],
    forecast_start: str,
    target_col: str,
    pred_days: int,
) -> list[dict[str, Any]]:
    window = prediction_window_from_forecast_start(records, forecast_start, pred_days * 24)
    _, values = get_series(window, target_col)
    true_rows: list[dict[str, Any]] = []
    for horizon in range(1, pred_days + 1):
        start = (horizon - 1) * 24
        day_values = values[start : start + 24]
        peak_hour = int(np.argmax(day_values))
        true_rows.append(
            {
                "预测天数": horizon,
                "目标峰值小时": peak_hour,
                "目标峰值": float(day_values[peak_hour]),
            }
        )
    return true_rows


def _true_peak_by_horizon(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["预测天数"]): row for row in rows}


def _series_value_for_prediction_hour(
    values: np.ndarray,
    horizon: int,
    peak_hour: int,
) -> float:
    if peak_hour < 0 or peak_hour > 23:
        raise ValueError(f"预测小时必须在 0 到 23 之间，发现: {peak_hour}")
    index = (horizon - 1) * 24 + peak_hour
    return float(values[index])


def _peak_prediction_coordinates(rows: list[dict[str, Any]], x_hour_column: str, y_value_column: str) -> tuple[np.ndarray, np.ndarray]:
    return _peak_prediction_coordinates_with_offset(rows, x_hour_column, y_value_column, x_offset=0)


def _peak_prediction_coordinates_with_offset(
    rows: list[dict[str, Any]],
    x_hour_column: str,
    y_value_column: str,
    x_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[int] = []
    y_values: list[float] = []
    for row in rows:
        horizon = int(row["预测天数"])
        hour = int(row[x_hour_column])
        if hour < 0 or hour > 23:
            raise ValueError(f"预测小时必须在 0 到 23 之间，发现: {hour}")
        x_values.append(x_offset + (horizon - 1) * 24 + hour + 1)
        y_values.append(float(row[y_value_column]))
    return np.array(x_values), np.array(y_values)


def _history_daily_peak_coordinates(values: np.ndarray, history_len: int) -> tuple[np.ndarray, np.ndarray]:
    peak_x: list[int] = []
    peak_y: list[float] = []
    for start in range(0, history_len, 24):
        day_values = values[start : start + 24]
        if len(day_values) != 24:
            continue
        peak_hour = int(np.argmax(day_values))
        peak_x.append(start + peak_hour + 1)
        peak_y.append(float(day_values[peak_hour]))
    return np.array(peak_x), np.array(peak_y)


def plot_peak_prediction_rows(
    records: list[dict[str, object]],
    prediction_rows: list[dict[str, Any]],
    target_col: str,
    dataset_name: str,
    output_dir: str | Path,
    task_name: str,
    prediction_label: str,
    filename_suffix: str,
    include_history: bool = True,
) -> Path:
    if not prediction_rows:
        raise ValueError(f"{target_col} 没有可绘制的{task_name}预测记录")

    sample_id = str(prediction_rows[0]["样本ID"])
    forecast_start = str(prediction_rows[0]["预测起点日期"])
    pred_days = max(int(row["预测天数"]) for row in prediction_rows)
    pred_len = pred_days * 24
    history_len = dataset_window_from_name(dataset_name).seq_len if include_history else 0
    window = plot_window_from_forecast_start(records, forecast_start, history_len, pred_len)
    _, values = get_series(window, target_col)
    x = np.arange(1, len(values) + 1)
    history_x, history_y = _history_daily_peak_coordinates(values, history_len)
    true_x, true_y = _peak_prediction_coordinates_with_offset(
        prediction_rows,
        "目标峰值小时",
        "目标峰值",
        x_offset=history_len,
    )
    pred_x, pred_y = _peak_prediction_coordinates_with_offset(
        prediction_rows,
        "baseline_peak_hour",
        "baseline_peak_value",
        x_offset=history_len,
    )

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 7), dpi=160)
    ax.plot(x, values, color="#34495e", linewidth=1.45, label="原始序列")
    if len(history_x):
        ax.scatter(history_x, history_y, marker="^", facecolors="none", edgecolors="#ff7f0e", linewidths=1.7, s=72, label="历史波峰", zorder=4)
    ax.scatter(true_x, true_y, marker="^", facecolors="none", edgecolors="#e41a1c", linewidths=1.7, s=82, label="真实波峰", zorder=4)
    ax.scatter(pred_x, pred_y, marker="^", color="#e41a1c", s=58, label=prediction_label, zorder=5)
    draw_day_boundaries(ax, len(values), split_at=history_len if history_len else None)
    finish_plot(
        ax,
        f"{dataset_name} {sample_id} {target_col} {task_name}",
        len(values),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{target_col}_{sample_id}_{filename_suffix}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_baseline_peak_predictions(
    records: list[dict[str, object]],
    prediction_rows: list[dict[str, Any]],
    target_col: str,
    dataset_name: str,
    output_dir: str | Path,
    value_baseline_name: str | None = DEFAULT_PEAK_VALUE_BASELINE_NAME,
    hour_baseline_name: str | None = DEFAULT_PEAK_HOUR_BASELINE_NAME,
) -> Path:
    return plot_peak_prediction_rows(
        records,
        prediction_rows,
        target_col=target_col,
        dataset_name=dataset_name,
        output_dir=output_dir,
        task_name="波峰小时图",
        prediction_label="波峰预测",
        filename_suffix="波峰预测",
    )


def _plot_single_baseline_rows(
    records: list[dict[str, object]],
    rows: list[dict[str, Any]],
    target_col: str,
    dataset_name: str,
    output_root: str | Path,
    task_name: str,
    baseline_name: str,
    prediction_label: str,
    filename_suffix: str,
) -> Path:
    output_dir = baseline_prediction_plot_dir(output_root, dataset_name, task_name, baseline_name, target_col)
    return plot_peak_prediction_rows(
        records,
        rows,
        target_col=target_col,
        dataset_name=dataset_name,
        output_dir=output_dir,
        task_name=task_name,
        prediction_label=prediction_label,
        filename_suffix=filename_suffix,
    )


def plot_peak_value_prediction_batch(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    value_prediction_csv: str | Path = DEFAULT_BASELINE_VALUE_PREDICTION_CSV,
    output_root: str | Path = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    target_cols: list[str] | None = None,
    split: str = "验证",
    sample_count: int = 6,
    baseline_names: list[str] | None = None,
    baseline_column: str = "baseline_name",
    predicted_value_column: str = "baseline_peak_value",
    true_value_column: str = "目标峰值",
) -> dict[str, dict[str, list[Path]]]:
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    value_rows = read_prediction_rows(value_prediction_csv)
    names = baseline_names or unique_prediction_names(value_rows, baseline_column)
    sample_ids = evenly_sample_sample_ids(value_rows, split=split, sample_count=sample_count)
    cols = target_cols or TARGET_COLUMNS
    output_paths: dict[str, dict[str, list[Path]]] = {name: {col: [] for col in cols} for name in names}

    for baseline_name in names:
        for target_col in cols:
            for sample_id in sample_ids:
                filtered_rows = _filter_prediction_rows(
                    value_rows,
                    sample_id=sample_id,
                    target_col=target_col,
                    split=split,
                    baseline_name=baseline_name,
                    baseline_column=baseline_column,
                )
                if not filtered_rows:
                    continue
                forecast_start = str(filtered_rows[0]["预测起点日期"])
                pred_days = max(int(float(row["预测天数"])) for row in filtered_rows)
                true_rows = _true_peak_by_horizon(_true_daily_peak_rows(records, forecast_start, target_col, pred_days))
                plot_rows: list[dict[str, Any]] = []
                for row in filtered_rows:
                    horizon = int(float(row["预测天数"]))
                    true_row = true_rows[horizon]
                    plot_rows.append(
                        {
                            "样本ID": row["样本ID"],
                            "预测起点日期": forecast_start,
                            "数据集划分": row["数据集划分"],
                            "预测天数": horizon,
                            "目标变量": row["目标变量"],
                            "目标峰值": float(row.get(true_value_column, true_row["目标峰值"])),
                            "目标峰值小时": int(true_row["目标峰值小时"]),
                            "baseline_peak_value": float(row[predicted_value_column]),
                            "baseline_peak_hour": int(true_row["目标峰值小时"]),
                        }
                    )
                output_paths[baseline_name][target_col].append(
                    _plot_single_baseline_rows(
                        records,
                        plot_rows,
                        target_col=target_col,
                        dataset_name=dataset_name,
                        output_root=output_root,
                        task_name="波峰值",
                        baseline_name=baseline_name,
                        prediction_label="预测峰值",
                        filename_suffix="波峰值预测",
                    )
                )
    return output_paths


def plot_peak_hour_prediction_batch(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    hour_prediction_csv: str | Path = DEFAULT_BASELINE_HOUR_PREDICTION_CSV,
    output_root: str | Path = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    target_cols: list[str] | None = None,
    split: str = "验证",
    sample_count: int = 6,
    baseline_names: list[str] | None = None,
    baseline_column: str = "baseline_name",
    predicted_hour_column: str = "baseline_peak_hour",
    true_hour_column: str = "目标峰值小时",
) -> dict[str, dict[str, list[Path]]]:
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    hour_rows = read_prediction_rows(hour_prediction_csv)
    names = baseline_names or unique_prediction_names(hour_rows, baseline_column)
    sample_ids = evenly_sample_sample_ids(hour_rows, split=split, sample_count=sample_count)
    cols = target_cols or TARGET_COLUMNS
    output_paths: dict[str, dict[str, list[Path]]] = {name: {col: [] for col in cols} for name in names}

    for baseline_name in names:
        for target_col in cols:
            for sample_id in sample_ids:
                filtered_rows = _filter_prediction_rows(
                    hour_rows,
                    sample_id=sample_id,
                    target_col=target_col,
                    split=split,
                    baseline_name=baseline_name,
                    baseline_column=baseline_column,
                )
                if not filtered_rows:
                    continue
                forecast_start = str(filtered_rows[0]["预测起点日期"])
                pred_days = max(int(float(row["预测天数"])) for row in filtered_rows)
                window = prediction_window_from_forecast_start(records, forecast_start, pred_days * 24)
                _, values = get_series(window, target_col)
                true_rows = _true_peak_by_horizon(_true_daily_peak_rows(records, forecast_start, target_col, pred_days))
                plot_rows: list[dict[str, Any]] = []
                for row in filtered_rows:
                    horizon = int(float(row["预测天数"]))
                    peak_hour = int(float(row[predicted_hour_column]))
                    true_row = true_rows[horizon]
                    plot_rows.append(
                        {
                            "样本ID": row["样本ID"],
                            "预测起点日期": forecast_start,
                            "数据集划分": row["数据集划分"],
                            "预测天数": horizon,
                            "目标变量": row["目标变量"],
                            "目标峰值": float(true_row["目标峰值"]),
                            "目标峰值小时": int(float(row.get(true_hour_column, true_row["目标峰值小时"]))),
                            "baseline_peak_value": _series_value_for_prediction_hour(values, horizon, peak_hour),
                            "baseline_peak_hour": peak_hour,
                        }
                    )
                output_paths[baseline_name][target_col].append(
                    _plot_single_baseline_rows(
                        records,
                        plot_rows,
                        target_col=target_col,
                        dataset_name=dataset_name,
                        output_root=output_root,
                        task_name="波峰小时",
                        baseline_name=baseline_name,
                        prediction_label="预测峰值小时",
                        filename_suffix="波峰小时预测",
                    )
                )
    return output_paths


def plot_peak_prediction_batch(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    value_prediction_csv: str | Path = DEFAULT_BASELINE_VALUE_PREDICTION_CSV,
    hour_prediction_csv: str | Path = DEFAULT_BASELINE_HOUR_PREDICTION_CSV,
    output_root: str | Path = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    target_cols: list[str] | None = None,
    split: str = "验证",
    sample_count: int = 6,
    value_baseline_name: str | None = DEFAULT_PEAK_VALUE_BASELINE_NAME,
    hour_baseline_name: str | None = DEFAULT_PEAK_HOUR_BASELINE_NAME,
    plot_group_name: str = "波峰小时图",
    value_baseline_column: str = "baseline_name",
    hour_baseline_column: str = "baseline_name",
    predicted_value_column: str = "baseline_peak_value",
    predicted_hour_column: str = "baseline_peak_hour",
    model_display_name: str | None = None,
) -> dict[str, list[Path]]:
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    value_rows = read_prediction_rows(value_prediction_csv)
    hour_rows = read_prediction_rows(hour_prediction_csv)
    sample_ids = evenly_sample_sample_ids(value_rows, split=split, sample_count=sample_count)
    cols = target_cols or TARGET_COLUMNS
    resolved_model_display_name = model_display_name
    if resolved_model_display_name is None and value_baseline_name is not None and hour_baseline_name is not None:
        resolved_model_display_name = f"{value_baseline_name}+{hour_baseline_name}"
    if resolved_model_display_name is None:
        resolved_model_display_name = baseline_peak_model_display_name(
            value_baseline_name,
            hour_baseline_name,
            model_display_name,
        )

    output_paths: dict[str, dict[str, list[Path]]] = {resolved_model_display_name: {col: [] for col in cols}}
    for target_col in cols:
        output_dir = baseline_prediction_plot_dir(output_root, dataset_name, plot_group_name, resolved_model_display_name, target_col)
        for sample_id in sample_ids:
            prediction_rows = merge_baseline_peak_predictions(
                value_rows,
                hour_rows,
                sample_id=sample_id,
                target_col=target_col,
                split=split,
                value_baseline_name=value_baseline_name,
                hour_baseline_name=hour_baseline_name,
                value_baseline_column=value_baseline_column,
                hour_baseline_column=hour_baseline_column,
                predicted_value_column=predicted_value_column,
                predicted_hour_column=predicted_hour_column,
            )
            if not prediction_rows:
                continue
            output_paths[resolved_model_display_name][target_col].append(
                plot_baseline_peak_predictions(
                    records,
                    prediction_rows,
                    target_col=target_col,
                    dataset_name=dataset_name,
                    output_dir=output_dir,
                    value_baseline_name=value_baseline_name,
                    hour_baseline_name=hour_baseline_name,
                )
            )
    return output_paths


def plot_selected_best_baseline_prediction_batch(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    selected_baseline_csv: str | Path = DEFAULT_SELECTED_BASELINE_CSV,
    output_root: str | Path = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    target_cols: list[str] | None = None,
    split: str = "验证",
    sample_count: int = 6,
    plot_group_name: str = DEFAULT_SELECTED_BASELINE_PLOT_GROUP_NAME,
) -> dict[str, list[Path]]:
    records = hourly_csv if isinstance(hourly_csv, list) else parse_hourly_csv(hourly_csv)
    selected_rows = read_prediction_rows(selected_baseline_csv)
    sample_ids = evenly_sample_sample_ids(selected_rows, split=split, sample_count=sample_count)
    cols = target_cols or TARGET_COLUMNS
    output_paths: dict[str, list[Path]] = {col: [] for col in cols}

    for target_col in cols:
        output_dir = Path(output_root) / dataset_name / plot_group_name / target_col
        for sample_id in sample_ids:
            filtered_rows = [
                row
                for row in selected_rows
                if row.get("样本ID") == sample_id
                and row.get("目标变量") == target_col
                and row.get("数据集划分") == split
            ]
            if not filtered_rows:
                continue
            plot_rows: list[dict[str, Any]] = []
            for row in sorted(filtered_rows, key=lambda item: int(float(item["预测天数"]))):
                plot_rows.append(
                    {
                        "样本ID": row["样本ID"],
                        "预测起点日期": row["预测起点日期"],
                        "数据集划分": row["数据集划分"],
                        "预测天数": int(float(row["预测天数"])),
                        "目标变量": row["目标变量"],
                        "目标峰值": float(row["目标峰值"]),
                        "目标峰值小时": int(float(row["目标峰值小时"])),
                        "baseline_peak_value": float(row["baseline_peak"]),
                        "baseline_peak_hour": int(float(row["baseline_peak_hour"])),
                    }
                )
            output_paths[target_col].append(
                plot_peak_prediction_rows(
                    records,
                    plot_rows,
                    target_col=target_col,
                    dataset_name=dataset_name,
                    output_dir=output_dir,
                    task_name=plot_group_name,
                    prediction_label="逐日最佳基线组合预测",
                    filename_suffix="逐日最佳基线组合预测",
                )
            )
    return output_paths


def plot_baseline_peak_prediction_batch(
    hourly_csv: list[dict[str, object]] | str | Path = "ETTh1.csv",
    value_prediction_csv: str | Path = DEFAULT_BASELINE_VALUE_PREDICTION_CSV,
    hour_prediction_csv: str | Path = DEFAULT_BASELINE_HOUR_PREDICTION_CSV,
    output_root: str | Path = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    target_cols: list[str] | None = None,
    split: str = "验证",
    sample_count: int = 6,
    value_baseline_name: str | None = DEFAULT_PEAK_VALUE_BASELINE_NAME,
    hour_baseline_name: str | None = DEFAULT_PEAK_HOUR_BASELINE_NAME,
    plot_group_name: str = "波峰小时图",
    value_baseline_column: str = "baseline_name",
    hour_baseline_column: str = "baseline_name",
    predicted_value_column: str = "baseline_peak_value",
    predicted_hour_column: str = "baseline_peak_hour",
    model_display_name: str | None = None,
) -> dict[str, dict[str, list[Path]]]:
    return plot_peak_prediction_batch(
        hourly_csv=hourly_csv,
        value_prediction_csv=value_prediction_csv,
        hour_prediction_csv=hour_prediction_csv,
        output_root=output_root,
        dataset_name=dataset_name,
        target_cols=target_cols,
        split=split,
        sample_count=sample_count,
        value_baseline_name=value_baseline_name,
        hour_baseline_name=hour_baseline_name,
        plot_group_name=plot_group_name,
        value_baseline_column=value_baseline_column,
        hour_baseline_column=hour_baseline_column,
        predicted_value_column=predicted_value_column,
        predicted_hour_column=predicted_hour_column,
        model_display_name=model_display_name,
    )


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
