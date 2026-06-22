from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .config import COLUMN_EN_TO_ZH, TARGET_COLUMNS, TURNING_COLUMNS_ZH
from .peak_dataset import parse_hourly_csv, write_csv


@dataclass
class TurningPointRow:
    value: float
    smooth_value: float
    turning_type: int
    turning_prominence: float
    turning_threshold: float
    valid_mask: int


def normalize_savgol_window(length: int, window: int, polyorder: int = 2) -> int:
    if window <= 1 or length <= polyorder:
        return 1
    actual = min(window, length if length % 2 == 1 else length - 1)
    if actual % 2 == 0:
        actual -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    return actual if actual >= minimum else 1


def smooth_series(values: list[float], window: int = 5, polyorder: int = 2) -> list[float]:
    if window <= 1 or len(values) < 3:
        return [float(value) for value in values]
    actual_window = normalize_savgol_window(len(values), window, polyorder)
    if actual_window <= 1:
        return [float(value) for value in values]
    actual_polyorder = min(polyorder, actual_window - 1)
    smoothed = savgol_filter(np.array(values, dtype=float), window_length=actual_window, polyorder=actual_polyorder, mode="interp")
    return [float(value) for value in smoothed]


def rolling_mad_threshold(values: list[float], smooth_values: list[float], mad_window: int, multiplier: float) -> list[float]:
    residuals = [abs(value - smooth) for value, smooth in zip(values, smooth_values)]
    radius = max(0, mad_window // 2)
    thresholds: list[float] = []
    for idx in range(len(residuals)):
        left = max(0, idx - radius)
        right = min(len(residuals), idx + radius + 1)
        window_values = residuals[left:right]
        center = median(window_values)
        mad = median([abs(item - center) for item in window_values])
        thresholds.append(float(mad * 1.4826 * multiplier))
    return thresholds


def candidate_prominence(smooth_values: list[float], idx: int, turning_type: int) -> float:
    left = smooth_values[idx - 1]
    center = smooth_values[idx]
    right = smooth_values[idx + 1]
    if turning_type == 1:
        return float(center - max(left, right))
    return float(min(left, right) - center)


def snap_to_original_extreme(values: list[float], idx: int, turning_type: int, radius: int) -> int:
    left = max(1, idx - radius)
    right = min(len(values) - 1, idx + radius + 1)
    candidates = list(range(left, right))
    if not candidates:
        return idx

    local_extremes: list[int] = []
    for pos in candidates:
        if turning_type == 1 and values[pos] >= values[pos - 1] and values[pos] >= values[pos + 1]:
            local_extremes.append(pos)
        if turning_type == -1 and values[pos] <= values[pos - 1] and values[pos] <= values[pos + 1]:
            local_extremes.append(pos)

    search_space = local_extremes or candidates
    if turning_type == 1:
        best_value = max(values[pos] for pos in search_space)
    else:
        best_value = min(values[pos] for pos in search_space)
    best_positions = [pos for pos in search_space if values[pos] == best_value]
    return min(best_positions, key=lambda pos: abs(pos - idx))


def detect_turning_points(
    values: list[float],
    smooth_window: int = 5,
    mad_window: int = 168,
    prominence_multiplier: float = 0.5,
) -> list[TurningPointRow]:
    if not values:
        return []
    smooth_values = smooth_series(values, smooth_window)
    thresholds = rolling_mad_threshold(values, smooth_values, mad_window, prominence_multiplier)
    rows = [
        TurningPointRow(
            value=float(value),
            smooth_value=float(smooth),
            turning_type=0,
            turning_prominence=0.0,
            turning_threshold=float(threshold),
            valid_mask=1,
        )
        for value, smooth, threshold in zip(values, smooth_values, thresholds)
    ]

    if len(values) < 3:
        for row in rows:
            row.valid_mask = 0
        return rows

    rows[0].valid_mask = 0
    rows[-1].valid_mask = 0
    snap_radius = max(1, smooth_window)
    peak_indices, peak_props = find_peaks(np.array(smooth_values, dtype=float), prominence=0, distance=2, width=1)
    trough_indices, trough_props = find_peaks(-np.array(smooth_values, dtype=float), prominence=0, distance=2, width=1)
    candidates: list[tuple[int, int, float]] = []
    for idx, prominence in zip(peak_indices.tolist(), peak_props.get("prominences", np.array([], dtype=float)).tolist()):
        candidates.append((idx, 1, float(prominence)))
    for idx, prominence in zip(trough_indices.tolist(), trough_props.get("prominences", np.array([], dtype=float)).tolist()):
        candidates.append((idx, -1, float(prominence)))

    for idx, turning_type, prominence in candidates:
        if prominence < thresholds[idx]:
            continue
        snapped_idx = snap_to_original_extreme(values, idx, turning_type, snap_radius)
        if prominence > rows[snapped_idx].turning_prominence:
            rows[snapped_idx].turning_type = turning_type
            rows[snapped_idx].turning_prominence = prominence

    return rows


def build_turning_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    smooth_window: int = 5,
    mad_window: int = 168,
    prominence_multiplier: float = 0.5,
) -> dict[str, object]:
    records = parse_hourly_csv(input_csv)
    output = Path(output_dir)
    label_dir = output / "拐点标签"
    label_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    for col in TARGET_COLUMNS:
        values = [float(row[col]) for row in records]
        detected = detect_turning_points(values, smooth_window, mad_window, prominence_multiplier)
        rows: list[dict[str, object]] = []
        for raw_row, turn_row in zip(records, detected):
            current_date = raw_row["date"]
            date_text = current_date.strftime("%Y-%m-%d %H:%M") if hasattr(current_date, "strftime") else str(current_date)
            rows.append(
                {
                    COLUMN_EN_TO_ZH["date"]: date_text,
                    COLUMN_EN_TO_ZH["value"]: turn_row.value,
                    COLUMN_EN_TO_ZH["smooth_value"]: turn_row.smooth_value,
                    COLUMN_EN_TO_ZH["turning_type"]: turn_row.turning_type,
                    COLUMN_EN_TO_ZH["turning_prominence"]: turn_row.turning_prominence,
                    COLUMN_EN_TO_ZH["turning_threshold"]: turn_row.turning_threshold,
                    COLUMN_EN_TO_ZH["valid_mask"]: turn_row.valid_mask,
                }
            )
        write_csv(label_dir / f"{col}拐点.csv", rows, TURNING_COLUMNS_ZH)
        counts[col] = {
            "峰型拐点": sum(1 for row in rows if row[COLUMN_EN_TO_ZH["turning_type"]] == 1),
            "谷型拐点": sum(1 for row in rows if row[COLUMN_EN_TO_ZH["turning_type"]] == -1),
            "非拐点": sum(1 for row in rows if row[COLUMN_EN_TO_ZH["turning_type"]] == 0),
        }

    report_path = output / "拐点数据集生成报告.md"
    report_path.write_text(build_turning_report(len(records), counts, smooth_window, mad_window, prominence_multiplier), encoding="utf-8")
    return {"raw_rows": len(records), "counts": counts, "output_dir": output}


def build_turning_report(
    raw_rows: int,
    counts: dict[str, dict[str, int]],
    smooth_window: int,
    mad_window: int,
    prominence_multiplier: float,
) -> str:
    lines = [
        "# 拐点数据集生成报告",
        "",
        f"- 原始小时行数: {raw_rows}",
        f"- 平滑窗口: {smooth_window}",
        "- 平滑方法: scipy.signal.savgol_filter",
        f"- MAD窗口: {mad_window} 小时",
        f"- 显著性倍数: {prominence_multiplier}",
        "- 拐点类型: 1=峰型, -1=谷型, 0=非拐点",
        "- 用途: 按文档作为任务2标签和任务1辅助分析材料，不默认并入波峰预测样本。",
        "",
        "| 变量 | 峰型拐点 | 谷型拐点 | 非拐点 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for col in TARGET_COLUMNS:
        item = counts[col]
        lines.append(f"| {col} | {item['峰型拐点']} | {item['谷型拐点']} | {item['非拐点']} |")
    return "\n".join(lines)
