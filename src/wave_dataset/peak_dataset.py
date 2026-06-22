from __future__ import annotations

import csv
import math
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np

from .config import COLUMN_EN_TO_ZH, DATE_COLUMN, PEAK_BASE_COLUMNS_ZH, PRED_DAYS, SEQ_DAYS, TARGET_COLUMNS


DATE_FORMATS = ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def parse_datetime(value: str) -> datetime:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期时间: {value}")


def parse_hourly_csv(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, object] = {DATE_COLUMN: parse_datetime(str(raw[DATE_COLUMN]))}
            for col in TARGET_COLUMNS:
                row[col] = float(raw[col])
            rows.append(row)
    return rows


def format_day(day: date) -> str:
    return day.isoformat()


def complete_day_records(records: Iterable[dict[str, object]]) -> tuple[OrderedDict[str, list[dict[str, object]]], list[str]]:
    grouped: dict[date, list[dict[str, object]]] = {}
    for row in records:
        current = row[DATE_COLUMN]
        if isinstance(current, str):
            row = dict(row)
            current = parse_datetime(current)
            row[DATE_COLUMN] = current
        if not isinstance(current, datetime):
            raise TypeError("date 字段必须是 datetime 或可解析的字符串")
        grouped.setdefault(current.date(), []).append(row)

    complete: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    discarded: list[str] = []
    for day in sorted(grouped):
        day_rows = sorted(grouped[day], key=lambda item: item[DATE_COLUMN])
        hours = [item[DATE_COLUMN].hour for item in day_rows if isinstance(item[DATE_COLUMN], datetime)]
        if len(day_rows) == 24 and hours == list(range(24)):
            complete[format_day(day)] = day_rows
        else:
            discarded.append(format_day(day))
    return complete, discarded


def build_daily_peak_rows(complete_days: OrderedDict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    daily_rows: list[dict[str, object]] = []
    for day, rows in complete_days.items():
        output: dict[str, object] = {"日期": day}
        for col in TARGET_COLUMNS:
            values = [float(row[col]) for row in rows]
            max_value = max(values)
            min_value = min(values)
            peak_hour = values.index(max_value)
            trough_hour = values.index(min_value)
            output[f"{col}_峰值"] = max_value
            output[f"{col}_峰值小时"] = peak_hour
            output[f"{col}_谷值"] = min_value
            output[f"{col}_谷值小时"] = trough_hour
            output[f"{col}_peak_value"] = max_value
            output[f"{col}_peak_hour"] = peak_hour
            output[f"{col}_trough_value"] = min_value
            output[f"{col}_trough_hour"] = trough_hour
        daily_rows.append(output)
    return daily_rows


def add_months(base: date, months: int) -> date:
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    month_lengths = [31, 29 if is_leap_year(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(year, month, min(base.day, month_lengths[month - 1]))


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def default_split_boundaries(first_day: date) -> tuple[date, date, date, date]:
    train_start = first_day
    train_end = add_months(first_day, 12)
    val_end = add_months(first_day, 16)
    test_end = add_months(first_day, 20)
    return train_start, train_end, val_end, test_end


def assign_split(
    target_start: date,
    pred_days: int,
    train_start: date,
    train_end: date,
    val_end: date,
    test_end: date,
) -> str:
    target_end = target_start + timedelta(days=pred_days)
    if train_start <= target_start and target_end <= train_end:
        return "训练"
    if train_end <= target_start and target_end <= val_end:
        return "验证"
    if val_end <= target_start and target_end <= test_end:
        return "测试"
    return ""


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=0))


def slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    centered_x = x - x.mean()
    denom = float(np.sum(centered_x * centered_x))
    if denom == 0.0:
        return 0.0
    return float(np.sum(centered_x * (y - y.mean())) / denom)


def mode_int(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def baseline_peak_for(daily_rows_by_day: dict[str, dict[str, object]], history_days: list[str], target_col: str) -> float:
    peaks = [float(daily_rows_by_day[day][f"{target_col}_peak_value"]) for day in history_days]
    return weighted_or_mean(peaks)


def baseline_trough_for(daily_rows_by_day: dict[str, dict[str, object]], history_days: list[str], target_col: str) -> float:
    troughs = [float(daily_rows_by_day[day][f"{target_col}_trough_value"]) for day in history_days]
    return weighted_or_mean(troughs)


def weighted_or_mean(values: list[float]) -> float:
    weights = [0.1, 0.2, 0.3, 0.4]
    if len(values) == len(weights):
        return float(sum(value * weight for value, weight in zip(values, weights)))
    return float(mean(values))


def build_features(
    complete_days: OrderedDict[str, list[dict[str, object]]],
    daily_rows_by_day: dict[str, dict[str, object]],
    history_days: list[str],
    target_date: date,
    horizon: int,
) -> dict[str, object]:
    features: dict[str, object] = {
        "日历_预测天数": horizon,
        "日历_星期": target_date.weekday(),
        "日历_月份": target_date.month,
        "日历_年内日序": target_date.timetuple().tm_yday,
        "日历_是否周末": int(target_date.weekday() >= 5),
    }

    for col in TARGET_COLUMNS:
        values: list[float] = []
        day_values: list[list[float]] = []
        for day in history_days:
            current_values = [float(row[col]) for row in complete_days[day]]
            day_values.append(current_values)
            values.extend(current_values)

        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        prefix = f"{col}_过去96小时"
        features[f"{prefix}_均值"] = float(mean(values))
        features[f"{prefix}_标准差"] = safe_std(values)
        features[f"{prefix}_最大值"] = max(values)
        features[f"{prefix}_最小值"] = min(values)
        features[f"{prefix}_首值"] = values[0]
        features[f"{prefix}_末值"] = values[-1]
        features[f"{prefix}_末首差"] = values[-1] - values[0]
        features[f"{prefix}_差分均值"] = float(mean(diffs)) if diffs else 0.0
        features[f"{prefix}_差分标准差"] = safe_std(diffs)
        features[f"{prefix}_差分最大值"] = max(diffs) if diffs else 0.0
        features[f"{prefix}_差分最小值"] = min(diffs) if diffs else 0.0
        features[f"{prefix}_趋势斜率"] = slope(values)

        peaks = [float(daily_rows_by_day[day][f"{col}_peak_value"]) for day in history_days]
        peak_hours = [int(daily_rows_by_day[day][f"{col}_peak_hour"]) for day in history_days]
        troughs = [min(items) for items in day_values]
        gaps = [peak - trough for peak, trough in zip(peaks, troughs)]

        for idx, (day, current_values) in enumerate(zip(history_days, day_values), start=1):
            day_prefix = f"{col}_过去第{idx}天"
            min_value = min(current_values)
            max_value = max(current_values)
            features[f"{day_prefix}_均值"] = float(mean(current_values))
            features[f"{day_prefix}_标准差"] = safe_std(current_values)
            features[f"{day_prefix}_最小值"] = min_value
            features[f"{day_prefix}_最大值"] = max_value
            features[f"{day_prefix}_末值"] = current_values[-1]
            features[f"{day_prefix}_峰值"] = float(daily_rows_by_day[day][f"{col}_peak_value"])
            features[f"{day_prefix}_峰值小时"] = int(daily_rows_by_day[day][f"{col}_peak_hour"])
            features[f"{day_prefix}_谷值"] = min_value
            features[f"{day_prefix}_谷值小时"] = current_values.index(min_value)
            features[f"{day_prefix}_峰谷差"] = max_value - min_value

        weighted_peak = sum(value * weight for value, weight in zip(peaks, [0.1, 0.2, 0.3, 0.4]))
        features[f"{col}_历史峰值_均值4天"] = float(mean(peaks))
        features[f"{col}_历史峰值_标准差4天"] = safe_std(peaks)
        features[f"{col}_历史峰值_最大值4天"] = max(peaks)
        features[f"{col}_历史峰值_最小值4天"] = min(peaks)
        features[f"{col}_历史峰值_最近值"] = peaks[-1]
        features[f"{col}_历史峰值_最近差"] = peaks[-1] - peaks[-2] if len(peaks) >= 2 else 0.0
        features[f"{col}_历史峰值_加权均值4天"] = weighted_peak

        features[f"{col}_历史峰值小时_均值4天"] = float(mean(peak_hours))
        features[f"{col}_历史峰值小时_标准差4天"] = safe_std([float(value) for value in peak_hours])
        features[f"{col}_历史峰值小时_中位数4天"] = float(median(peak_hours))
        features[f"{col}_历史峰值小时_众数4天"] = mode_int(peak_hours)
        features[f"{col}_历史峰值小时_最近值"] = peak_hours[-1]

        features[f"{col}_峰谷差_均值4天"] = float(mean(gaps))
        features[f"{col}_峰谷差_标准差4天"] = safe_std(gaps)
        features[f"{col}_峰谷差_最大值4天"] = max(gaps)
        features[f"{col}_峰谷差_最近值"] = gaps[-1]
        features[f"{col}_峰谷差_最近差"] = gaps[-1] - gaps[-2] if len(gaps) >= 2 else 0.0

    return features


def build_peak_sample_rows(
    complete_days: OrderedDict[str, list[dict[str, object]]],
    daily_rows: list[dict[str, object]],
    seq_days: int = SEQ_DAYS,
    pred_days: int = PRED_DAYS,
    split_boundaries: tuple[date, date, date, date] | None = None,
) -> list[dict[str, object]]:
    day_keys = list(complete_days.keys())
    if len(day_keys) < seq_days + pred_days:
        return []

    daily_rows_by_day = {str(row["日期"]): row for row in daily_rows}
    first_day = datetime.strptime(day_keys[0], "%Y-%m-%d").date()
    actual_split_boundaries = split_boundaries or default_split_boundaries(first_day)

    samples: list[dict[str, object]] = []
    sample_index = 1
    for anchor_idx in range(seq_days, len(day_keys) - pred_days + 1):
        forecast_day_text = day_keys[anchor_idx]
        forecast_day = datetime.strptime(forecast_day_text, "%Y-%m-%d").date()
        history_days = day_keys[anchor_idx - seq_days : anchor_idx]
        split = assign_split(forecast_day, pred_days, *actual_split_boundaries)
        if not split:
            continue
        features_by_horizon: dict[int, dict[str, object]] = {}
        for horizon in range(1, pred_days + 1):
            target_day_text = day_keys[anchor_idx + horizon - 1]
            target_day = datetime.strptime(target_day_text, "%Y-%m-%d").date()
            features_by_horizon[horizon] = build_features(complete_days, daily_rows_by_day, history_days, target_day, horizon)

        for target_col in TARGET_COLUMNS:
            baseline_peak = baseline_peak_for(daily_rows_by_day, history_days, target_col)
            baseline_trough = baseline_trough_for(daily_rows_by_day, history_days, target_col)
            for horizon in range(1, pred_days + 1):
                target_day_text = day_keys[anchor_idx + horizon - 1]
                target_day = datetime.strptime(target_day_text, "%Y-%m-%d").date()
                target_peak = float(daily_rows_by_day[target_day_text][f"{target_col}_peak_value"])
                target_hour = int(daily_rows_by_day[target_day_text][f"{target_col}_peak_hour"])
                target_trough = float(daily_rows_by_day[target_day_text][f"{target_col}_trough_value"])
                target_trough_hour = int(daily_rows_by_day[target_day_text][f"{target_col}_trough_hour"])
                row = {
                    COLUMN_EN_TO_ZH["sample_id"]: f"S{sample_index:06d}",
                    COLUMN_EN_TO_ZH["forecast_start_date"]: forecast_day_text,
                    COLUMN_EN_TO_ZH["target_col"]: target_col,
                    COLUMN_EN_TO_ZH["horizon"]: horizon,
                    COLUMN_EN_TO_ZH["target_date"]: target_day_text,
                    COLUMN_EN_TO_ZH["baseline_peak"]: baseline_peak,
                    COLUMN_EN_TO_ZH["target_peak_value"]: target_peak,
                    COLUMN_EN_TO_ZH["target_peak_residual"]: target_peak - baseline_peak,
                    COLUMN_EN_TO_ZH["target_peak_hour"]: target_hour,
                    COLUMN_EN_TO_ZH["target_trough_value"]: target_trough,
                    COLUMN_EN_TO_ZH["target_trough_residual"]: target_trough - baseline_trough,
                    COLUMN_EN_TO_ZH["target_trough_hour"]: target_trough_hour,
                    COLUMN_EN_TO_ZH["split"]: split,
                }
                row.update(features_by_horizon[horizon])
                samples.append(row)
        sample_index += 1
    return samples


def ensure_no_bad_numbers(rows: list[dict[str, object]]) -> None:
    for row_index, row in enumerate(rows, start=1):
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError(f"第 {row_index} 行字段 {key} 出现非法数值: {value}")


def collect_fieldnames(rows: list[dict[str, object]], leading: list[str] | None = None) -> list[str]:
    names: list[str] = []
    for name in leading or []:
        if name not in names:
            names.append(name)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or collect_fieldnames(rows)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_rows(samples: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    split_col = COLUMN_EN_TO_ZH["split"]
    return {
        "训练": [row for row in samples if row.get(split_col) == "训练"],
        "验证": [row for row in samples if row.get(split_col) == "验证"],
        "测试": [row for row in samples if row.get(split_col) == "测试"],
    }


def resolve_output_dir(output_dir: str | Path, input_csv: str | Path, seq_days: int, pred_days: int) -> Path:
    base = Path(output_dir)
    dataset_name = Path(input_csv).stem.upper()
    return base / f"{dataset_name}_pred{pred_days}_seq{seq_days}"


def variable_sample_fieldnames(sample_fieldnames: list[str], target_col: str, pred_days: int) -> list[str]:
    names = [
        COLUMN_EN_TO_ZH["sample_id"],
        COLUMN_EN_TO_ZH["forecast_start_date"],
        COLUMN_EN_TO_ZH["target_col"],
        COLUMN_EN_TO_ZH["horizon"],
        COLUMN_EN_TO_ZH["baseline_peak"],
        COLUMN_EN_TO_ZH["split"],
    ]

    daily_fields = [
        COLUMN_EN_TO_ZH["target_peak_value"],
        COLUMN_EN_TO_ZH["target_peak_residual"],
        COLUMN_EN_TO_ZH["target_peak_hour"],
        COLUMN_EN_TO_ZH["target_trough_value"],
        COLUMN_EN_TO_ZH["target_trough_residual"],
        COLUMN_EN_TO_ZH["target_trough_hour"],
        "日历_星期",
        "日历_月份",
        "日历_年内日序",
        "日历_是否周末",
    ]
    for horizon in range(1, pred_days + 1):
        for name in daily_fields:
            names.append(f"第{horizon}天_{name}")

    for name in sample_fieldnames:
        if name in names:
            continue
        if name.startswith(f"{target_col}_") and not is_redundant_feature_name(name, target_col):
            names.append(name)
    return names


def is_redundant_feature_name(name: str, target_col: str) -> bool:
    prefix = f"{target_col}_"
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix) :]
    if suffix.startswith("过去第") and (suffix.endswith("_峰值") or suffix.endswith("_谷值")):
        return True
    return suffix in {
        "过去96小时_末值",
        "历史峰值_最大值4天",
        "历史峰值_最近值",
        "历史峰值_加权均值4天",
        "历史峰值小时_最近值",
        "峰谷差_最近值",
    }


def is_protected_output_field(name: str) -> bool:
    protected_base = set(PEAK_BASE_COLUMNS_ZH)
    if name in protected_base:
        return True
    return any(
        marker in name
        for marker in (
            COLUMN_EN_TO_ZH["target_peak_value"],
            COLUMN_EN_TO_ZH["target_peak_residual"],
            COLUMN_EN_TO_ZH["target_peak_hour"],
            COLUMN_EN_TO_ZH["target_trough_value"],
            COLUMN_EN_TO_ZH["target_trough_residual"],
            COLUMN_EN_TO_ZH["target_trough_hour"],
        )
    )


def deduplicate_fieldnames_by_values(rows: list[dict[str, object]], fieldnames: list[str]) -> list[str]:
    seen: dict[tuple[str, ...], str] = {}
    unique: list[str] = []
    for name in fieldnames:
        values = tuple(str(row.get(name, "")) for row in rows)
        if is_protected_output_field(name):
            unique.append(name)
            seen.setdefault(values, name)
            continue
        if values in seen:
            continue
        seen[values] = name
        unique.append(name)
    return unique


def variable_sample_rows(samples: list[dict[str, object]], target_col: str) -> list[dict[str, object]]:
    target_col_name = COLUMN_EN_TO_ZH["target_col"]
    return [row for row in samples if row.get(target_col_name) == target_col]


def to_wide_variable_sample_rows(rows: list[dict[str, object]], target_col: str, pred_days: int) -> list[dict[str, object]]:
    grouped: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    sample_id_col = COLUMN_EN_TO_ZH["sample_id"]
    for row in rows:
        grouped.setdefault(str(row[sample_id_col]), []).append(row)

    wide_rows: list[dict[str, object]] = []
    for group_rows in grouped.values():
        ordered_rows = sorted(group_rows, key=lambda item: int(item[COLUMN_EN_TO_ZH["horizon"]]))
        first = ordered_rows[0]
        wide: dict[str, object] = {
            COLUMN_EN_TO_ZH["sample_id"]: first[COLUMN_EN_TO_ZH["sample_id"]],
            COLUMN_EN_TO_ZH["forecast_start_date"]: first[COLUMN_EN_TO_ZH["forecast_start_date"]],
            COLUMN_EN_TO_ZH["target_col"]: target_col,
            COLUMN_EN_TO_ZH["horizon"]: pred_days,
            COLUMN_EN_TO_ZH["baseline_peak"]: first[COLUMN_EN_TO_ZH["baseline_peak"]],
            COLUMN_EN_TO_ZH["split"]: first[COLUMN_EN_TO_ZH["split"]],
        }

        for row in ordered_rows:
            horizon = int(row[COLUMN_EN_TO_ZH["horizon"]])
            for name in [
                COLUMN_EN_TO_ZH["target_peak_value"],
                COLUMN_EN_TO_ZH["target_peak_residual"],
                COLUMN_EN_TO_ZH["target_peak_hour"],
                COLUMN_EN_TO_ZH["target_trough_value"],
                COLUMN_EN_TO_ZH["target_trough_residual"],
                COLUMN_EN_TO_ZH["target_trough_hour"],
                "日历_星期",
                "日历_月份",
                "日历_年内日序",
                "日历_是否周末",
            ]:
                wide[f"第{horizon}天_{name}"] = row[name]

        for name, value in first.items():
            if name.startswith(f"{target_col}_"):
                wide[name] = value
        wide_rows.append(wide)
    return wide_rows


def write_variable_sample_csvs(output: Path, samples: list[dict[str, object]], seq_days: int, pred_days: int) -> None:
    all_fieldnames = collect_fieldnames(samples, PEAK_BASE_COLUMNS_ZH)
    split_col = COLUMN_EN_TO_ZH["split"]
    suffix_map = {"训练": "训练", "验证": "验证", "测试": "测试"}
    for target_col in TARGET_COLUMNS:
        target_dir = output / target_col
        target_rows = variable_sample_rows(samples, target_col)
        wide_rows = to_wide_variable_sample_rows(target_rows, target_col, pred_days)
        target_fieldnames = deduplicate_fieldnames_by_values(
            wide_rows,
            variable_sample_fieldnames(all_fieldnames, target_col, pred_days),
        )
        base_name = f"峰谷预测样本_seq{seq_days * 24}_pred{pred_days * 24}_{target_col}"
        write_csv(target_dir / f"{base_name}.csv", wide_rows, target_fieldnames)
        for split_name, suffix in suffix_map.items():
            split_rows_for_target = [row for row in wide_rows if row.get(split_col) == split_name]
            write_csv(target_dir / f"{base_name}_{suffix}.csv", split_rows_for_target, target_fieldnames)


def build_peak_dataset(input_csv: str | Path, output_dir: str | Path, seq_days: int = SEQ_DAYS, pred_days: int = PRED_DAYS) -> dict[str, object]:
    records = parse_hourly_csv(input_csv)
    complete_days, discarded_days = complete_day_records(records)
    daily_rows = build_daily_peak_rows(complete_days)
    samples = build_peak_sample_rows(complete_days, daily_rows, seq_days=seq_days, pred_days=pred_days)
    ensure_no_bad_numbers(samples)

    output = resolve_output_dir(output_dir, input_csv, seq_days, pred_days)
    output.mkdir(parents=True, exist_ok=True)

    day_list_rows = [{"日期": day} for day in complete_days.keys()]
    daily_fieldnames = ["日期"]
    for col in TARGET_COLUMNS:
        daily_fieldnames.extend([f"{col}_峰值", f"{col}_峰值小时", f"{col}_谷值", f"{col}_谷值小时"])

    write_csv(output / "完整自然日清单.csv", day_list_rows, ["日期"])
    write_csv(output / "日周期峰值标签.csv", daily_rows, daily_fieldnames)

    write_variable_sample_csvs(output, samples, seq_days, pred_days)
    splits = split_rows(samples)

    old_report_path = output / "峰值数据集生成报告.md"
    if old_report_path.exists():
        old_report_path.unlink()
    report_path = output / "峰谷数据集生成报告.md"
    report_path.write_text(
        build_peak_report(records, complete_days, discarded_days, samples, splits, seq_days, pred_days),
        encoding="utf-8",
    )

    return {
        "raw_rows": len(records),
        "complete_days": len(complete_days),
        "discarded_days": discarded_days,
        "sample_rows": len(samples),
        "eligible_anchor_count": eligible_anchor_count(len(complete_days), seq_days, pred_days),
        "kept_anchor_count": kept_anchor_count(samples),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "output_dir": output,
    }


def eligible_anchor_count(complete_day_count: int, seq_days: int, pred_days: int) -> int:
    return max(0, complete_day_count - seq_days - pred_days + 1)


def kept_anchor_count(samples: list[dict[str, object]]) -> int:
    return len({row[COLUMN_EN_TO_ZH["sample_id"]] for row in samples})


def build_peak_report(
    records: list[dict[str, object]],
    complete_days: OrderedDict[str, list[dict[str, object]]],
    discarded_days: list[str],
    samples: list[dict[str, object]],
    splits: dict[str, list[dict[str, object]]],
    seq_days: int,
    pred_days: int,
) -> str:
    if complete_days:
        first_day = next(iter(complete_days))
        last_day = next(reversed(complete_days))
        train_start, train_end, val_end, test_end = default_split_boundaries(datetime.strptime(first_day, "%Y-%m-%d").date())
        split_text = (
            f"训练区间: [{train_start}, {train_end})\n"
            f"验证区间: [{train_end}, {val_end})\n"
            f"测试区间: [{val_end}, {test_end})"
        )
    else:
        first_day = ""
        last_day = ""
        split_text = "无完整自然日，无法计算切分区间"

    feature_count = 0
    if samples:
        feature_count = len([name for name in samples[0] if name not in PEAK_BASE_COLUMNS_ZH])

    return "\n".join(
        [
            "# 峰谷数据集生成报告",
            "",
            f"- 原始小时行数: {len(records)}",
            f"- 完整自然日数量: {len(complete_days)}",
            f"- 完整自然日范围: {first_day} 至 {last_day}",
            f"- 丢弃日期: {', '.join(discarded_days) if discarded_days else '无'}",
            f"- seq_len: {seq_days * 24} 小时",
            f"- pred_len: {pred_days * 24} 小时",
            f"- 目标列: {', '.join(TARGET_COLUMNS)}",
            f"- 样本总行数: {len(samples)}",
            f"- 可生成锚点数: {eligible_anchor_count(len(complete_days), seq_days, pred_days)}",
            f"- 保留锚点数: {kept_anchor_count(samples)}",
            f"- 跨区间丢弃锚点数: {eligible_anchor_count(len(complete_days), seq_days, pred_days) - kept_anchor_count(samples)}",
            f"- 训练样本行数: {len(splits['训练'])}",
            f"- 验证样本行数: {len(splits['验证'])}",
            f"- 测试样本行数: {len(splits['测试'])}",
            f"- 特征列数量: {feature_count}",
            f"- 最少完整自然日要求: {seq_days + pred_days} 天",
            "- 切分规则: 预测窗口完整落入对应区间才保留该 split；跨区间样本的数据集划分为空。",
            "- 波谷字段: 每个目标日同步输出目标谷值、目标谷值小时、目标谷值残差。",
            "",
            "## 切分区间",
            "",
            split_text,
            "",
            "## 说明",
            "",
            "本阶段只生成训练数据 CSV 和报告，不运行规则基线、LightGBM、XGBoost 或模型评估。",
        ]
    )
