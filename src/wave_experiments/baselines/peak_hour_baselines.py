"""Generate rule-based peak-hour baseline predictions and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from wave_experiments.baselines.load_baseline_data import (
    DEFAULT_DATA_PATH,
    HORIZON_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    load_peak_dataset,
)
from wave_experiments.baselines.visualize_baselines import maybe_plot_peak_baseline_predictions


SAMPLE_ID_COLUMN = "样本ID"
START_DATE_COLUMN = "预测起点日期"
TARGET_PEAK_HOUR_COLUMN = "目标峰值小时"
WEEKDAY_COLUMN = "日历_星期"
BASELINE_PEAK_HOUR_COLUMN = "baseline_peak_hour"
BASELINE_NAME_COLUMN = "baseline_name"

DEFAULT_PREDICTION_PATH = Path("实验输出/results/baselines/peak_hour_baseline_predictions.csv")
DEFAULT_METRICS_PATH = Path("实验输出/results/baselines/peak_hour_baseline_metrics.csv")

BASELINE_NAMES = (
    "mode_last_4",
    "median_last_4",
    "global_mode",
    "weekday_mode",
)

PREDICTION_OUTPUT_COLUMNS = (
    SAMPLE_ID_COLUMN,
    START_DATE_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
    TARGET_PEAK_HOUR_COLUMN,
    BASELINE_NAME_COLUMN,
    BASELINE_PEAK_HOUR_COLUMN,
)

METRIC_COLUMNS = (
    "普通小时误差",
    "环形小时误差",
    "Top-1 accuracy",
    "±1h 命中率",
    "±2h 命中率",
)


def _history_peak_hour_columns(target_variable: str) -> List[str]:
    return [f"{target_variable}_过去第{day}天_峰值小时" for day in range(1, 5)]


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


def _validate_required_prediction_columns(df: pd.DataFrame) -> None:
    required_columns = (
        SAMPLE_ID_COLUMN,
        START_DATE_COLUMN,
        SPLIT_COLUMN,
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        TARGET_PEAK_HOUR_COLUMN,
        WEEKDAY_COLUMN,
    )
    missing = _missing_columns(df, required_columns)
    if missing:
        raise ValueError(f"输入数据缺少生成峰值小时基线所需字段：{', '.join(missing)}")


def _validate_history_peak_hour_columns(df: pd.DataFrame) -> None:
    missing: List[str] = []
    for target_variable in sorted(df[TARGET_VARIABLE_COLUMN].dropna().unique()):
        day_columns = _history_peak_hour_columns(str(target_variable))
        missing.extend(column for column in day_columns if column not in df.columns)
    if missing:
        missing_text = ", ".join(dict.fromkeys(missing))
        raise ValueError(f"输入数据缺少历史峰值小时字段：{missing_text}")


def _validate_hour_values(values: Iterable[object], label: str) -> None:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{label} 存在缺失或非数值小时")
    invalid = numeric.loc[(numeric < 0) | (numeric > 23) | (numeric % 1 != 0)]
    if not invalid.empty:
        invalid_text = ", ".join(str(value) for value in invalid.head(5).tolist())
        raise ValueError(f"{label} 必须为 0 到 23 的整数小时，发现：{invalid_text}")


def _mode_hour(values: Iterable[object]) -> int:
    hours = [int(value) for value in values if not pd.isna(value)]
    if not hours:
        raise ValueError("无法从空小时序列计算众数")
    _validate_hour_values(hours, "峰值小时")
    counts = pd.Series(hours).value_counts()
    max_count = counts.max()
    return int(min(counts.loc[counts == max_count].index))


def _round_half_up(value: float) -> int:
    rounded = int(np.floor(value + 0.5))
    if rounded < 0 or rounded > 23:
        raise ValueError(f"四舍五入后的峰值小时超出 0 到 23：{rounded}")
    return rounded


def _row_history_peak_hours(row: pd.Series) -> List[int]:
    target_variable = str(row[TARGET_VARIABLE_COLUMN])
    columns = _history_peak_hour_columns(target_variable)
    values = [row[column] for column in columns]
    _validate_hour_values(values, f"样本 {row.get(SAMPLE_ID_COLUMN, '<unknown>')} 的历史峰值小时")
    return [int(value) for value in values]


def _mode_lookup(train_df: pd.DataFrame, group_columns: List[str]) -> Dict[Tuple[object, ...], int]:
    modes = train_df.groupby(group_columns, dropna=False)[TARGET_PEAK_HOUR_COLUMN].apply(_mode_hour)
    return {key if isinstance(key, tuple) else (key,): int(value) for key, value in modes.items()}


def _build_mode_lookups(train_df: pd.DataFrame) -> Dict[str, object]:
    return {
        "target_weekday": _mode_lookup(train_df, [TARGET_VARIABLE_COLUMN, WEEKDAY_COLUMN]),
        "target": _mode_lookup(train_df, [TARGET_VARIABLE_COLUMN]),
        "global": _mode_hour(train_df[TARGET_PEAK_HOUR_COLUMN]),
    }


def _global_mode_prediction(row: pd.Series, lookup: Dict[str, object]) -> int:
    target_table = lookup["target"]
    key = (row[TARGET_VARIABLE_COLUMN],)
    if isinstance(target_table, dict) and key in target_table:
        return int(target_table[key])
    return int(lookup["global"])


def _weekday_mode_prediction(row: pd.Series, lookup: Dict[str, object]) -> int:
    weekday_table = lookup["target_weekday"]
    key = (row[TARGET_VARIABLE_COLUMN], row[WEEKDAY_COLUMN])
    if isinstance(weekday_table, dict) and key in weekday_table:
        return int(weekday_table[key])
    return _global_mode_prediction(row, lookup)


def _history_rule_predictions(predict_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for _, row in predict_df.iterrows():
        history_hours = _row_history_peak_hours(row)
        median_hour = _round_half_up(float(np.median(history_hours)))
        values = {
            "mode_last_4": _mode_hour(history_hours),
            "median_last_4": median_hour,
        }

        base_record = {
            SAMPLE_ID_COLUMN: row[SAMPLE_ID_COLUMN],
            START_DATE_COLUMN: row[START_DATE_COLUMN],
            SPLIT_COLUMN: row[SPLIT_COLUMN],
            TARGET_VARIABLE_COLUMN: row[TARGET_VARIABLE_COLUMN],
            HORIZON_COLUMN: int(row[HORIZON_COLUMN]),
            TARGET_PEAK_HOUR_COLUMN: int(row[TARGET_PEAK_HOUR_COLUMN]),
        }
        for baseline_name, baseline_peak_hour in values.items():
            record = dict(base_record)
            record[BASELINE_NAME_COLUMN] = baseline_name
            record[BASELINE_PEAK_HOUR_COLUMN] = int(baseline_peak_hour)
            records.append(record)

    return pd.DataFrame.from_records(records, columns=PREDICTION_OUTPUT_COLUMNS)


def _train_mode_predictions(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> pd.DataFrame:
    lookup = _build_mode_lookups(train_df)
    records: List[Dict[str, object]] = []

    for _, row in predict_df.iterrows():
        base_record = {
            SAMPLE_ID_COLUMN: row[SAMPLE_ID_COLUMN],
            START_DATE_COLUMN: row[START_DATE_COLUMN],
            SPLIT_COLUMN: row[SPLIT_COLUMN],
            TARGET_VARIABLE_COLUMN: row[TARGET_VARIABLE_COLUMN],
            HORIZON_COLUMN: int(row[HORIZON_COLUMN]),
            TARGET_PEAK_HOUR_COLUMN: int(row[TARGET_PEAK_HOUR_COLUMN]),
        }
        values = {
            "global_mode": _global_mode_prediction(row, lookup),
            "weekday_mode": _weekday_mode_prediction(row, lookup),
        }
        for baseline_name, baseline_peak_hour in values.items():
            record = dict(base_record)
            record[BASELINE_NAME_COLUMN] = baseline_name
            record[BASELINE_PEAK_HOUR_COLUMN] = int(baseline_peak_hour)
            records.append(record)

    return pd.DataFrame.from_records(records, columns=PREDICTION_OUTPUT_COLUMNS)


def build_peak_hour_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build peak-hour baseline predictions for validation and test rows."""

    predict_df = pd.concat([val_df, test_df], ignore_index=True)
    if train_df.empty:
        raise ValueError("训练集为空，无法构建峰值小时训练集众数基线")
    if predict_df.empty:
        return pd.DataFrame(columns=PREDICTION_OUTPUT_COLUMNS)

    _validate_required_prediction_columns(train_df)
    _validate_required_prediction_columns(predict_df)
    _validate_history_peak_hour_columns(predict_df)
    _validate_hour_values(train_df[TARGET_PEAK_HOUR_COLUMN], "训练集目标峰值小时")
    _validate_hour_values(predict_df[TARGET_PEAK_HOUR_COLUMN], "预测集目标峰值小时")

    history_predictions = _history_rule_predictions(predict_df)
    train_mode_predictions = _train_mode_predictions(train_df, predict_df)
    predictions = pd.concat([history_predictions, train_mode_predictions], ignore_index=True)
    predictions[HORIZON_COLUMN] = predictions[HORIZON_COLUMN].astype(int)
    predictions[BASELINE_PEAK_HOUR_COLUMN] = pd.to_numeric(
        predictions[BASELINE_PEAK_HOUR_COLUMN], errors="coerce"
    )

    missing_count = int(predictions[BASELINE_PEAK_HOUR_COLUMN].isna().sum())
    if missing_count:
        raise ValueError(f"baseline_peak_hour 存在缺失：{missing_count} 行")
    _validate_hour_values(predictions[BASELINE_PEAK_HOUR_COLUMN], "baseline_peak_hour")
    predictions[BASELINE_PEAK_HOUR_COLUMN] = predictions[BASELINE_PEAK_HOUR_COLUMN].astype(int)

    return predictions.loc[:, PREDICTION_OUTPUT_COLUMNS]


def _circular_hour_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    ordinary = np.abs(pred - true)
    return np.minimum(ordinary, 24 - ordinary)


def _metric_summary(group: pd.DataFrame) -> pd.Series:
    true = group[TARGET_PEAK_HOUR_COLUMN].astype(int).to_numpy()
    pred = group[BASELINE_PEAK_HOUR_COLUMN].astype(int).to_numpy()
    ordinary_error = np.abs(pred - true)
    circular_error = _circular_hour_error(pred, true)

    return pd.Series(
        {
            "row_count": int(len(group)),
            "普通小时误差": float(np.mean(ordinary_error)),
            "环形小时误差": float(np.mean(circular_error)),
            "Top-1 accuracy": float(np.mean(pred == true)),
            "±1h 命中率": float(np.mean(circular_error <= 1)),
            "±2h 命中率": float(np.mean(circular_error <= 2)),
        }
    )


def _evaluate_level(
    predictions: pd.DataFrame,
    eval_level: str,
    group_columns: List[str],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, key_tuple))
        record.update(_metric_summary(group).to_dict())
        records.append(record)
    grouped = pd.DataFrame.from_records(records)
    grouped.insert(0, "eval_level", eval_level)

    if TARGET_VARIABLE_COLUMN not in grouped.columns:
        grouped[TARGET_VARIABLE_COLUMN] = pd.NA
    if HORIZON_COLUMN not in grouped.columns:
        grouped[HORIZON_COLUMN] = pd.NA

    return grouped[
        [
            "eval_level",
            SPLIT_COLUMN,
            BASELINE_NAME_COLUMN,
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            "row_count",
            *METRIC_COLUMNS,
        ]
    ]


def evaluate_peak_hour_baselines(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate baseline predictions at global, variable, horizon, and combined levels."""

    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "eval_level",
                SPLIT_COLUMN,
                BASELINE_NAME_COLUMN,
                TARGET_VARIABLE_COLUMN,
                HORIZON_COLUMN,
                "row_count",
                *METRIC_COLUMNS,
            ]
        )

    _validate_hour_values(predictions[TARGET_PEAK_HOUR_COLUMN], TARGET_PEAK_HOUR_COLUMN)
    _validate_hour_values(predictions[BASELINE_PEAK_HOUR_COLUMN], BASELINE_PEAK_HOUR_COLUMN)

    levels = (
        ("global", [SPLIT_COLUMN, BASELINE_NAME_COLUMN]),
        ("target_variable", [SPLIT_COLUMN, BASELINE_NAME_COLUMN, TARGET_VARIABLE_COLUMN]),
        ("horizon", [SPLIT_COLUMN, BASELINE_NAME_COLUMN, HORIZON_COLUMN]),
        (
            "target_variable_horizon",
            [SPLIT_COLUMN, BASELINE_NAME_COLUMN, TARGET_VARIABLE_COLUMN, HORIZON_COLUMN],
        ),
    )
    metrics = [_evaluate_level(predictions, eval_level, columns) for eval_level, columns in levels]
    return pd.concat(metrics, ignore_index=True)


def run_peak_hour_baselines(
    input_path: Optional[Union[Path, str]] = None,
    prediction_path: Optional[Union[Path, str]] = None,
    metrics_path: Optional[Union[Path, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data, generate peak-hour baselines, write predictions and metrics."""

    resolved_input_path = Path(input_path) if input_path is not None else DEFAULT_DATA_PATH
    resolved_prediction_path = (
        Path(prediction_path) if prediction_path is not None else DEFAULT_PREDICTION_PATH
    )
    resolved_metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH

    train_df, val_df, test_df = load_peak_dataset(resolved_input_path)
    predictions = build_peak_hour_predictions(train_df, val_df, test_df)
    metrics = evaluate_peak_hour_baselines(predictions)

    resolved_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(resolved_prediction_path, index=False)
    metrics.to_csv(resolved_metrics_path, index=False)

    return predictions, metrics


def main() -> None:
    predictions, metrics = run_peak_hour_baselines()
    print(f"input_path: {DEFAULT_DATA_PATH}")
    print(f"prediction_path: {DEFAULT_PREDICTION_PATH}")
    print(f"metrics_path: {DEFAULT_METRICS_PATH}")
    print(f"prediction_rows: {len(predictions)}")
    print(f"metric_rows: {len(metrics)}")
    print(f"baselines: {', '.join(BASELINE_NAMES)}")
    print(
        "prediction_splits: "
        + ", ".join(f"{split}={count}" for split, count in predictions[SPLIT_COLUMN].value_counts().items())
    )
    plot_paths = maybe_plot_peak_baseline_predictions()
    if plot_paths:
        print("波峰预测可视化: " + ", ".join(f"{col}={len(paths)}" for col, paths in plot_paths.items()))


if __name__ == "__main__":
    main()
