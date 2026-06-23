"""Generate rule-based peak-value baseline predictions and metrics."""

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
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    load_peak_dataset,
)
from wave_experiments.baselines.visualize_baselines import maybe_plot_peak_baseline_predictions


SAMPLE_ID_COLUMN = "样本ID"
START_DATE_COLUMN = "预测起点日期"
TARGET_PEAK_COLUMN = "目标峰值"
WEEKDAY_COLUMN = "日历_星期"
BASELINE_PEAK_VALUE_COLUMN = "baseline_peak_value"
BASELINE_NAME_COLUMN = "baseline_name"

DEFAULT_PREDICTION_PATH = Path("实验输出/results/baselines/peak_value_baseline_predictions.csv")
DEFAULT_METRICS_PATH = Path("实验输出/results/baselines/peak_value_baseline_metrics.csv")

BASELINE_NAMES = (
    "mean_last_4",
    "weighted_mean_last_4",
    "cycle_mod_4",
    "weekday_horizon_mean",
)

PREDICTION_OUTPUT_COLUMNS = (
    SAMPLE_ID_COLUMN,
    START_DATE_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
    TARGET_PEAK_COLUMN,
    BASELINE_NAME_COLUMN,
    BASELINE_PEAK_VALUE_COLUMN,
)

METRIC_COLUMNS = ("MAE", "RMSE", "sMAPE")


def _history_peak_columns(target_variable: str) -> List[str]:
    return [f"{target_variable}_过去第{day}天_最大值" for day in range(1, 5)]


def _history_mean_column(target_variable: str) -> str:
    return f"{target_variable}_历史峰值_均值4天"


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


def _validate_required_prediction_columns(df: pd.DataFrame) -> None:
    required_columns = (
        SAMPLE_ID_COLUMN,
        START_DATE_COLUMN,
        SPLIT_COLUMN,
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        TARGET_PEAK_COLUMN,
        WEEKDAY_COLUMN,
    )
    missing = _missing_columns(df, required_columns)
    if missing:
        raise ValueError(f"输入数据缺少生成峰值基线所需字段：{', '.join(missing)}")


def _validate_history_peak_columns(df: pd.DataFrame) -> None:
    missing: List[str] = []
    for target_variable in sorted(df[TARGET_VARIABLE_COLUMN].dropna().unique()):
        day_columns = _history_peak_columns(str(target_variable))
        mean_column = _history_mean_column(str(target_variable))
        if any(column not in df.columns for column in day_columns) and mean_column not in df.columns:
            missing.extend(day_columns)
            missing.append(mean_column)
    if missing:
        missing_text = ", ".join(dict.fromkeys(missing))
        raise ValueError(f"输入数据缺少历史峰值字段：{missing_text}")


def _row_history_peaks(row: pd.Series) -> List[float]:
    target_variable = str(row[TARGET_VARIABLE_COLUMN])
    day_columns = _history_peak_columns(target_variable)
    values = [row[column] for column in day_columns if column in row.index]

    if len(values) == 4 and not pd.isna(values).any():
        return [float(value) for value in values]

    mean_column = _history_mean_column(target_variable)
    if mean_column in row.index and not pd.isna(row[mean_column]):
        mean_value = float(row[mean_column])
        return [mean_value, mean_value, mean_value, mean_value]

    raise ValueError(f"样本 {row.get(SAMPLE_ID_COLUMN, '<unknown>')} 缺少目标变量 {target_variable} 的历史峰值")


def _history_rule_predictions(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        peaks_new_to_old = _row_history_peaks(row)
        horizon = int(row[HORIZON_COLUMN])
        cycle_index = (horizon - 1) % 4
        weighted_peak = (
            0.4 * peaks_new_to_old[0]
            + 0.3 * peaks_new_to_old[1]
            + 0.2 * peaks_new_to_old[2]
            + 0.1 * peaks_new_to_old[3]
        )
        values = {
            "mean_last_4": float(np.mean(peaks_new_to_old)),
            "weighted_mean_last_4": float(weighted_peak),
            "cycle_mod_4": float(peaks_new_to_old[cycle_index]),
        }

        base_record = {
            SAMPLE_ID_COLUMN: row[SAMPLE_ID_COLUMN],
            START_DATE_COLUMN: row[START_DATE_COLUMN],
            SPLIT_COLUMN: row[SPLIT_COLUMN],
            TARGET_VARIABLE_COLUMN: row[TARGET_VARIABLE_COLUMN],
            HORIZON_COLUMN: horizon,
            TARGET_PEAK_COLUMN: row[TARGET_PEAK_COLUMN],
        }
        for baseline_name, baseline_peak_value in values.items():
            record = dict(base_record)
            record[BASELINE_NAME_COLUMN] = baseline_name
            record[BASELINE_PEAK_VALUE_COLUMN] = baseline_peak_value
            records.append(record)

    return pd.DataFrame.from_records(records, columns=PREDICTION_OUTPUT_COLUMNS)


def _mean_lookup(
    train_df: pd.DataFrame,
    group_columns: List[str],
) -> Dict[Tuple[object, ...], float]:
    means = train_df.groupby(group_columns, dropna=False)[TARGET_PEAK_COLUMN].mean()
    return {key if isinstance(key, tuple) else (key,): float(value) for key, value in means.items()}


def _build_weekday_horizon_lookup(train_df: pd.DataFrame) -> Dict[str, object]:
    global_mean = float(train_df[TARGET_PEAK_COLUMN].mean())
    return {
        "target_weekday_horizon": _mean_lookup(
            train_df, [TARGET_VARIABLE_COLUMN, WEEKDAY_COLUMN, HORIZON_COLUMN]
        ),
        "target_horizon": _mean_lookup(train_df, [TARGET_VARIABLE_COLUMN, HORIZON_COLUMN]),
        "target_weekday": _mean_lookup(train_df, [TARGET_VARIABLE_COLUMN, WEEKDAY_COLUMN]),
        "target": _mean_lookup(train_df, [TARGET_VARIABLE_COLUMN]),
        "global": global_mean,
    }


def _weekday_horizon_prediction(row: pd.Series, lookup: Dict[str, object]) -> float:
    target_variable = row[TARGET_VARIABLE_COLUMN]
    weekday = row[WEEKDAY_COLUMN]
    horizon = row[HORIZON_COLUMN]

    lookup_steps = (
        ("target_weekday_horizon", (target_variable, weekday, horizon)),
        ("target_horizon", (target_variable, horizon)),
        ("target_weekday", (target_variable, weekday)),
        ("target", (target_variable,)),
    )
    for lookup_name, key in lookup_steps:
        table = lookup[lookup_name]
        if isinstance(table, dict) and key in table:
            return float(table[key])

    return float(lookup["global"])


def _weekday_horizon_rule_predictions(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
) -> pd.DataFrame:
    lookup = _build_weekday_horizon_lookup(train_df)
    records: List[Dict[str, object]] = []

    for _, row in predict_df.iterrows():
        records.append(
            {
                SAMPLE_ID_COLUMN: row[SAMPLE_ID_COLUMN],
                START_DATE_COLUMN: row[START_DATE_COLUMN],
                SPLIT_COLUMN: row[SPLIT_COLUMN],
                TARGET_VARIABLE_COLUMN: row[TARGET_VARIABLE_COLUMN],
                HORIZON_COLUMN: int(row[HORIZON_COLUMN]),
                TARGET_PEAK_COLUMN: row[TARGET_PEAK_COLUMN],
                BASELINE_NAME_COLUMN: "weekday_horizon_mean",
                BASELINE_PEAK_VALUE_COLUMN: _weekday_horizon_prediction(row, lookup),
            }
        )

    return pd.DataFrame.from_records(records, columns=PREDICTION_OUTPUT_COLUMNS)


def build_peak_value_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build peak-value baseline predictions for validation and test rows."""

    predict_df = pd.concat([val_df, test_df], ignore_index=True)
    if train_df.empty:
        raise ValueError("训练集为空，无法构建 weekday_horizon_mean 基线")
    if predict_df.empty:
        return pd.DataFrame(columns=PREDICTION_OUTPUT_COLUMNS)

    _validate_required_prediction_columns(train_df)
    _validate_required_prediction_columns(predict_df)
    _validate_history_peak_columns(predict_df)

    history_predictions = _history_rule_predictions(predict_df)
    weekday_predictions = _weekday_horizon_rule_predictions(train_df, predict_df)
    predictions = pd.concat([history_predictions, weekday_predictions], ignore_index=True)
    predictions[HORIZON_COLUMN] = predictions[HORIZON_COLUMN].astype(int)
    predictions[BASELINE_PEAK_VALUE_COLUMN] = pd.to_numeric(
        predictions[BASELINE_PEAK_VALUE_COLUMN], errors="coerce"
    )

    missing_count = int(predictions[BASELINE_PEAK_VALUE_COLUMN].isna().sum())
    if missing_count:
        raise ValueError(f"baseline_peak_value 存在缺失：{missing_count} 行")

    return predictions.loc[:, PREDICTION_OUTPUT_COLUMNS]


def _metric_summary(group: pd.DataFrame) -> pd.Series:
    true = group[TARGET_PEAK_COLUMN].astype(float).to_numpy()
    pred = group[BASELINE_PEAK_VALUE_COLUMN].astype(float).to_numpy()
    abs_error = np.abs(pred - true)
    squared_error = np.square(pred - true)
    denominator = np.abs(pred) + np.abs(true)
    smape_values = np.zeros_like(abs_error, dtype=float)
    np.divide(2.0 * abs_error, denominator, out=smape_values, where=denominator != 0.0)

    return pd.Series(
        {
            "row_count": int(len(group)),
            "MAE": float(np.mean(abs_error)),
            "RMSE": float(np.sqrt(np.mean(squared_error))),
            "sMAPE": float(np.mean(smape_values)),
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


def evaluate_peak_value_baselines(predictions: pd.DataFrame) -> pd.DataFrame:
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


def run_peak_value_baselines(
    input_path: Optional[Union[Path, str]] = None,
    prediction_path: Optional[Union[Path, str]] = None,
    metrics_path: Optional[Union[Path, str]] = None,
    plot_predictions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data, generate peak-value baselines, write predictions and metrics."""

    resolved_input_path = Path(input_path) if input_path is not None else DEFAULT_DATA_PATH
    resolved_prediction_path = (
        Path(prediction_path) if prediction_path is not None else DEFAULT_PREDICTION_PATH
    )
    resolved_metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH

    train_df, val_df, test_df = load_peak_dataset(resolved_input_path)
    predictions = build_peak_value_predictions(train_df, val_df, test_df)
    metrics = evaluate_peak_value_baselines(predictions)

    resolved_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(resolved_prediction_path, index=False)
    metrics.to_csv(resolved_metrics_path, index=False)

    if plot_predictions:
        maybe_plot_peak_baseline_predictions(
            value_prediction_csv=resolved_prediction_path,
            plot_hour=False,
        )

    return predictions, metrics


def main() -> None:
    predictions, metrics = run_peak_value_baselines()
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
    print("可视化任务: 波峰值")


if __name__ == "__main__":
    main()
