"""Select validation-best rule baselines for residual learning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from wave_dataset.visualization import plot_selected_best_baseline_prediction_batch
from wave_experiments.baselines.load_baseline_data import (
    BASELINE_COLUMNS as SOURCE_BASELINE_COLUMNS,
    DEFAULT_DATA_PATH,
    HORIZON_COLUMN,
    ID_COLUMNS as SOURCE_ID_COLUMNS,
    LABEL_COLUMNS as SOURCE_LABEL_COLUMNS,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    VAL_SPLIT,
    get_calendar_columns,
    get_history_feature_columns,
)


SAMPLE_ID_COLUMN = "样本ID"
START_DATE_COLUMN = "预测起点日期"
BASELINE_NAME_COLUMN = "baseline_name"
TARGET_PEAK_COLUMN = "目标峰值"
TARGET_PEAK_HOUR_COLUMN = "目标峰值小时"
BASELINE_PEAK_VALUE_COLUMN = "baseline_peak_value"
BASELINE_PEAK_HOUR_COLUMN = "baseline_peak_hour"
BASELINE_PEAK_COLUMN = "baseline_peak"
TARGET_PEAK_RESIDUAL_COLUMN = "target_peak_residual"

TARGET_HORIZON_LEVEL = "target_variable_horizon"

RESULTS_DIR = Path("实验输出/results/baselines")
REPORTS_DIR = Path("实验输出/reports")

DEFAULT_PEAK_VALUE_PREDICTION_PATH = RESULTS_DIR / "peak_value_baseline_predictions.csv"
DEFAULT_PEAK_VALUE_METRICS_PATH = RESULTS_DIR / "peak_value_baseline_metrics.csv"
DEFAULT_PEAK_HOUR_PREDICTION_PATH = RESULTS_DIR / "peak_hour_baseline_predictions.csv"
DEFAULT_PEAK_HOUR_METRICS_PATH = RESULTS_DIR / "peak_hour_baseline_metrics.csv"
DEFAULT_SOURCE_DATA_PATH = DEFAULT_DATA_PATH

DEFAULT_BEST_PEAK_VALUE_PATH = RESULTS_DIR / "best_peak_value_baseline_by_group.csv"
DEFAULT_BEST_PEAK_HOUR_PATH = RESULTS_DIR / "best_peak_hour_baseline_by_group.csv"
DEFAULT_SELECTED_DATASET_PATH = RESULTS_DIR / "dataset_with_selected_baselines.csv"
DEFAULT_REPORT_PATH = REPORTS_DIR / "规则基线总结.md"

GROUP_COLUMNS = (TARGET_VARIABLE_COLUMN, HORIZON_COLUMN)
SAMPLE_KEY_COLUMNS = (
    SAMPLE_ID_COLUMN,
    START_DATE_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
)

PEAK_VALUE_SELECTION_RULE = "验证集 target_variable_horizon 层级内 MAE 最低"
PEAK_HOUR_SELECTION_RULE = (
    "验证集 target_variable_horizon 层级内 ±2h 命中率最高；并列时普通小时误差最低"
)


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = _missing_columns(df, columns)
    if missing:
        raise ValueError(f"{label} 缺少必要字段：{', '.join(missing)}")


def _validation_group_metrics(
    metrics: pd.DataFrame,
    metric_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    required_columns = (
        "eval_level",
        SPLIT_COLUMN,
        BASELINE_NAME_COLUMN,
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        "row_count",
        *metric_columns,
    )
    _require_columns(metrics, required_columns, label)

    selected = metrics.loc[
        (metrics["eval_level"] == TARGET_HORIZON_LEVEL) & (metrics[SPLIT_COLUMN] == VAL_SPLIT)
    ].copy()
    selected = selected.dropna(subset=[TARGET_VARIABLE_COLUMN, HORIZON_COLUMN])
    if selected.empty:
        raise ValueError(f"{label} 未找到验证集 target_variable_horizon 指标")

    selected[HORIZON_COLUMN] = pd.to_numeric(selected[HORIZON_COLUMN], errors="coerce")
    if selected[HORIZON_COLUMN].isna().any():
        raise ValueError(f"{label} 的预测天数字段存在非数值")
    selected[HORIZON_COLUMN] = selected[HORIZON_COLUMN].astype(int)
    selected["row_count"] = pd.to_numeric(selected["row_count"], errors="coerce")

    for column in metric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        if selected[column].isna().any():
            raise ValueError(f"{label} 的 {column} 存在缺失或非数值")

    return selected


def _validate_one_best_per_group(best: pd.DataFrame, label: str) -> None:
    duplicated = best.duplicated(list(GROUP_COLUMNS), keep=False)
    if duplicated.any():
        examples = best.loc[duplicated, list(GROUP_COLUMNS)].drop_duplicates().head(5)
        raise ValueError(f"{label} 每个目标变量+预测天数必须只有一个最佳基线：\n{examples}")


def _sort_best(best: pd.DataFrame) -> pd.DataFrame:
    sorted_best = best.sort_values(list(GROUP_COLUMNS), kind="mergesort").reset_index(drop=True)
    sorted_best[HORIZON_COLUMN] = sorted_best[HORIZON_COLUMN].astype(int)
    return sorted_best


def select_best_peak_value_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-validation-MAE peak-value baseline by variable and horizon."""

    selected = _validation_group_metrics(metrics, ("MAE", "RMSE", "sMAPE"), "peak_value 指标表")
    selected = selected.sort_values(
        [TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, "MAE", BASELINE_NAME_COLUMN],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    best = selected.groupby(list(GROUP_COLUMNS), as_index=False, sort=False).first()
    best = best.rename(
        columns={
            BASELINE_NAME_COLUMN: "best_baseline_name",
            "row_count": "validation_row_count",
            "MAE": "validation_MAE",
            "RMSE": "validation_RMSE",
            "sMAPE": "validation_sMAPE",
        }
    )
    best["selection_metric"] = "MAE"
    best["selection_rule"] = PEAK_VALUE_SELECTION_RULE

    output_columns = [
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        "best_baseline_name",
        "selection_metric",
        "selection_rule",
        "validation_row_count",
        "validation_MAE",
        "validation_RMSE",
        "validation_sMAPE",
    ]
    best = _sort_best(best.loc[:, output_columns])
    _validate_one_best_per_group(best, "peak_value 最佳基线表")
    return best


def select_best_peak_hour_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select peak-hour baselines by validation ±2h hit rate, then mean hour error."""

    metric_columns = (
        "普通小时误差",
        "环形小时误差",
        "Top-1 accuracy",
        "±1h 命中率",
        "±2h 命中率",
    )
    selected = _validation_group_metrics(metrics, metric_columns, "peak_hour 指标表")
    selected = selected.sort_values(
        [
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            "±2h 命中率",
            "普通小时误差",
            "环形小时误差",
            BASELINE_NAME_COLUMN,
        ],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    )
    best = selected.groupby(list(GROUP_COLUMNS), as_index=False, sort=False).first()
    best = best.rename(
        columns={
            BASELINE_NAME_COLUMN: "best_baseline_name",
            "row_count": "validation_row_count",
            "普通小时误差": "validation_普通小时误差",
            "环形小时误差": "validation_环形小时误差",
            "Top-1 accuracy": "validation_Top-1 accuracy",
            "±1h 命中率": "validation_±1h 命中率",
            "±2h 命中率": "validation_±2h 命中率",
        }
    )
    best["selection_metric"] = "±2h 命中率, then 普通小时误差"
    best["selection_rule"] = PEAK_HOUR_SELECTION_RULE

    output_columns = [
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        "best_baseline_name",
        "selection_metric",
        "selection_rule",
        "validation_row_count",
        "validation_±2h 命中率",
        "validation_普通小时误差",
        "validation_环形小时误差",
        "validation_Top-1 accuracy",
        "validation_±1h 命中率",
    ]
    best = _sort_best(best.loc[:, output_columns])
    _validate_one_best_per_group(best, "peak_hour 最佳基线表")
    return best


def _selected_prediction_rows(
    predictions: pd.DataFrame,
    best: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    label: str,
) -> pd.DataFrame:
    required_prediction_columns = (
        *SAMPLE_KEY_COLUMNS,
        target_column,
        BASELINE_NAME_COLUMN,
        prediction_column,
    )
    _require_columns(predictions, required_prediction_columns, f"{label} 预测表")
    _require_columns(best, (*GROUP_COLUMNS, "best_baseline_name"), f"{label} 最佳基线表")
    _validate_one_best_per_group(best, f"{label} 最佳基线表")

    prediction_rows = predictions.copy()
    prediction_rows[HORIZON_COLUMN] = pd.to_numeric(
        prediction_rows[HORIZON_COLUMN], errors="coerce"
    )
    if prediction_rows[HORIZON_COLUMN].isna().any():
        raise ValueError(f"{label} 预测表的预测天数字段存在非数值")
    prediction_rows[HORIZON_COLUMN] = prediction_rows[HORIZON_COLUMN].astype(int)

    best_lookup = best.loc[:, [*GROUP_COLUMNS, "best_baseline_name"]].rename(
        columns={"best_baseline_name": BASELINE_NAME_COLUMN}
    )
    best_lookup[HORIZON_COLUMN] = best_lookup[HORIZON_COLUMN].astype(int)

    selected = prediction_rows.merge(
        best_lookup,
        on=[TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, BASELINE_NAME_COLUMN],
        how="inner",
    )
    expected_rows = predictions.loc[:, list(SAMPLE_KEY_COLUMNS)].drop_duplicates()
    actual_rows = selected.loc[:, list(SAMPLE_KEY_COLUMNS)].drop_duplicates()
    if len(actual_rows) != len(expected_rows):
        raise ValueError(
            f"{label} 选中预测行数不完整：期望 {len(expected_rows)} 行，实际 {len(actual_rows)} 行"
        )
    if selected.duplicated(list(SAMPLE_KEY_COLUMNS)).any():
        raise ValueError(f"{label} 选中预测存在重复样本键")

    selected[prediction_column] = pd.to_numeric(selected[prediction_column], errors="coerce")
    if selected[prediction_column].isna().any():
        raise ValueError(f"{label} 选中预测的 {prediction_column} 存在缺失")
    return selected


def _circular_hour_error(pred: pd.Series, true: pd.Series) -> pd.Series:
    ordinary = (pred.astype(int) - true.astype(int)).abs()
    return pd.Series(np.minimum(ordinary.to_numpy(), 24 - ordinary.to_numpy()), index=pred.index)


def _validate_unique_sample_keys(df: pd.DataFrame, label: str) -> None:
    duplicated = df.duplicated(list(SAMPLE_KEY_COLUMNS), keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, list(SAMPLE_KEY_COLUMNS)].head(5)
        raise ValueError(f"{label} 存在重复样本键，无法一对一合并：\n{examples}")


def _coerce_horizon(df: pd.DataFrame, label: str) -> pd.DataFrame:
    coerced = df.copy()
    coerced[HORIZON_COLUMN] = pd.to_numeric(coerced[HORIZON_COLUMN], errors="coerce")
    if coerced[HORIZON_COLUMN].isna().any():
        raise ValueError(f"{label} 的预测天数字段存在非数值")
    coerced[HORIZON_COLUMN] = coerced[HORIZON_COLUMN].astype(int)
    return coerced


def build_dataset_with_selected_baselines(
    source_df: pd.DataFrame,
    value_predictions: pd.DataFrame,
    hour_predictions: pd.DataFrame,
    best_value: pd.DataFrame,
    best_hour: pd.DataFrame,
) -> pd.DataFrame:
    """Append selected value/hour baselines to the complete long-table rows."""

    _require_columns(source_df, SAMPLE_KEY_COLUMNS, "原始完整长表")
    source = _coerce_horizon(source_df, "原始完整长表")
    _validate_unique_sample_keys(source, "原始完整长表")

    selected_value = _selected_prediction_rows(
        value_predictions,
        best_value,
        BASELINE_PEAK_VALUE_COLUMN,
        TARGET_PEAK_COLUMN,
        "peak_value",
    )
    selected_hour = _selected_prediction_rows(
        hour_predictions,
        best_hour,
        BASELINE_PEAK_HOUR_COLUMN,
        TARGET_PEAK_HOUR_COLUMN,
        "peak_hour",
    )

    value_output = selected_value.loc[
        :,
        [
            *SAMPLE_KEY_COLUMNS,
            TARGET_PEAK_COLUMN,
            BASELINE_NAME_COLUMN,
            BASELINE_PEAK_VALUE_COLUMN,
        ],
    ].rename(
        columns={
            BASELINE_NAME_COLUMN: "peak_value_baseline_name",
            BASELINE_PEAK_VALUE_COLUMN: BASELINE_PEAK_COLUMN,
        }
    )
    value_output[TARGET_PEAK_COLUMN] = pd.to_numeric(value_output[TARGET_PEAK_COLUMN], errors="coerce")
    value_output[BASELINE_PEAK_COLUMN] = pd.to_numeric(
        value_output[BASELINE_PEAK_COLUMN], errors="coerce"
    )
    value_output[TARGET_PEAK_RESIDUAL_COLUMN] = (
        value_output[TARGET_PEAK_COLUMN] - value_output[BASELINE_PEAK_COLUMN]
    )

    hour_output = selected_hour.loc[
        :,
        [
            *SAMPLE_KEY_COLUMNS,
            TARGET_PEAK_HOUR_COLUMN,
            BASELINE_NAME_COLUMN,
            BASELINE_PEAK_HOUR_COLUMN,
        ],
    ].rename(columns={BASELINE_NAME_COLUMN: "peak_hour_baseline_name"})
    hour_output[TARGET_PEAK_HOUR_COLUMN] = pd.to_numeric(
        hour_output[TARGET_PEAK_HOUR_COLUMN], errors="coerce"
    ).astype(int)
    hour_output[BASELINE_PEAK_HOUR_COLUMN] = pd.to_numeric(
        hour_output[BASELINE_PEAK_HOUR_COLUMN], errors="coerce"
    ).astype(int)
    hour_output["peak_hour_ordinary_error"] = (
        hour_output[BASELINE_PEAK_HOUR_COLUMN] - hour_output[TARGET_PEAK_HOUR_COLUMN]
    ).abs()
    hour_output["peak_hour_circular_error"] = _circular_hour_error(
        hour_output[BASELINE_PEAK_HOUR_COLUMN],
        hour_output[TARGET_PEAK_HOUR_COLUMN],
    )
    hour_output["peak_hour_within_2h"] = hour_output["peak_hour_circular_error"] <= 2

    selected_baselines = value_output.merge(
        hour_output,
        on=list(SAMPLE_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("_value", "_hour"),
    )
    if len(selected_baselines) != len(value_output):
        raise ValueError(
            f"合并后的最佳基线数据行数不完整：期望 {len(value_output)} 行，实际 {len(selected_baselines)} 行"
        )

    for column in (TARGET_PEAK_COLUMN, TARGET_PEAK_HOUR_COLUMN):
        suffixed = f"{column}_hour"
        if suffixed in selected_baselines.columns:
            selected_baselines = selected_baselines.drop(columns=[suffixed])
        suffixed = f"{column}_value"
        if suffixed in selected_baselines.columns:
            selected_baselines = selected_baselines.rename(columns={suffixed: column})

    appended_columns = [
        BASELINE_PEAK_COLUMN,
        TARGET_PEAK_RESIDUAL_COLUMN,
        "peak_value_baseline_name",
        BASELINE_PEAK_HOUR_COLUMN,
        "peak_hour_baseline_name",
        "peak_hour_ordinary_error",
        "peak_hour_circular_error",
        "peak_hour_within_2h",
    ]
    source_without_appended = source.drop(
        columns=[column for column in appended_columns if column in source.columns]
    )
    dataset = source_without_appended.merge(
        selected_baselines.loc[:, [*SAMPLE_KEY_COLUMNS, *appended_columns]],
        on=list(SAMPLE_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if len(dataset) != len(source):
        raise ValueError(f"合并后的残差学习数据行数不完整：期望 {len(source)} 行，实际 {len(dataset)} 行")

    dataset = dataset.sort_values(list(SAMPLE_KEY_COLUMNS)).reset_index(drop=True)

    missing_baseline_peak = int(dataset[BASELINE_PEAK_COLUMN].isna().sum())
    if missing_baseline_peak:
        raise ValueError(f"{BASELINE_PEAK_COLUMN} 存在缺失：{missing_baseline_peak} 行")
    missing_baseline_peak_hour = int(dataset[BASELINE_PEAK_HOUR_COLUMN].isna().sum())
    if missing_baseline_peak_hour:
        raise ValueError(f"{BASELINE_PEAK_HOUR_COLUMN} 存在缺失：{missing_baseline_peak_hour} 行")
    if dataset[TARGET_PEAK_RESIDUAL_COLUMN].isna().any():
        raise ValueError(f"{TARGET_PEAK_RESIDUAL_COLUMN} 存在缺失")

    return dataset


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.loc[:, columns].iterrows():
        rows.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _list_text(values: Sequence[str], limit: int = 12) -> str:
    if not values:
        return "无"
    shown = [f"`{value}`" for value in values[:limit]]
    if len(values) > limit:
        shown.append(f"...（共 {len(values)} 个）")
    return "、".join(shown)


def build_report(
    best_value: pd.DataFrame,
    best_hour: pd.DataFrame,
    selected_dataset: pd.DataFrame,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
) -> str:
    """Create a markdown report describing the selected rule baselines."""

    value_report = best_value.rename(
        columns={
            "best_baseline_name": "最佳基线",
            "validation_MAE": "验证MAE",
            "validation_RMSE": "验证RMSE",
            "validation_sMAPE": "验证sMAPE",
        }
    )
    hour_report = best_hour.rename(
        columns={
            "best_baseline_name": "最佳基线",
            "validation_±2h 命中率": "验证±2h命中率",
            "validation_普通小时误差": "验证普通小时误差",
            "validation_环形小时误差": "验证环形小时误差",
        }
    )

    baseline_peak_missing = int(selected_dataset[BASELINE_PEAK_COLUMN].isna().sum())
    baseline_peak_hour_missing = int(selected_dataset[BASELINE_PEAK_HOUR_COLUMN].isna().sum())
    calendar_columns = get_calendar_columns(selected_dataset.columns)
    history_columns = get_history_feature_columns(selected_dataset.columns)
    source_id_columns = [column for column in SOURCE_ID_COLUMNS if column in selected_dataset.columns]
    source_label_columns = [column for column in SOURCE_LABEL_COLUMNS if column in selected_dataset.columns]
    source_baseline_columns = [
        column for column in SOURCE_BASELINE_COLUMNS if column in selected_dataset.columns
    ]
    lines = [
        "# 规则基线总结",
        "",
        "## 输入文件",
        "",
        *[f"- `{path}`" for path in input_paths],
        "",
        "## 选择依据",
        "",
        f"- peak_value：按 `{TARGET_VARIABLE_COLUMN}` + `{HORIZON_COLUMN}` 分组，只使用验证集 `{TARGET_HORIZON_LEVEL}` 指标，选择 MAE 最低的基线。",
        "- peak_hour：按 `目标变量` + `预测天数` 分组，只使用验证集 "
        f"`{TARGET_HORIZON_LEVEL}` 指标，选择 `±2h 命中率` 最高的基线；并列时选择 `普通小时误差`（平均小时误差）更低者。",
        "",
        "## 输出文件与检查",
        "",
        *[f"- `{path}`" for path in output_paths],
        "",
        f"- peak_value 最佳基线组数：{len(best_value)}",
        f"- peak_hour 最佳基线组数：{len(best_hour)}",
        f"- 残差学习数据行数：{len(selected_dataset)}",
        f"- 输出表列数：{len(selected_dataset.columns)}",
        f"- `baseline_peak` 缺失数：{baseline_peak_missing}",
        f"- `baseline_peak_hour` 缺失数：{baseline_peak_hour_missing}",
        "",
        "## 输出表字段说明",
        "",
        "- `dataset_with_selected_baselines.csv` 以原始完整长表为主表，保留原始 ID、标签、日历、历史统计和跨变量特征，再追加验证集选出的最佳规则基线预测结果。",
        f"- ID 字段：{_list_text(source_id_columns)}",
        f"- 标签字段：{_list_text(source_label_columns)}",
        f"- 原始基线字段：{_list_text(source_baseline_columns)}",
        f"- 日历字段：{_list_text(calendar_columns)}",
        f"- 历史统计特征字段：{_list_text(history_columns)}",
        "- 追加字段：`baseline_peak`、`peak_value_baseline_name`、`baseline_peak_hour`、`peak_hour_baseline_name`、`peak_hour_ordinary_error`、`peak_hour_circular_error`、`peak_hour_within_2h`。",
        "- 原始长表中的 `目标峰值残差` 或 `基线峰值` 如存在会被保留；它们不是第 7 步残差学习的最终标签。第 7 步应使用本步骤追加的 `baseline_peak` 重新生成最终残差标签。",
        "- 未来真实标签字段可保留在数据表中用于监督目标或评估，但不得作为模型输入；后续第 8 步必须排除未来标签及由标签计算的误差/命中字段。",
        "",
        "## peak_value 最佳基线",
        "",
        _markdown_table(
            value_report,
            [TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, "最佳基线", "验证MAE", "验证RMSE", "验证sMAPE"],
        ),
        "",
        "## peak_hour 最佳基线",
        "",
        _markdown_table(
            hour_report,
            [
                TARGET_VARIABLE_COLUMN,
                HORIZON_COLUMN,
                "最佳基线",
                "验证±2h命中率",
                "验证普通小时误差",
                "验证环形小时误差",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def run_select_best_baselines(
    source_data_path: Optional[Union[Path, str]] = None,
    peak_value_prediction_path: Optional[Union[Path, str]] = None,
    peak_value_metrics_path: Optional[Union[Path, str]] = None,
    peak_hour_prediction_path: Optional[Union[Path, str]] = None,
    peak_hour_metrics_path: Optional[Union[Path, str]] = None,
    best_peak_value_path: Optional[Union[Path, str]] = None,
    best_peak_hour_path: Optional[Union[Path, str]] = None,
    selected_dataset_path: Optional[Union[Path, str]] = None,
    report_path: Optional[Union[Path, str]] = None,
    plot_selected_best_baselines: bool = True,
    hourly_csv: Union[Path, str] = "ETTh1.csv",
    plot_output_root: Union[Path, str] = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    plot_split: str = VAL_SPLIT,
    plot_sample_count: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Read baseline outputs, select best baselines, and write all artifacts."""

    resolved_source_data_path = (
        Path(source_data_path) if source_data_path is not None else DEFAULT_SOURCE_DATA_PATH
    )
    resolved_peak_value_prediction_path = (
        Path(peak_value_prediction_path)
        if peak_value_prediction_path is not None
        else DEFAULT_PEAK_VALUE_PREDICTION_PATH
    )
    resolved_peak_value_metrics_path = (
        Path(peak_value_metrics_path)
        if peak_value_metrics_path is not None
        else DEFAULT_PEAK_VALUE_METRICS_PATH
    )
    resolved_peak_hour_prediction_path = (
        Path(peak_hour_prediction_path)
        if peak_hour_prediction_path is not None
        else DEFAULT_PEAK_HOUR_PREDICTION_PATH
    )
    resolved_peak_hour_metrics_path = (
        Path(peak_hour_metrics_path)
        if peak_hour_metrics_path is not None
        else DEFAULT_PEAK_HOUR_METRICS_PATH
    )
    resolved_best_peak_value_path = (
        Path(best_peak_value_path) if best_peak_value_path is not None else DEFAULT_BEST_PEAK_VALUE_PATH
    )
    resolved_best_peak_hour_path = (
        Path(best_peak_hour_path) if best_peak_hour_path is not None else DEFAULT_BEST_PEAK_HOUR_PATH
    )
    resolved_selected_dataset_path = (
        Path(selected_dataset_path)
        if selected_dataset_path is not None
        else DEFAULT_SELECTED_DATASET_PATH
    )
    resolved_report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    source_df = pd.read_csv(resolved_source_data_path)
    value_predictions = pd.read_csv(resolved_peak_value_prediction_path)
    value_metrics = pd.read_csv(resolved_peak_value_metrics_path)
    hour_predictions = pd.read_csv(resolved_peak_hour_prediction_path)
    hour_metrics = pd.read_csv(resolved_peak_hour_metrics_path)

    best_value = select_best_peak_value_baselines(value_metrics)
    best_hour = select_best_peak_hour_baselines(hour_metrics)
    selected_dataset = build_dataset_with_selected_baselines(
        source_df,
        value_predictions,
        hour_predictions,
        best_value,
        best_hour,
    )

    for path in (
        resolved_best_peak_value_path,
        resolved_best_peak_hour_path,
        resolved_selected_dataset_path,
        resolved_report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    best_value.to_csv(resolved_best_peak_value_path, index=False)
    best_hour.to_csv(resolved_best_peak_hour_path, index=False)
    selected_dataset.to_csv(resolved_selected_dataset_path, index=False)
    if plot_selected_best_baselines:
        plot_selected_best_baseline_prediction_batch(
            hourly_csv=hourly_csv,
            selected_baseline_csv=resolved_selected_dataset_path,
            output_root=plot_output_root,
            dataset_name=dataset_name,
            split=plot_split,
            sample_count=plot_sample_count,
        )

    report = build_report(
        best_value,
        best_hour,
        selected_dataset,
        [
            resolved_source_data_path,
            resolved_peak_value_prediction_path,
            resolved_peak_value_metrics_path,
            resolved_peak_hour_prediction_path,
            resolved_peak_hour_metrics_path,
        ],
        [
            resolved_best_peak_value_path,
            resolved_best_peak_hour_path,
            resolved_selected_dataset_path,
            resolved_report_path,
        ],
    )
    resolved_report_path.write_text(report, encoding="utf-8")

    return best_value, best_hour, selected_dataset, report


def main() -> None:
    best_value, best_hour, selected_dataset, _ = run_select_best_baselines()
    print(f"best_peak_value_path: {DEFAULT_BEST_PEAK_VALUE_PATH}")
    print(f"best_peak_hour_path: {DEFAULT_BEST_PEAK_HOUR_PATH}")
    print(f"selected_dataset_path: {DEFAULT_SELECTED_DATASET_PATH}")
    print(f"report_path: {DEFAULT_REPORT_PATH}")
    print(f"best_peak_value_groups: {len(best_value)}")
    print(f"best_peak_hour_groups: {len(best_hour)}")
    print(f"selected_dataset_rows: {len(selected_dataset)}")
    print(f"baseline_peak_missing: {int(selected_dataset[BASELINE_PEAK_COLUMN].isna().sum())}")


if __name__ == "__main__":
    main()
