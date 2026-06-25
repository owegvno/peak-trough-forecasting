"""Build residual targets from selected rule baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd


INPUT_PATH = Path("实验输出/results/baselines/dataset_with_selected_baselines.csv")
OUTPUT_PATH = Path("实验输出/results/features/dataset_with_residual_targets.csv")
REPORT_PATH = Path("实验输出/reports/残差标签报告.md")

TARGET_VARIABLE_COLUMN = "目标变量"
HORIZON_COLUMN = "预测天数"
TARGET_PEAK_COLUMN = "目标峰值"
BASELINE_PEAK_COLUMN = "baseline_peak"
TARGET_PEAK_RESIDUAL_COLUMN = "target_peak_residual"

REQUIRED_COLUMNS = (
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
    TARGET_PEAK_COLUMN,
    BASELINE_PEAK_COLUMN,
)
FLOAT_TOLERANCE = 1e-9


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list:
    return [column for column in columns if column not in df.columns]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = _missing_columns(df, columns)
    if missing:
        raise ValueError(f"{label} 缺少必要字段：{', '.join(missing)}")


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if invalid.any():
        raise ValueError(f"{column} 存在缺失、非数值或 Inf：{int(invalid.sum())} 行")
    return values


def build_residual_target_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a training-ready table with a freshly computed peak residual target."""

    _require_columns(df, REQUIRED_COLUMNS, "规则基线数据集")

    dataset = df.copy()
    target_peak = _numeric_column(dataset, TARGET_PEAK_COLUMN)
    baseline_peak = _numeric_column(dataset, BASELINE_PEAK_COLUMN)

    dataset[TARGET_PEAK_COLUMN] = target_peak
    dataset[BASELINE_PEAK_COLUMN] = baseline_peak
    dataset[TARGET_PEAK_RESIDUAL_COLUMN] = target_peak - baseline_peak

    residual = dataset[TARGET_PEAK_RESIDUAL_COLUMN]
    if residual.isna().any():
        raise ValueError(f"{TARGET_PEAK_RESIDUAL_COLUMN} 存在缺失：{int(residual.isna().sum())} 行")

    errors = (dataset[BASELINE_PEAK_COLUMN] + residual - dataset[TARGET_PEAK_COLUMN]).abs()
    max_error = float(errors.max()) if len(errors) else 0.0
    if max_error > FLOAT_TOLERANCE:
        raise ValueError(
            f"{BASELINE_PEAK_COLUMN} + {TARGET_PEAK_RESIDUAL_COLUMN} 无法还原 "
            f"{TARGET_PEAK_COLUMN}；最大误差 {max_error:.12g}"
        )

    return dataset


def residual_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize residual distribution by target variable and forecast horizon."""

    _require_columns(
        dataset,
        (TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, TARGET_PEAK_RESIDUAL_COLUMN),
        "残差目标数据集",
    )
    residual = _numeric_column(dataset, TARGET_PEAK_RESIDUAL_COLUMN)
    stats_input = dataset.copy()
    stats_input[TARGET_PEAK_RESIDUAL_COLUMN] = residual
    stats_input[HORIZON_COLUMN] = pd.to_numeric(stats_input[HORIZON_COLUMN], errors="coerce")
    if stats_input[HORIZON_COLUMN].isna().any():
        raise ValueError(f"{HORIZON_COLUMN} 存在缺失或非数值")
    stats_input[HORIZON_COLUMN] = stats_input[HORIZON_COLUMN].astype(int)

    stats = (
        stats_input.groupby([TARGET_VARIABLE_COLUMN, HORIZON_COLUMN], dropna=False)[
            TARGET_PEAK_RESIDUAL_COLUMN
        ]
        .agg(row_count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values([TARGET_VARIABLE_COLUMN, HORIZON_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )
    return stats


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
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


def build_report(
    dataset: pd.DataFrame,
    stats: pd.DataFrame,
    input_path: Union[Path, str],
    output_path: Union[Path, str],
) -> str:
    """Create a markdown report for residual target generation."""

    residual = dataset[TARGET_PEAK_RESIDUAL_COLUMN]
    reconstruction_error = (
        dataset[BASELINE_PEAK_COLUMN] + residual - dataset[TARGET_PEAK_COLUMN]
    ).abs()
    max_error = float(reconstruction_error.max()) if len(reconstruction_error) else 0.0
    residual_nan_count = int(residual.isna().sum())

    lines = [
        "# 残差标签报告",
        "",
        "## 输入输出",
        "",
        f"- 输入：`{input_path}`",
        f"- 输出：`{output_path}`",
        f"- 行数：{len(dataset)}",
        f"- 列数：{len(dataset.columns)}",
        "",
        "## 校验结果",
        "",
        f"- 残差公式：`{TARGET_PEAK_RESIDUAL_COLUMN} = {TARGET_PEAK_COLUMN} - {BASELINE_PEAK_COLUMN}`",
        (
            f"- 还原检查：`{BASELINE_PEAK_COLUMN} + {TARGET_PEAK_RESIDUAL_COLUMN}` "
            f"还原 `{TARGET_PEAK_COLUMN}` 的最大绝对误差为 {max_error:.12g}"
        ),
        f"- `{TARGET_PEAK_RESIDUAL_COLUMN}` 缺失数：{residual_nan_count}",
        "",
        "## 残差分布",
        "",
        _markdown_table(
            stats,
            [TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, "row_count", "mean", "std", "min", "max"],
        ),
        "",
    ]
    return "\n".join(lines)


def run_build_residual_targets(
    input_path: Union[Path, str] = INPUT_PATH,
    output_path: Union[Path, str] = OUTPUT_PATH,
    report_path: Union[Path, str] = REPORT_PATH,
) -> Tuple[pd.DataFrame, str]:
    """Read selected-baseline data, write residual-target data and report."""

    resolved_input_path = Path(input_path)
    resolved_output_path = Path(output_path)
    resolved_report_path = Path(report_path)

    source = pd.read_csv(resolved_input_path)
    dataset = build_residual_target_dataset(source)
    stats = residual_distribution(dataset)
    report = build_report(dataset, stats, resolved_input_path, resolved_output_path)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(resolved_output_path, index=False)
    resolved_report_path.write_text(report, encoding="utf-8")
    return dataset, report


def main() -> None:
    dataset, _ = run_build_residual_targets()
    print(f"已写出残差目标数据集：{OUTPUT_PATH}，行数={len(dataset)}")
    print(f"已写出残差标签报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
