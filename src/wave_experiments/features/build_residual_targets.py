"""Build residual targets from selected rule baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

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

ID_COLUMNS = (
    "样本ID",
    "预测起点日期",
    "数据集划分",
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
)
LABEL_COLUMNS = (
    TARGET_PEAK_COLUMN,
    TARGET_PEAK_RESIDUAL_COLUMN,
    "目标峰值小时",
)
BASELINE_COLUMNS = (
    BASELINE_PEAK_COLUMN,
    "peak_value_baseline_name",
    "baseline_peak_hour",
    "peak_hour_baseline_name",
)
REQUIRED_COLUMNS = (
    *ID_COLUMNS,
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
    TARGET_PEAK_COLUMN,
    BASELINE_PEAK_COLUMN,
)
DROP_COLUMNS = (
    "目标谷值",
    "目标谷值残差",
    "目标谷值小时",
    "目标峰值残差",
    "基线峰值",
    "peak_hour_ordinary_error",
    "peak_hour_circular_error",
    "peak_hour_within_2h",
)
CORE_FEATURE_CHECKS = (
    ("日历_星期", lambda columns: "日历_星期" in columns),
    ("日历_月份", lambda columns: "日历_月份" in columns),
    ("包含 `过去96小时` 的字段", lambda columns: any("过去96小时" in column for column in columns)),
    ("包含 `过去第1天` 的字段", lambda columns: any("过去第1天" in column for column in columns)),
    ("包含 `历史峰值_` 的字段", lambda columns: any("历史峰值_" in column for column in columns)),
    ("包含 `历史峰值小时_` 的字段", lambda columns: any("历史峰值小时_" in column for column in columns)),
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


def check_core_features(df: pd.DataFrame) -> Dict[str, bool]:
    """Check that the selected-baseline table still contains full model features."""

    columns = set(df.columns)
    results = {name: bool(check(columns)) for name, check in CORE_FEATURE_CHECKS}
    missing = [name for name, ok in results.items() if not ok]
    if missing:
        raise ValueError(
            "输入表缺少完整历史/日历特征，疑似旧的 15 列瘦身表，"
            "不能生成 peak_value 残差学习数据集。缺失检查项："
            + "、".join(missing)
        )
    return results


def _is_feature_column(column: str) -> bool:
    if column.startswith("日历_"):
        return True
    return any(
        pattern in column
        for pattern in (
            "过去96小时",
            "过去第1天",
            "过去第2天",
            "过去第3天",
            "过去第4天",
            "历史峰值_",
            "历史峰值小时_",
            "峰谷差",
        )
    )


def retained_columns(df: pd.DataFrame) -> List[str]:
    """Return ordered columns for the clean peak-value residual learning table."""

    keep_candidates: List[str] = []
    keep_candidates.extend(ID_COLUMNS)
    keep_candidates.extend(LABEL_COLUMNS)
    keep_candidates.extend(BASELINE_COLUMNS)
    keep_candidates.extend(column for column in df.columns if _is_feature_column(column))

    keep: List[str] = []
    seen = set(DROP_COLUMNS)
    for column in keep_candidates:
        if column in df.columns and column not in seen:
            keep.append(column)
            seen.add(column)
    return keep


def dropped_input_columns(df: pd.DataFrame) -> List[str]:
    """Return input columns intentionally excluded from the residual-learning table."""

    keep = set(retained_columns(df))
    return [column for column in df.columns if column not in keep]


def build_residual_target_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a training-ready table with a freshly computed peak residual target."""

    _require_columns(df, REQUIRED_COLUMNS, "规则基线数据集")
    check_core_features(df)

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

    return dataset.loc[:, retained_columns(dataset)].copy()


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
    source: pd.DataFrame,
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
    feature_checks = check_core_features(source)
    dropped_columns = dropped_input_columns(source)
    dropped_columns_text = "、".join(dropped_columns) if dropped_columns else "无"
    feature_check_lines = [
        f"- {name}：{'通过' if ok else '未通过'}" for name, ok in feature_checks.items()
    ]

    lines = [
        "# 残差标签报告",
        "",
        "## 输入输出",
        "",
        f"- 输入：`{input_path}`",
        f"- 输出：`{output_path}`",
        f"- 输入行数：{len(source)}",
        f"- 输入列数：{len(source.columns)}",
        f"- 输出行数：{len(dataset)}",
        f"- 输出列数：{len(dataset.columns)}",
        "",
        "## 保留字段类别",
        "",
        "- ID / split 字段：样本ID、预测起点日期、数据集划分、目标变量、预测天数",
        "- 监督标签字段：目标峰值、target_peak_residual、目标峰值小时",
        "- 规则基线字段：baseline_peak、peak_value_baseline_name、baseline_peak_hour、peak_hour_baseline_name",
        "- 可用特征字段：日历特征、过去 96 小时统计、过去 4 天自然日统计、历史峰值/峰值小时聚合、峰谷差历史统计、跨变量历史统计特征",
        "",
        "## 删除字段清单",
        "",
        dropped_columns_text,
        "",
        "## 核心历史/日历特征检查结果",
        "",
        *feature_check_lines,
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
    report = build_report(source, dataset, stats, resolved_input_path, resolved_output_path)

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
