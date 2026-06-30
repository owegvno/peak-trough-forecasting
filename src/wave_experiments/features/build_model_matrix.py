"""Build a full numeric model matrix from residual-learning feature data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd


INPUT_PATH = Path("实验输出/results/features/dataset_with_residual_targets.csv")
MODEL_MATRIX_OUTPUT_PATH = Path("实验输出/results/features/model_matrix_seq96_pred336.csv")
FEATURE_COLUMNS_OUTPUT_PATH = Path(
    "实验输出/results/features/feature_columns_seq96_pred336.txt"
)
REPORT_PATH = Path("实验输出/reports/模型特征表报告.md")

SAMPLE_ID_COLUMN = "样本ID"
START_DATE_COLUMN = "预测起点日期"
SPLIT_COLUMN = "数据集划分"
TARGET_VARIABLE_COLUMN = "目标变量"
HORIZON_COLUMN = "预测天数"

TRAIN_SPLIT = "训练"
VAL_SPLIT = "验证"
TEST_SPLIT = "测试"
EXPECTED_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

ID_COLUMNS = (
    SAMPLE_ID_COLUMN,
    START_DATE_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
)
LABEL_COLUMNS = (
    "目标峰值",
    "target_peak_residual",
    "目标峰值小时",
)
LEGACY_LABEL_COLUMNS = (
    "目标谷值",
    "目标谷值残差",
    "目标谷值小时",
    "目标峰值残差",
)
NUMERIC_BASELINE_COLUMNS = (
    "baseline_peak",
    "baseline_peak_hour",
)
CATEGORICAL_BASELINE_COLUMNS = (
    "peak_value_baseline_name",
    "peak_hour_baseline_name",
)
BASELINE_COLUMNS = (*NUMERIC_BASELINE_COLUMNS, *CATEGORICAL_BASELINE_COLUMNS)
CATEGORY_COLUMNS = (TARGET_VARIABLE_COLUMN, *CATEGORICAL_BASELINE_COLUMNS)

FORBIDDEN_FEATURE_COLUMNS = (
    *LABEL_COLUMNS,
    *LEGACY_LABEL_COLUMNS,
    "基线峰值",
    "peak_hour_ordinary_error",
    "peak_hour_circular_error",
    "peak_hour_within_2h",
)
MATRIX_EXCLUDED_COLUMNS = (
    *LEGACY_LABEL_COLUMNS,
    "基线峰值",
    "peak_hour_ordinary_error",
    "peak_hour_circular_error",
    "peak_hour_within_2h",
)

REQUIRED_COLUMNS = (
    *ID_COLUMNS,
    *LABEL_COLUMNS,
    *NUMERIC_BASELINE_COLUMNS,
    *CATEGORICAL_BASELINE_COLUMNS,
)
REQUIRED_CALENDAR_COLUMNS = (
    "日历_星期",
    "日历_月份",
    "日历_年内日序",
    "日历_是否周末",
)
REQUIRED_FEATURE_PATTERNS = (
    "过去96小时",
    "过去第1天",
    "历史峰值_",
    "历史峰值小时_",
    "峰谷差",
)
VARIABLE_PREFIXES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
MIN_FULL_FEATURE_COUNT = 100


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


def _existing_columns(columns: Sequence[str], candidates: Iterable[str]) -> List[str]:
    column_set = set(columns)
    return [column for column in candidates if column in column_set]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = _missing_columns(df, columns)
    if missing:
        raise ValueError(f"{label} 缺少必要字段：{', '.join(missing)}")


def _columns_containing(columns: Sequence[str], pattern: str) -> List[str]:
    return [column for column in columns if pattern in column]


def _columns_starting_with(columns: Sequence[str], prefix: str) -> List[str]:
    return [column for column in columns if column.startswith(prefix)]


def _validate_splits(df: pd.DataFrame) -> None:
    split_values = set(df[SPLIT_COLUMN].dropna().astype(str))
    missing = [split for split in EXPECTED_SPLITS if split not in split_values]
    if missing:
        raise ValueError(f"{SPLIT_COLUMN} 缺少必要划分：{', '.join(missing)}")


def _validate_unique_sample_keys(df: pd.DataFrame) -> None:
    duplicated = df.duplicated(list(ID_COLUMNS)).sum()
    if duplicated:
        raise ValueError(f"样本键存在重复：{int(duplicated)} 行")


def _validate_full_feature_input(df: pd.DataFrame) -> None:
    """Reject skinny inputs before model-matrix generation."""

    _require_columns(df, REQUIRED_COLUMNS, "残差学习数据集")
    missing_calendar = _missing_columns(df, REQUIRED_CALENDAR_COLUMNS)
    missing_patterns = [
        pattern for pattern in REQUIRED_FEATURE_PATTERNS if not _columns_containing(df.columns, pattern)
    ]
    missing_variable_prefixes = [
        prefix for prefix in VARIABLE_PREFIXES if not _columns_starting_with(df.columns, f"{prefix}_")
    ]

    if missing_calendar or missing_patterns or missing_variable_prefixes:
        details = []
        if missing_calendar:
            details.append(f"缺少日历字段：{', '.join(missing_calendar)}")
        if missing_patterns:
            details.append(f"缺少历史特征模式：{', '.join(missing_patterns)}")
        if missing_variable_prefixes:
            details.append(f"缺少跨变量历史特征：{', '.join(missing_variable_prefixes)}")
        raise ValueError("缺少完整历史/日历特征，拒绝生成瘦身版模型矩阵；" + "；".join(details))


def _classify_feature_counts(feature_columns: Sequence[str]) -> Dict[str, int]:
    counts = {
        "预测天数": int(HORIZON_COLUMN in feature_columns),
        "数值基线": sum(column in NUMERIC_BASELINE_COLUMNS for column in feature_columns),
        "类别 one-hot": sum("__" in column for column in feature_columns),
        "日历特征": sum(column.startswith("日历_") for column in feature_columns),
        "过去96小时统计": sum("过去96小时" in column for column in feature_columns),
        "过去4天自然日统计": sum("过去第" in column for column in feature_columns),
        "历史峰值聚合": sum("历史峰值_" in column for column in feature_columns),
        "历史峰值小时聚合": sum("历史峰值小时_" in column for column in feature_columns),
        "峰谷差历史统计": sum("峰谷差" in column for column in feature_columns),
        "跨变量历史统计": sum(
            any(column.startswith(f"{prefix}_") for prefix in VARIABLE_PREFIXES)
            for column in feature_columns
        ),
    }
    counted = set()
    for column in feature_columns:
        if (
            column == HORIZON_COLUMN
            or column in NUMERIC_BASELINE_COLUMNS
            or "__" in column
            or column.startswith("日历_")
            or any(column.startswith(f"{prefix}_") for prefix in VARIABLE_PREFIXES)
        ):
            counted.add(column)
    counts["其他数值特征"] = len([column for column in feature_columns if column not in counted])
    return counts


def _validate_required_feature_categories(feature_columns: Sequence[str]) -> None:
    checks: Mapping[str, bool] = {
        HORIZON_COLUMN: HORIZON_COLUMN in feature_columns,
        "baseline_peak": "baseline_peak" in feature_columns,
        "baseline_peak_hour": "baseline_peak_hour" in feature_columns,
        "日历_星期": "日历_星期" in feature_columns,
        "日历_月份": "日历_月份" in feature_columns,
        "过去96小时": any("过去96小时" in column for column in feature_columns),
        "过去第1天": any("过去第1天" in column for column in feature_columns),
        "历史峰值_": any("历史峰值_" in column for column in feature_columns),
        "历史峰值小时_": any("历史峰值小时_" in column for column in feature_columns),
        "峰谷差": any("峰谷差" in column for column in feature_columns),
        "目标变量 one-hot": any(column.startswith("目标变量__") for column in feature_columns),
    }
    missing = [label for label, ok in checks.items() if not ok]
    if missing:
        raise ValueError(f"完整模型特征缺少必要类别：{', '.join(missing)}")

    for prefix in VARIABLE_PREFIXES:
        if not any(column.startswith(f"{prefix}_") for column in feature_columns):
            raise ValueError(f"完整模型特征缺少跨变量历史统计：{prefix}")

    if len(feature_columns) <= MIN_FULL_FEATURE_COUNT:
        raise ValueError(
            f"最终特征数量过少：{len(feature_columns)}，可能退化为瘦身版；"
            f"阈值为 {MIN_FULL_FEATURE_COUNT}"
        )


def _numeric_feature_source_columns(df: pd.DataFrame) -> List[str]:
    excluded = (set(ID_COLUMNS) - {HORIZON_COLUMN}) | set(FORBIDDEN_FEATURE_COLUMNS) | set(
        CATEGORY_COLUMNS
    )
    columns = []
    for column in df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def _coerce_numeric_features(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    cleaned_columns: Dict[str, pd.Series] = {}
    fill_records: List[Dict[str, object]] = []
    train_mask = df[SPLIT_COLUMN].eq(TRAIN_SPLIT)

    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").astype("float64")
        invalid_mask = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
        train_values = values[train_mask & ~invalid_mask]
        if len(train_values) > 0:
            fill_value = float(train_values.median())
            fill_source = "训练集 median"
        else:
            global_values = values[~invalid_mask]
            if len(global_values) > 0:
                fill_value = float(global_values.median())
                fill_source = "全表 median"
            else:
                fill_value = 0.0
                fill_source = "常数 0"

        cleaned_columns[column] = values.mask(invalid_mask, fill_value).astype("float64")
        fill_records.append(
            {
                "column": column,
                "invalid_count": int(invalid_mask.sum()),
                "fill_value": fill_value,
                "fill_source": fill_source,
            }
        )

    return pd.DataFrame(cleaned_columns, index=df.index), fill_records


def _category_values(series: pd.Series) -> pd.Series:
    values = series.fillna("__missing__").astype(str).str.strip()
    return values.mask(values.eq(""), "__missing__")


def _encode_categorical_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    encoded_frames = []
    for column in columns:
        values = _category_values(df[column])
        categories = sorted(values.unique(), key=str)
        categorized = pd.Categorical(values, categories=categories)
        encoded = pd.get_dummies(categorized, prefix=column, prefix_sep="__").astype("int8")
        encoded.index = df.index
        encoded_frames.append(encoded)
    if not encoded_frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(encoded_frames, axis=1)


def _validate_numeric_features(matrix: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    numeric = matrix.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"特征列存在无法转为数值或缺失：{', '.join(bad_columns)}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_columns = numeric.columns[~np.isfinite(values).all(axis=0)].tolist()
        raise ValueError(f"特征列存在 Inf：{', '.join(bad_columns)}")


def _split_feature_columns_are_consistent(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
) -> bool:
    expected = list(feature_columns)
    for _, split_frame in matrix.groupby(SPLIT_COLUMN, sort=False):
        if list(split_frame.loc[:, expected].columns) != expected:
            return False
    return True


def _format_number(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _list_text(values: Sequence[str]) -> str:
    if not values:
        return "无"
    return "、".join(f"`{value}`" for value in values)


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    table_rows = []
    for row in rows:
        table_rows.append(
            "| " + " | ".join(_format_number(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join([header, separator, *table_rows])


def _nan_inf_counts(matrix: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[int, int]:
    numeric = matrix.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    return int(numeric.isna().sum().sum()), int((~np.isfinite(values)).sum())


def build_report(
    source: pd.DataFrame,
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    numeric_source_columns: Sequence[str],
    categorical_source_columns: Sequence[str],
    fill_records: Sequence[Mapping[str, object]],
    actual_excluded_columns: Sequence[str],
    input_path: Union[Path, str],
    matrix_output_path: Union[Path, str],
    feature_columns_output_path: Union[Path, str],
) -> str:
    """Create a markdown report describing the model matrix contract."""

    input_rows, input_columns = source.shape
    matrix_rows, matrix_columns = matrix.shape
    existing_id_columns = _existing_columns(source.columns, ID_COLUMNS)
    existing_label_columns = _existing_columns(source.columns, (*LABEL_COLUMNS, *LEGACY_LABEL_COLUMNS))
    existing_baseline_columns = _existing_columns(source.columns, BASELINE_COLUMNS)
    existing_forbidden_columns = _existing_columns(source.columns, FORBIDDEN_FEATURE_COLUMNS)
    feature_count_rows = [
        {"类别": label, "数量": count}
        for label, count in _classify_feature_counts(feature_columns).items()
    ]
    split_rows = [
        {
            "split": split,
            "row_count": int((matrix[SPLIT_COLUMN] == split).sum()),
            "feature_count": len(feature_columns),
        }
        for split in EXPECTED_SPLITS
    ]
    invalid_fill_rows = [
        {
            "字段": record["column"],
            "无效值数量": record["invalid_count"],
            "填充值": record["fill_value"],
            "来源": record["fill_source"],
        }
        for record in fill_records
        if int(record["invalid_count"]) > 0
    ]
    fill_section = (
        _markdown_table(invalid_fill_rows, ["字段", "无效值数量", "填充值", "来源"])
        if invalid_fill_rows
        else "所有数值特征源字段均无 NaN/Inf；未触发实际填充。"
    )

    nan_count, inf_count = _nan_inf_counts(matrix, feature_columns)
    split_consistent = _split_feature_columns_are_consistent(matrix, feature_columns)
    forbidden_overlap = [column for column in feature_columns if column in FORBIDDEN_FEATURE_COLUMNS]
    label_overlap = [
        column
        for column in feature_columns
        if column in set(LABEL_COLUMNS) | set(LEGACY_LABEL_COLUMNS)
    ]
    old_baseline_overlap = [column for column in feature_columns if column == "基线峰值"]

    lines = [
        "# 模型特征表报告",
        "",
        "## 输入输出",
        "",
        f"- 输入：`{input_path}`",
        f"- 完整特征矩阵：`{matrix_output_path}`",
        f"- 特征列清单：`{feature_columns_output_path}`",
        f"- 输入行数、列数：{input_rows} 行，{input_columns} 列",
        f"- 输出矩阵行数、列数：{matrix_rows} 行，{matrix_columns} 列",
        f"- 最终特征数：{len(feature_columns)}",
        "",
        "## 字段划分",
        "",
        f"- ID 字段清单：{_list_text(existing_id_columns)}",
        f"- 标签字段清单：{_list_text(existing_label_columns)}",
        f"- 基线字段清单：{_list_text(existing_baseline_columns)}",
        f"- 类别字段清单：{_list_text(list(categorical_source_columns))}",
        f"- 数值特征字段数：{len(numeric_source_columns)}",
        f"- 禁止字段清单：{_list_text(list(FORBIDDEN_FEATURE_COLUMNS))}",
        f"- 输入中存在的禁止字段：{_list_text(existing_forbidden_columns)}",
        f"- 实际排除字段（特征列）：{_list_text(list(actual_excluded_columns))}",
        "",
        "## 各类特征数量",
        "",
        _markdown_table(feature_count_rows, ["类别", "数量"]),
        "",
        "## NaN/Inf 处理策略",
        "",
        "数值特征源字段先转为数值，将 NaN、非数值、正负 Inf 视为无效值；优先使用训练集 median 填充，训练集不可用时使用全表 median，再不可用时使用常数 0。",
        "",
        fill_section,
        "",
        "## split 行数和特征一致性",
        "",
        _markdown_table(split_rows, ["split", "row_count", "feature_count"]),
        "",
        f"- 训练、验证、测试三个 split 的特征列完全一致：{'通过' if split_consistent else '失败'}",
        "",
        "## 信息泄露检查",
        "",
        f"- 标签字段不进入特征列：{'通过' if not label_overlap else '失败'}",
        f"- 标签字段与特征列交集：{_list_text(label_overlap)}",
        f"- 禁止字段不进入特征列：{'通过' if not forbidden_overlap else '失败'}",
        f"- 禁止字段与特征列交集：{_list_text(forbidden_overlap)}",
        f"- 旧基线字段 `基线峰值` 不进入特征列：{'通过' if not old_baseline_overlap else '失败'}",
        f"- 特征 NaN 数量：{nan_count}",
        f"- 特征 Inf 数量：{inf_count}",
        "",
        "## 特征列",
        "",
        "\n".join(f"- `{column}`" for column in feature_columns),
        "",
    ]
    return "\n".join(lines)


def build_model_matrix(
    df: pd.DataFrame,
    input_path: Union[Path, str] = INPUT_PATH,
    matrix_output_path: Union[Path, str] = MODEL_MATRIX_OUTPUT_PATH,
    feature_columns_output_path: Union[Path, str] = FEATURE_COLUMNS_OUTPUT_PATH,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Return the full model matrix, final numeric feature list, and report text."""

    _validate_full_feature_input(df)
    _validate_splits(df)
    _validate_unique_sample_keys(df)

    source = df.copy()
    actual_excluded_columns = _existing_columns(source.columns, FORBIDDEN_FEATURE_COLUMNS)
    matrix_excluded_columns = _existing_columns(source.columns, MATRIX_EXCLUDED_COLUMNS)
    raw_output_columns = [column for column in source.columns if column not in matrix_excluded_columns]
    raw_matrix = source.loc[:, raw_output_columns].copy()

    numeric_source_columns = _numeric_feature_source_columns(source)
    categorical_source_columns = _existing_columns(source.columns, CATEGORY_COLUMNS)
    numeric_features, fill_records = _coerce_numeric_features(source, numeric_source_columns)
    categorical_features = _encode_categorical_features(source, categorical_source_columns)

    feature_frame = pd.concat([numeric_features, categorical_features], axis=1)
    feature_columns = feature_frame.columns.tolist()
    _validate_required_feature_categories(feature_columns)

    matrix_base = raw_matrix.drop(columns=[column for column in feature_columns if column in raw_matrix.columns])
    matrix = pd.concat([matrix_base, feature_frame], axis=1)

    overlap = [column for column in feature_columns if column in FORBIDDEN_FEATURE_COLUMNS]
    if overlap:
        raise ValueError(f"禁止字段不能进入特征列：{', '.join(overlap)}")
    _validate_numeric_features(matrix, feature_columns)
    if not _split_feature_columns_are_consistent(matrix, feature_columns):
        raise ValueError("训练、验证、测试三个 split 的特征列不一致")

    report = build_report(
        source=source,
        matrix=matrix,
        feature_columns=feature_columns,
        numeric_source_columns=numeric_source_columns,
        categorical_source_columns=categorical_source_columns,
        fill_records=fill_records,
        actual_excluded_columns=actual_excluded_columns,
        input_path=input_path,
        matrix_output_path=matrix_output_path,
        feature_columns_output_path=feature_columns_output_path,
    )
    return matrix, feature_columns, report


def run_build_model_matrix(
    input_path: Union[Path, str] = INPUT_PATH,
    matrix_output_path: Union[Path, str] = MODEL_MATRIX_OUTPUT_PATH,
    feature_columns_output_path: Union[Path, str] = FEATURE_COLUMNS_OUTPUT_PATH,
    report_path: Union[Path, str] = REPORT_PATH,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Read residual-learning data, then write matrix, feature list, and report."""

    resolved_input_path = Path(input_path)
    resolved_matrix_output_path = Path(matrix_output_path)
    resolved_feature_columns_output_path = Path(feature_columns_output_path)
    resolved_report_path = Path(report_path)

    source = pd.read_csv(resolved_input_path)
    matrix, feature_columns, report = build_model_matrix(
        source,
        input_path=resolved_input_path,
        matrix_output_path=resolved_matrix_output_path,
        feature_columns_output_path=resolved_feature_columns_output_path,
    )

    resolved_matrix_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_feature_columns_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(resolved_matrix_output_path, index=False)
    resolved_feature_columns_output_path.write_text(
        "\n".join(feature_columns) + "\n",
        encoding="utf-8",
    )
    resolved_report_path.write_text(report, encoding="utf-8")
    return matrix, feature_columns, report


def main() -> None:
    matrix, feature_columns, _ = run_build_model_matrix()
    print(f"已写出模型特征矩阵：{MODEL_MATRIX_OUTPUT_PATH}，行数={len(matrix)}")
    print(f"已写出特征列清单：{FEATURE_COLUMNS_OUTPUT_PATH}，特征数={len(feature_columns)}")
    print(f"已写出模型特征表报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
