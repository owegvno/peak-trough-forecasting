"""Validate the long-table peak prediction dataset and write reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


INPUT_CSV = Path("数据集/ETTH1_pred14_seq4/长表/峰谷预测长表_seq96_pred336_全部变量.csv")
REPORT_PATH = Path("实验输出/reports/数据集验证报告.md")
SUMMARY_PATH = Path("实验输出/results/dataset_validation_summary.csv")

EXPECTED_SPLITS = ("训练", "验证", "测试")
EXPECTED_HORIZONS = tuple(range(1, 15))
EXPECTED_TARGET_VARIABLES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")

REQUIRED_COLUMNS = ("目标变量", "预测天数", "目标峰值", "目标峰值小时")
METADATA_COLUMNS = {
    "样本ID",
    "预测起点日期",
    "数据集划分",
    "目标变量",
    "预测天数",
}
LABEL_COLUMNS = {
    "目标峰值",
    "目标峰值残差",
    "目标峰值小时",
    "目标谷值",
    "目标谷值残差",
    "目标谷值小时",
}
LEAKAGE_KEYWORDS = (
    "未来",
    "真实目标",
    "预测期",
    "标签",
    "label",
    "target",
    "future",
    "horizon",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str


@dataclass(frozen=True)
class ValidationResult:
    input_path: Path
    row_count: int
    column_count: int
    feature_columns: Sequence[str]
    split_values_check: CheckResult
    required_fields_check: CheckResult
    horizon_coverage: CheckResult
    target_variable_coverage: CheckResult
    peak_hour_range: CheckResult
    peak_value_numeric: CheckResult
    feature_nan_inf_check: CheckResult
    leakage_check: CheckResult
    time_order_check: CheckResult
    split_date_ranges: Dict[str, Dict[str, str]]
    allow_step3: bool

    @property
    def checks(self) -> Sequence[CheckResult]:
        return (
            self.split_values_check,
            self.required_fields_check,
            self.horizon_coverage,
            self.target_variable_coverage,
            self.peak_hour_range,
            self.peak_value_numeric,
            self.feature_nan_inf_check,
            self.leakage_check,
            self.time_order_check,
        )


def _format_values(values: Iterable[object]) -> str:
    items = [str(value) for value in values]
    return "、".join(items) if items else "无"


def _missing_values(expected: Sequence[object], actual: Sequence[object]) -> List[object]:
    actual_set = set(actual)
    return [value for value in expected if value not in actual_set]


def _extra_values(expected: Sequence[object], actual: Sequence[object]) -> List[object]:
    expected_set = set(expected)
    return [value for value in actual if value not in expected_set]


def get_feature_columns(columns: Sequence[str]) -> List[str]:
    excluded = METADATA_COLUMNS | LABEL_COLUMNS
    return [column for column in columns if column not in excluded]


def _check_split_values(df: pd.DataFrame) -> CheckResult:
    if "数据集划分" not in df.columns:
        return CheckResult("数据集划分取值", False, "缺少 `数据集划分` 列。")

    actual = sorted(df["数据集划分"].dropna().unique().tolist())
    missing = _missing_values(EXPECTED_SPLITS, actual)
    extra = _extra_values(EXPECTED_SPLITS, actual)
    ok = not missing and not extra and not df["数据集划分"].isna().any()
    if ok:
        details = "数据集划分只包含 训练、验证、测试。"
    else:
        details = (
            f"实际取值：{_format_values(actual)}；"
            f"缺失：{_format_values(missing)}；额外：{_format_values(extra)}；"
            f"空值数：{int(df['数据集划分'].isna().sum())}。"
        )
    return CheckResult("数据集划分取值", ok, details)


def _check_required_fields(df: pd.DataFrame) -> CheckResult:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        return CheckResult(
            "样本必要字段完整性",
            False,
            f"缺少必要列：{_format_values(missing_columns)}。",
        )

    null_counts = df.loc[:, REQUIRED_COLUMNS].isna().sum()
    fields_with_nulls = {
        column: int(count) for column, count in null_counts.items() if int(count) > 0
    }
    ok = not fields_with_nulls
    if ok:
        details = "每条样本均包含完整的 目标变量、预测天数、目标峰值、目标峰值小时。"
    else:
        details = "存在空值：" + "；".join(
            f"{column}={count}" for column, count in fields_with_nulls.items()
        )
    return CheckResult("样本必要字段完整性", ok, details)


def _check_horizon_coverage(df: pd.DataFrame) -> CheckResult:
    if "预测天数" not in df.columns:
        return CheckResult("预测天数覆盖", False, "缺少 `预测天数` 列。")

    numeric_horizons = pd.to_numeric(df["预测天数"], errors="coerce")
    actual = sorted(numeric_horizons.dropna().astype(int).unique().tolist())
    missing = _missing_values(EXPECTED_HORIZONS, actual)
    extra = _extra_values(EXPECTED_HORIZONS, actual)
    has_non_numeric = numeric_horizons.isna().any()
    ok = not missing and not extra and not has_non_numeric
    if ok:
        details = "预测天数覆盖 1 到 14。"
    else:
        details = (
            f"实际覆盖：{_format_values(actual)}；"
            f"缺失：{_format_values(missing)}；额外：{_format_values(extra)}；"
            f"非数值或空值数：{int(numeric_horizons.isna().sum())}。"
        )
    return CheckResult("预测天数覆盖", ok, details)


def _check_target_variable_coverage(df: pd.DataFrame) -> CheckResult:
    if "目标变量" not in df.columns:
        return CheckResult("目标变量覆盖", False, "缺少 `目标变量` 列。")

    actual = sorted(df["目标变量"].dropna().unique().tolist())
    missing = _missing_values(EXPECTED_TARGET_VARIABLES, actual)
    extra = _extra_values(EXPECTED_TARGET_VARIABLES, actual)
    ok = not missing and not extra and not df["目标变量"].isna().any()
    if ok:
        details = "覆盖 7 个目标变量：HUFL、HULL、MUFL、MULL、LUFL、LULL、OT。"
    else:
        details = (
            f"实际覆盖 {len(actual)} 个：{_format_values(actual)}；"
            f"缺失：{_format_values(missing)}；额外：{_format_values(extra)}；"
            f"空值数：{int(df['目标变量'].isna().sum())}。"
        )
    return CheckResult("目标变量覆盖", ok, details)


def _check_peak_hour_range(df: pd.DataFrame) -> CheckResult:
    if "目标峰值小时" not in df.columns:
        return CheckResult("目标峰值小时范围", False, "缺少 `目标峰值小时` 列。")

    peak_hours = pd.to_numeric(df["目标峰值小时"], errors="coerce")
    invalid_mask = peak_hours.isna() | (peak_hours < 0) | (peak_hours > 23)
    ok = not invalid_mask.any()
    if ok:
        details = "`目标峰值小时` 全部在 0 到 23。"
    else:
        invalid_examples = df.loc[invalid_mask, "目标峰值小时"].head(10).tolist()
        details = (
            f"`目标峰值小时` 存在 {int(invalid_mask.sum())} 条越界或非数值记录；"
            f"示例：{_format_values(invalid_examples)}。"
        )
    return CheckResult("目标峰值小时范围", ok, details)


def _check_peak_value_numeric(df: pd.DataFrame) -> CheckResult:
    if "目标峰值" not in df.columns:
        return CheckResult("目标峰值数值类型", False, "缺少 `目标峰值` 列。")

    values = pd.to_numeric(df["目标峰值"], errors="coerce")
    invalid_mask = values.isna() | ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    ok = not invalid_mask.any()
    if ok:
        details = "`目标峰值` 全部为有限数值。"
    else:
        details = f"`目标峰值` 存在 {int(invalid_mask.sum())} 条非数值、NaN 或 Inf 记录。"
    return CheckResult("目标峰值数值类型", ok, details)


def _check_feature_nan_inf(df: pd.DataFrame, feature_columns: Sequence[str]) -> CheckResult:
    if not feature_columns:
        return CheckResult("特征列 NaN/Inf", False, "未识别到特征列。")

    feature_df = df.loc[:, feature_columns]
    null_counts = feature_df.isna().sum()
    columns_with_nan = {
        column: int(count) for column, count in null_counts.items() if int(count) > 0
    }

    numeric_feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    inf_mask = np.isinf(numeric_feature_df.to_numpy(dtype=float, na_value=np.nan))
    inf_counts = pd.Series(inf_mask.sum(axis=0), index=feature_columns)
    columns_with_inf = {
        column: int(count) for column, count in inf_counts.items() if int(count) > 0
    }

    ok = not columns_with_nan and not columns_with_inf
    if ok:
        details = f"{len(feature_columns)} 个特征列未发现 NaN 或 Inf。"
    else:
        nan_detail = "、".join(
            f"{column}={count}" for column, count in list(columns_with_nan.items())[:20]
        )
        inf_detail = "、".join(
            f"{column}={count}" for column, count in list(columns_with_inf.items())[:20]
        )
        details = (
            f"存在 NaN 的特征列：{nan_detail or '无'}；"
            f"存在 Inf 的特征列：{inf_detail or '无'}。"
        )
    return CheckResult("特征列 NaN/Inf", ok, details)


def _check_leakage(feature_columns: Sequence[str]) -> CheckResult:
    suspicious = []
    for column in feature_columns:
        lower_column = column.lower()
        if column in LABEL_COLUMNS or any(keyword in lower_column for keyword in LEAKAGE_KEYWORDS):
            suspicious.append(column)
            continue
        if "目标" in column and "目标变量" not in column:
            suspicious.append(column)

    suspicious = sorted(set(suspicious))
    ok = not suspicious
    if ok:
        details = "未发现明显泄露字段。"
    else:
        details = "疑似泄露字段：" + "、".join(suspicious)
    return CheckResult("信息泄露字段", ok, details)


def _check_time_order(df: pd.DataFrame) -> tuple[CheckResult, Dict[str, Dict[str, str]]]:
    required = ("数据集划分", "预测起点日期")
    missing = [column for column in required if column not in df.columns]
    if missing:
        return (
            CheckResult("预测起点日期时间顺序", False, f"缺少列：{_format_values(missing)}。"),
            {},
        )

    with_dates = df.loc[:, required].copy()
    with_dates["预测起点日期"] = pd.to_datetime(with_dates["预测起点日期"], errors="coerce")
    invalid_dates = int(with_dates["预测起点日期"].isna().sum())
    ranges: Dict[str, Dict[str, str]] = {}
    for split in EXPECTED_SPLITS:
        split_dates = with_dates.loc[with_dates["数据集划分"] == split, "预测起点日期"].dropna()
        if split_dates.empty:
            ranges[split] = {"min": "", "max": ""}
            continue
        ranges[split] = {
            "min": split_dates.min().strftime("%Y-%m-%d"),
            "max": split_dates.max().strftime("%Y-%m-%d"),
        }

    has_all_splits = all(ranges.get(split, {}).get("min") for split in EXPECTED_SPLITS)
    ordered = False
    if has_all_splits:
        train_max = pd.Timestamp(ranges["训练"]["max"])
        valid_min = pd.Timestamp(ranges["验证"]["min"])
        valid_max = pd.Timestamp(ranges["验证"]["max"])
        test_min = pd.Timestamp(ranges["测试"]["min"])
        ordered = train_max < valid_min and valid_max < test_min

    ok = invalid_dates == 0 and has_all_splits and ordered
    range_details = "；".join(
        f"{split}={ranges.get(split, {}).get('min', '')} 至 {ranges.get(split, {}).get('max', '')}"
        for split in EXPECTED_SPLITS
    )
    if ok:
        details = f"训练、验证、测试的预测起点日期按时间顺序且不重叠划分（{range_details}）。"
    else:
        details = (
            f"时间顺序检查未通过；日期范围：{range_details}；"
            f"无法解析日期数：{invalid_dates}。"
        )
    return CheckResult("预测起点日期时间顺序", ok, details), ranges


def validate_dataset(df: pd.DataFrame, input_path: Path = INPUT_CSV) -> ValidationResult:
    feature_columns = get_feature_columns(df.columns.tolist())
    split_values_check = _check_split_values(df)
    required_fields_check = _check_required_fields(df)
    horizon_coverage = _check_horizon_coverage(df)
    target_variable_coverage = _check_target_variable_coverage(df)
    peak_hour_range = _check_peak_hour_range(df)
    peak_value_numeric = _check_peak_value_numeric(df)
    feature_nan_inf_check = _check_feature_nan_inf(df, feature_columns)
    leakage_check = _check_leakage(feature_columns)
    time_order_check, split_date_ranges = _check_time_order(df)

    checks = (
        split_values_check,
        required_fields_check,
        horizon_coverage,
        target_variable_coverage,
        peak_hour_range,
        peak_value_numeric,
        feature_nan_inf_check,
        leakage_check,
        time_order_check,
    )
    allow_step3 = all(check.ok for check in checks)

    return ValidationResult(
        input_path=input_path,
        row_count=len(df),
        column_count=len(df.columns),
        feature_columns=feature_columns,
        split_values_check=split_values_check,
        required_fields_check=required_fields_check,
        horizon_coverage=horizon_coverage,
        target_variable_coverage=target_variable_coverage,
        peak_hour_range=peak_hour_range,
        peak_value_numeric=peak_value_numeric,
        feature_nan_inf_check=feature_nan_inf_check,
        leakage_check=leakage_check,
        time_order_check=time_order_check,
        split_date_ranges=split_date_ranges,
        allow_step3=allow_step3,
    )


def _status(ok: bool) -> str:
    return "通过" if ok else "未通过"


def build_markdown_report(result: ValidationResult) -> str:
    lines = [
        "# 数据集验证报告",
        "",
        "## 基本信息",
        "",
        f"- 输入文件：`{result.input_path}`",
        f"- 样本数：{result.row_count}",
        f"- 列数：{result.column_count}",
        f"- 特征列数：{len(result.feature_columns)}",
        "",
        "## 验证结论",
        "",
        f"- 允许进入第 3 步：{'是' if result.allow_step3 else '否'}",
        "",
        "## 检查结果",
        "",
        "| 检查项 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for check in result.checks:
        lines.append(f"| {check.name} | {_status(check.ok)} | {check.details} |")

    lines.extend(
        [
            "",
            "## 覆盖情况",
            "",
            f"- horizon：{result.horizon_coverage.details}",
            f"- 目标变量：{result.target_variable_coverage.details}",
            f"- 目标峰值小时：{result.peak_hour_range.details}",
            "",
            "## 信息泄露检查",
            "",
            f"- {result.leakage_check.details}",
            "",
            "## 时间划分检查",
            "",
        ]
    )
    for split in EXPECTED_SPLITS:
        split_range = result.split_date_ranges.get(split, {})
        lines.append(
            f"- {split}：{split_range.get('min', '') or '无'} 至 {split_range.get('max', '') or '无'}"
        )
    lines.extend(["", f"- {result.time_order_check.details}", ""])
    return "\n".join(lines)


def write_markdown_report(result: ValidationResult, output_path: Path = REPORT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_report(result), encoding="utf-8")


def write_summary_csv(result: ValidationResult, output_path: Path = SUMMARY_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "check": check.name,
                "passed": check.ok,
                "details": check.details,
            }
            for check in result.checks
        ]
    )
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> int:
    df = pd.read_csv(INPUT_CSV)
    result = validate_dataset(df, INPUT_CSV)
    write_markdown_report(result, REPORT_PATH)
    write_summary_csv(result, SUMMARY_PATH)
    print(f"写入报告：{REPORT_PATH}")
    print(f"写入摘要：{SUMMARY_PATH}")
    print(f"允许进入第 3 步：{'是' if result.allow_step3 else '否'}")
    return 0 if result.allow_step3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
