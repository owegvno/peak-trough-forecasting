from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

from .config import TARGET_COLUMNS
from .peak_dataset import write_csv


TARGET_FIELD_MAP = {
    "目标峰值": "目标峰值",
    "目标峰值残差": "目标峰值残差",
    "目标峰值小时": "目标峰值小时",
    "目标谷值": "目标谷值",
    "目标谷值残差": "目标谷值残差",
    "目标谷值小时": "目标谷值小时",
    "日历_星期": "日历_星期",
    "日历_月份": "日历_月份",
    "日历_年内日序": "日历_年内日序",
    "日历_是否周末": "日历_是否周末",
}
LONG_BASE_COLUMNS = [
    "样本ID",
    "预测起点日期",
    "目标变量",
    "预测天数",
    "基线峰值",
    "数据集划分",
    "目标峰值",
    "目标峰值残差",
    "目标峰值小时",
    "目标谷值",
    "目标谷值残差",
    "目标谷值小时",
    "日历_星期",
    "日历_月份",
    "日历_年内日序",
    "日历_是否周末",
]


def parse_dataset_window(dataset_dir: str | Path, seq_days: int | None = None, pred_days: int | None = None) -> tuple[int, int]:
    if seq_days is not None and pred_days is not None:
        return seq_days, pred_days
    match = re.search(r"_pred(?P<pred>\d+)_seq(?P<seq>\d+)$", Path(dataset_dir).name)
    if not match:
        raise ValueError("无法从数据集目录名解析 seq/pred 参数，请显式传入 seq_days 和 pred_days")
    return int(match.group("seq")), int(match.group("pred"))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_file_for(dataset_dir: Path, target_col: str, seq_days: int, pred_days: int, split: str = "") -> Path:
    suffix = f"_{split}" if split else ""
    return dataset_dir / target_col / f"峰谷预测样本_seq{seq_days * 24}_pred{pred_days * 24}_{target_col}{suffix}.csv"


def feature_columns(row: dict[str, str], source_col: str) -> list[str]:
    return [name for name in row if name.startswith(f"{source_col}_") and not name.startswith("第")]


def rows_by_sample(rows: list[dict[str, str]]) -> OrderedDict[str, dict[str, str]]:
    grouped: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in rows:
        grouped[row["样本ID"]] = row
    return grouped


def merge_feature_rows(
    dataset_dir: Path,
    target_cols: list[str],
    seq_days: int,
    pred_days: int,
    split: str = "",
) -> tuple[dict[str, OrderedDict[str, dict[str, str]]], list[str]]:
    by_col: dict[str, OrderedDict[str, dict[str, str]]] = {}
    feature_names: list[str] = []
    for col in target_cols:
        path = sample_file_for(dataset_dir, col, seq_days, pred_days, split)
        rows = read_csv_rows(path)
        grouped = rows_by_sample(rows)
        by_col[col] = grouped
        if rows:
            for name in feature_columns(rows[0], col):
                if name not in feature_names:
                    feature_names.append(name)
    return by_col, feature_names


def horizon_value(row: dict[str, str], horizon: int, suffix: str) -> str:
    return row.get(f"第{horizon}天_{suffix}", "")


def calendar_values(forecast_start_date: str, horizon: int) -> dict[str, str]:
    target_date = datetime.strptime(forecast_start_date, "%Y-%m-%d").date() + timedelta(days=horizon - 1)
    return {
        "日历_星期": str(target_date.weekday()),
        "日历_月份": str(target_date.month),
        "日历_年内日序": str(target_date.timetuple().tm_yday),
        "日历_是否周末": str(int(target_date.weekday() >= 5)),
    }


def target_long_rows(
    target_col: str,
    all_rows_by_col: dict[str, OrderedDict[str, dict[str, str]]],
    target_cols: list[str],
    pred_days: int,
) -> list[dict[str, str]]:
    target_samples = all_rows_by_col[target_col]
    rows: list[dict[str, str]] = []
    for sample_id, target_row in target_samples.items():
        feature_source_rows = {col: all_rows_by_col[col][sample_id] for col in target_cols if sample_id in all_rows_by_col[col]}
        for horizon in range(1, pred_days + 1):
            output: dict[str, str] = {
                "样本ID": sample_id,
                "预测起点日期": target_row["预测起点日期"],
                "目标变量": target_col,
                "预测天数": str(horizon),
                "基线峰值": target_row["基线峰值"],
                "数据集划分": target_row["数据集划分"],
            }
            for suffix, output_name in TARGET_FIELD_MAP.items():
                output[output_name] = horizon_value(target_row, horizon, suffix)
            output.update(calendar_values(target_row["预测起点日期"], horizon))
            for source_col in target_cols:
                source_row = feature_source_rows.get(source_col)
                if not source_row:
                    continue
                for name in feature_columns(source_row, source_col):
                    output[name] = source_row[name]
            rows.append(output)
    return rows


def collect_fieldnames(rows: list[dict[str, str]], feature_names: list[str]) -> list[str]:
    names = list(LONG_BASE_COLUMNS)
    for name in feature_names:
        if name not in names:
            names.append(name)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def convert_dataset_to_long_tables(
    dataset_dir: str | Path,
    target_cols: list[str] | None = None,
    seq_days: int | None = None,
    pred_days: int | None = None,
    split: str = "",
    output_subdir: str = "长表",
) -> dict[str, object]:
    dataset_path = Path(dataset_dir)
    actual_seq_days, actual_pred_days = parse_dataset_window(dataset_path, seq_days, pred_days)
    cols = target_cols or list(TARGET_COLUMNS)
    all_rows_by_col, feature_names = merge_feature_rows(dataset_path, cols, actual_seq_days, actual_pred_days, split)
    output_dir = dataset_path / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, int] = {}
    all_long_rows: list[dict[str, str]] = []
    for target_col in cols:
        rows = target_long_rows(target_col, all_rows_by_col, cols, actual_pred_days)
        fieldnames = collect_fieldnames(rows, feature_names)
        target_dir = output_dir / target_col
        base_name = f"峰谷预测长表_seq{actual_seq_days * 24}_pred{actual_pred_days * 24}_{target_col}"
        write_csv(target_dir / f"{base_name}.csv", rows, fieldnames)
        for split_name in ["训练", "验证", "测试"]:
            split_rows = [row for row in rows if row.get("数据集划分") == split_name]
            write_csv(target_dir / f"{base_name}_{split_name}.csv", split_rows, fieldnames)
        row_counts[target_col] = len(rows)
        all_long_rows.extend(rows)

    all_fieldnames = collect_fieldnames(all_long_rows, feature_names)
    all_name = f"峰谷预测长表_seq{actual_seq_days * 24}_pred{actual_pred_days * 24}_全部变量"
    write_csv(output_dir / f"{all_name}.csv", all_long_rows, all_fieldnames)
    for split_name in ["训练", "验证", "测试"]:
        split_rows = [row for row in all_long_rows if row.get("数据集划分") == split_name]
        write_csv(output_dir / f"{all_name}_{split_name}.csv", split_rows, all_fieldnames)

    return {
        "output_dir": output_dir,
        "target_cols": cols,
        "seq_days": actual_seq_days,
        "pred_days": actual_pred_days,
        "row_counts": row_counts,
        "total_rows": len(all_long_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将现有每变量宽表样本转换为训练用长表")
    parser.add_argument("--dataset-dir", default="数据集/ETTH1_pred14_seq4", help="现有数据集目录")
    parser.add_argument("--target-cols", nargs="*", default=None, help="要转换的变量，默认全部变量")
    parser.add_argument("--seq-days", type=int, default=None, help="输入自然日数量；默认从目录名解析")
    parser.add_argument("--pred-days", type=int, default=None, help="预测自然日数量；默认从目录名解析")
    parser.add_argument("--split", default="", help="只读取某个划分的宽表；默认读取全量宽表")
    parser.add_argument("--output-subdir", default="长表", help="输出子目录名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_dataset_to_long_tables(
        args.dataset_dir,
        target_cols=args.target_cols,
        seq_days=args.seq_days,
        pred_days=args.pred_days,
        split=args.split,
        output_subdir=args.output_subdir,
    )
    print(f"长表生成完成: {result['output_dir']}")
    print(f"总行数: {result['total_rows']}")
    for col, count in result["row_counts"].items():
        print(f"  {col}: {count}")


if __name__ == "__main__":
    main()
