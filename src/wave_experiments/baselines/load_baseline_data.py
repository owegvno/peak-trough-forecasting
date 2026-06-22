"""Load the long-table peak/trough dataset for baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


DEFAULT_DATA_PATH = Path("数据集/ETTH1_pred14_seq4/长表/峰谷预测长表_seq96_pred336_全部变量.csv")

SPLIT_COLUMN = "数据集划分"
TRAIN_SPLIT = "训练"
VAL_SPLIT = "验证"
TEST_SPLIT = "测试"
EXPECTED_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

TARGET_VARIABLE_COLUMN = "目标变量"
HORIZON_COLUMN = "预测天数"

ID_COLUMNS = (
    "样本ID",
    "预测起点日期",
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
    SPLIT_COLUMN,
)
BASELINE_COLUMNS = ("基线峰值",)
LABEL_COLUMNS = (
    "目标峰值",
    "目标峰值残差",
    "目标峰值小时",
    "目标谷值",
    "目标谷值残差",
    "目标谷值小时",
)

ID_COLUMNS_ATTR = "id_columns"
LABEL_COLUMNS_ATTR = "label_columns"
FEATURE_COLUMNS_ATTR = "feature_columns"
BASELINE_COLUMNS_ATTR = "baseline_columns"
CALENDAR_COLUMNS_ATTR = "calendar_columns"
HISTORY_FEATURE_COLUMNS_ATTR = "history_feature_columns"


def _existing_columns(columns: Sequence[str], candidates: Iterable[str]) -> List[str]:
    column_set = set(columns)
    return [column for column in candidates if column in column_set]


def get_calendar_columns(columns: Sequence[str]) -> List[str]:
    """Return calendar fields retained from the long table."""

    return [column for column in columns if column.startswith("日历_")]


def get_history_feature_columns(columns: Sequence[str]) -> List[str]:
    """Return historical/statistical fields retained as baseline features."""

    excluded = set(ID_COLUMNS) | set(BASELINE_COLUMNS) | set(LABEL_COLUMNS)
    calendar_columns = set(get_calendar_columns(columns))
    return [
        column
        for column in columns
        if column not in excluded and column not in calendar_columns
    ]


def get_feature_columns(columns: Sequence[str]) -> List[str]:
    """Return model feature columns, excluding future real label fields."""

    baseline_columns = _existing_columns(columns, BASELINE_COLUMNS)
    calendar_columns = get_calendar_columns(columns)
    history_columns = get_history_feature_columns(columns)
    return baseline_columns + calendar_columns + history_columns


def _required_columns() -> Tuple[str, ...]:
    return (*ID_COLUMNS, *BASELINE_COLUMNS, *LABEL_COLUMNS)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in _required_columns() if column not in df.columns]
    if missing:
        missing_text = "、".join(missing)
        raise ValueError(f"输入长表缺少必要字段：{missing_text}")


def _attach_column_metadata(df: pd.DataFrame, all_columns: Sequence[str]) -> pd.DataFrame:
    id_columns = _existing_columns(all_columns, ID_COLUMNS)
    label_columns = _existing_columns(all_columns, LABEL_COLUMNS)
    baseline_columns = _existing_columns(all_columns, BASELINE_COLUMNS)
    calendar_columns = get_calendar_columns(all_columns)
    history_columns = get_history_feature_columns(all_columns)
    feature_columns = get_feature_columns(all_columns)

    df.attrs[ID_COLUMNS_ATTR] = id_columns
    df.attrs[LABEL_COLUMNS_ATTR] = label_columns
    df.attrs[BASELINE_COLUMNS_ATTR] = baseline_columns
    df.attrs[CALENDAR_COLUMNS_ATTR] = calendar_columns
    df.attrs[HISTORY_FEATURE_COLUMNS_ATTR] = history_columns
    df.attrs[FEATURE_COLUMNS_ATTR] = feature_columns
    return df


def load_peak_dataset(
    csv_path: Optional[Union[Path, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the peak/trough long table and split it into train/val/test frames.

    The returned frames retain ID fields, label fields, baseline fields, calendar
    fields, and historical statistical feature fields. Future real labels stay
    in the frames for supervised learning targets, but they are excluded from
    ``df.attrs["feature_columns"]``.
    """

    input_path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    df = pd.read_csv(input_path)
    _validate_columns(df)

    all_columns = list(df.columns)
    train_df = df.loc[df[SPLIT_COLUMN] == TRAIN_SPLIT, all_columns].copy()
    val_df = df.loc[df[SPLIT_COLUMN] == VAL_SPLIT, all_columns].copy()
    test_df = df.loc[df[SPLIT_COLUMN] == TEST_SPLIT, all_columns].copy()

    for split_df in (train_df, val_df, test_df):
        _attach_column_metadata(split_df, all_columns)

    return train_df, val_df, test_df


def _format_distribution(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).sort_index()
    return ", ".join(f"{index}: {count}" for index, count in counts.items())


def _print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"{name} shape: {df.shape}")
    print(f"{name} row_count: {len(df)}")
    print(f"{name} column_count: {len(df.columns)}")
    print(f"{name} target_variable_distribution: {_format_distribution(df[TARGET_VARIABLE_COLUMN])}")
    print(f"{name} horizon_distribution: {_format_distribution(df[HORIZON_COLUMN])}")


def main() -> None:
    train_df, val_df, test_df = load_peak_dataset()
    sample_df = train_df if not train_df.empty else val_df if not val_df.empty else test_df

    print(f"input_path: {DEFAULT_DATA_PATH}")
    print(f"id_columns ({len(sample_df.attrs[ID_COLUMNS_ATTR])}): {', '.join(sample_df.attrs[ID_COLUMNS_ATTR])}")
    print(
        f"label_columns ({len(sample_df.attrs[LABEL_COLUMNS_ATTR])}): "
        f"{', '.join(sample_df.attrs[LABEL_COLUMNS_ATTR])}"
    )
    print(
        f"feature_columns ({len(sample_df.attrs[FEATURE_COLUMNS_ATTR])}): "
        f"{', '.join(sample_df.attrs[FEATURE_COLUMNS_ATTR])}"
    )
    print(
        f"baseline_columns ({len(sample_df.attrs[BASELINE_COLUMNS_ATTR])}): "
        f"{', '.join(sample_df.attrs[BASELINE_COLUMNS_ATTR])}"
    )
    print(
        f"calendar_columns ({len(sample_df.attrs[CALENDAR_COLUMNS_ATTR])}): "
        f"{', '.join(sample_df.attrs[CALENDAR_COLUMNS_ATTR])}"
    )
    print(
        f"history_feature_columns ({len(sample_df.attrs[HISTORY_FEATURE_COLUMNS_ATTR])}): "
        f"{', '.join(sample_df.attrs[HISTORY_FEATURE_COLUMNS_ATTR])}"
    )

    _print_split_summary("train", train_df)
    _print_split_summary("val", val_df)
    _print_split_summary("test", test_df)


if __name__ == "__main__":
    main()
