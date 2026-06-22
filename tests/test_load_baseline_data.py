from pathlib import Path

import pandas as pd

from wave_experiments.baselines.load_baseline_data import (
    FEATURE_COLUMNS_ATTR,
    ID_COLUMNS_ATTR,
    LABEL_COLUMNS,
    LABEL_COLUMNS_ATTR,
    load_peak_dataset,
)


def test_load_peak_dataset_splits_rows_and_marks_label_feature_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "mini_long_table.csv"
    rows = [
        {
            "样本ID": "S1",
            "预测起点日期": "2016-01-01",
            "目标变量": "HUFL",
            "预测天数": 1,
            "基线峰值": 10.0,
            "数据集划分": "训练",
            "目标峰值": 11.0,
            "目标峰值残差": 1.0,
            "目标峰值小时": 8,
            "目标谷值": 5.0,
            "目标谷值残差": -1.0,
            "目标谷值小时": 4,
            "日历_星期": 5,
            "HUFL_过去96小时_均值": 9.0,
        },
        {
            "样本ID": "S2",
            "预测起点日期": "2016-01-02",
            "目标变量": "HULL",
            "预测天数": 2,
            "基线峰值": 20.0,
            "数据集划分": "验证",
            "目标峰值": 21.0,
            "目标峰值残差": 1.0,
            "目标峰值小时": 9,
            "目标谷值": 6.0,
            "目标谷值残差": -1.0,
            "目标谷值小时": 5,
            "日历_星期": 6,
            "HULL_历史峰值_均值4天": 19.0,
        },
        {
            "样本ID": "S3",
            "预测起点日期": "2016-01-03",
            "目标变量": "OT",
            "预测天数": 3,
            "基线峰值": 30.0,
            "数据集划分": "测试",
            "目标峰值": 31.0,
            "目标峰值残差": 1.0,
            "目标峰值小时": 10,
            "目标谷值": 7.0,
            "目标谷值残差": -1.0,
            "目标谷值小时": 6,
            "日历_星期": 0,
            "OT_峰谷差_最近差": 2.0,
        },
    ]
    pd.DataFrame(rows).to_csv(input_path, index=False)

    train_df, val_df, test_df = load_peak_dataset(input_path)

    assert train_df["数据集划分"].tolist() == ["训练"]
    assert val_df["数据集划分"].tolist() == ["验证"]
    assert test_df["数据集划分"].tolist() == ["测试"]

    for split_df in (train_df, val_df, test_df):
        assert set(split_df.attrs[LABEL_COLUMNS_ATTR]) == set(LABEL_COLUMNS)
        assert set(LABEL_COLUMNS).issubset(split_df.columns)
        assert set(LABEL_COLUMNS).isdisjoint(split_df.attrs[FEATURE_COLUMNS_ATTR])
        assert "样本ID" in split_df.attrs[ID_COLUMNS_ATTR]
        assert "样本ID" not in split_df.attrs[FEATURE_COLUMNS_ATTR]
        assert "数据集划分" not in split_df.attrs[FEATURE_COLUMNS_ATTR]
        assert "目标变量" not in split_df.attrs[FEATURE_COLUMNS_ATTR]
        assert "预测天数" not in split_df.attrs[FEATURE_COLUMNS_ATTR]
        assert "基线峰值" in split_df.attrs[FEATURE_COLUMNS_ATTR]
        assert "日历_星期" in split_df.attrs[FEATURE_COLUMNS_ATTR]
        assert any("过去" in column or "历史" in column or "峰谷差" in column for column in split_df.attrs[FEATURE_COLUMNS_ATTR])
