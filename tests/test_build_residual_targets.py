from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wave_experiments.features.build_residual_targets import (
    TARGET_PEAK_RESIDUAL_COLUMN,
    build_residual_target_dataset,
    residual_distribution,
    run_build_residual_targets,
)


def _baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "样本ID": "S1",
                "预测起点日期": "2017-01-01",
                "数据集划分": "验证",
                "目标变量": "HUFL",
                "预测天数": 1,
                "基线峰值": 0.0,
                "目标峰值": 10.0,
                "目标峰值残差": -99.0,
                "baseline_peak": 8.5,
                TARGET_PEAK_RESIDUAL_COLUMN: 999.0,
                "peak_value_baseline_name": "old",
                "目标峰值小时": 5,
                "目标谷值": -2.0,
                "目标谷值残差": -3.0,
                "目标谷值小时": 2,
                "baseline_peak_hour": 4,
                "peak_hour_baseline_name": "mode_last_4",
                "peak_hour_ordinary_error": 1,
                "peak_hour_circular_error": 1,
                "peak_hour_within_2h": True,
                "日历_星期": 1,
                "日历_月份": 1,
                "HUFL_过去96小时_均值": 1.0,
                "HUFL_过去96小时_标准差": 0.5,
                "HUFL_过去第1天_均值": 1.1,
                "HUFL_过去第1天_最大值": 2.0,
                "HUFL_过去第4天_峰谷差": 1.5,
                "HUFL_历史峰值_均值4天": 2.5,
                "HUFL_历史峰值小时_众数4天": 5,
                "HUFL_峰谷差_均值4天": 1.8,
                "HULL_过去96小时_均值": 9.0,
            },
            {
                "样本ID": "S2",
                "预测起点日期": "2017-01-02",
                "数据集划分": "测试",
                "目标变量": "HUFL",
                "预测天数": 1,
                "基线峰值": 0.0,
                "目标峰值": 12.0,
                "目标峰值残差": -99.0,
                "baseline_peak": 11.0,
                TARGET_PEAK_RESIDUAL_COLUMN: 999.0,
                "peak_value_baseline_name": "old",
                "目标峰值小时": 6,
                "目标谷值": -2.0,
                "目标谷值残差": -3.0,
                "目标谷值小时": 2,
                "baseline_peak_hour": 8,
                "peak_hour_baseline_name": "mode_last_4",
                "peak_hour_ordinary_error": 2,
                "peak_hour_circular_error": 2,
                "peak_hour_within_2h": True,
                "日历_星期": 2,
                "日历_月份": 1,
                "HUFL_过去96小时_均值": 1.2,
                "HUFL_过去96小时_标准差": 0.6,
                "HUFL_过去第1天_均值": 1.3,
                "HUFL_过去第1天_最大值": 2.2,
                "HUFL_过去第4天_峰谷差": 1.6,
                "HUFL_历史峰值_均值4天": 2.7,
                "HUFL_历史峰值小时_众数4天": 6,
                "HUFL_峰谷差_均值4天": 1.9,
                "HULL_过去96小时_均值": 9.1,
            },
            {
                "样本ID": "S3",
                "预测起点日期": "2017-01-03",
                "数据集划分": "测试",
                "目标变量": "OT",
                "预测天数": 2,
                "基线峰值": 0.0,
                "目标峰值": 5.0,
                "目标峰值残差": -99.0,
                "baseline_peak": 7.0,
                TARGET_PEAK_RESIDUAL_COLUMN: 999.0,
                "peak_value_baseline_name": "old",
                "目标峰值小时": 11,
                "目标谷值": -2.0,
                "目标谷值残差": -3.0,
                "目标谷值小时": 2,
                "baseline_peak_hour": 12,
                "peak_hour_baseline_name": "global_mode",
                "peak_hour_ordinary_error": 1,
                "peak_hour_circular_error": 1,
                "peak_hour_within_2h": True,
                "日历_星期": 3,
                "日历_月份": 1,
                "OT_过去96小时_均值": 3.0,
                "OT_过去96小时_标准差": 0.7,
                "OT_过去第1天_均值": 3.1,
                "OT_过去第1天_最大值": 4.0,
                "OT_过去第4天_峰谷差": 1.7,
                "OT_历史峰值_均值4天": 4.5,
                "OT_历史峰值小时_众数4天": 11,
                "OT_峰谷差_均值4天": 2.1,
                "HULL_过去96小时_均值": 9.2,
            },
        ]
    )


def _slim_baseline_frame() -> pd.DataFrame:
    return _baseline_frame().loc[
        :,
        [
            "样本ID",
            "预测起点日期",
            "数据集划分",
            "目标变量",
            "预测天数",
            "目标峰值",
            "baseline_peak",
            TARGET_PEAK_RESIDUAL_COLUMN,
            "peak_value_baseline_name",
            "目标峰值小时",
            "baseline_peak_hour",
            "peak_hour_baseline_name",
            "peak_hour_ordinary_error",
            "peak_hour_circular_error",
            "peak_hour_within_2h",
        ],
    ]


def test_build_residual_target_dataset_recomputes_residual_and_keeps_peak_value_features() -> None:
    df = _baseline_frame()

    result = build_residual_target_dataset(df)

    assert result[TARGET_PEAK_RESIDUAL_COLUMN].tolist() == pytest.approx([1.5, 1.0, -2.0])
    restored = result["baseline_peak"] + result[TARGET_PEAK_RESIDUAL_COLUMN]
    assert restored.tolist() == pytest.approx(result["目标峰值"].tolist())
    assert result[TARGET_PEAK_RESIDUAL_COLUMN].isna().sum() == 0
    assert len(result.columns) > 15
    assert {
        "样本ID",
        "预测起点日期",
        "数据集划分",
        "目标变量",
        "预测天数",
        "目标峰值",
        TARGET_PEAK_RESIDUAL_COLUMN,
        "目标峰值小时",
        "baseline_peak",
        "peak_value_baseline_name",
        "baseline_peak_hour",
        "peak_hour_baseline_name",
        "日历_星期",
        "日历_月份",
        "HUFL_过去96小时_均值",
        "HUFL_过去第1天_最大值",
        "HUFL_历史峰值_均值4天",
        "HUFL_历史峰值小时_众数4天",
        "HUFL_峰谷差_均值4天",
        "HULL_过去96小时_均值",
    }.issubset(result.columns)


def test_build_residual_target_dataset_removes_unrelated_future_and_old_fields() -> None:
    result = build_residual_target_dataset(_baseline_frame())

    assert {
        "目标谷值",
        "目标谷值残差",
        "目标谷值小时",
        "目标峰值残差",
        "基线峰值",
        "peak_hour_ordinary_error",
        "peak_hour_circular_error",
        "peak_hour_within_2h",
    }.isdisjoint(result.columns)


def test_residual_distribution_groups_by_target_variable_and_horizon() -> None:
    dataset = build_residual_target_dataset(_baseline_frame())

    stats = residual_distribution(dataset)

    first = stats.loc[
        (stats["目标变量"] == "HUFL") & (stats["预测天数"] == 1)
    ].iloc[0]
    assert first["row_count"] == 2
    assert first["mean"] == pytest.approx(1.25)
    assert first["std"] == pytest.approx(0.3535533905932738)
    assert first["min"] == pytest.approx(1.0)
    assert first["max"] == pytest.approx(1.5)


def test_run_build_residual_targets_writes_dataset_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "dataset_with_selected_baselines.csv"
    output_path = tmp_path / "dataset_with_residual_targets.csv"
    report_path = tmp_path / "残差标签报告.md"
    _baseline_frame().to_csv(input_path, index=False)

    dataset, report = run_build_residual_targets(input_path, output_path, report_path)

    written = pd.read_csv(output_path)
    assert TARGET_PEAK_RESIDUAL_COLUMN in written.columns
    assert written[TARGET_PEAK_RESIDUAL_COLUMN].isna().sum() == 0
    assert dataset[TARGET_PEAK_RESIDUAL_COLUMN].tolist() == pytest.approx(
        written[TARGET_PEAK_RESIDUAL_COLUMN].tolist()
    )
    assert "残差分布" in report
    assert "输入列数" in report
    assert "输出列数" in report
    assert "删除字段清单" in report
    assert "核心历史/日历特征检查结果" in report
    assert "| 目标变量 | 预测天数 | row_count | mean | std | min | max |" in report
    assert report_path.read_text(encoding="utf-8") == report


def test_build_residual_target_dataset_rejects_nan_residual_sources() -> None:
    df = _baseline_frame()
    df.loc[0, "baseline_peak"] = None

    with pytest.raises(ValueError, match="baseline_peak"):
        build_residual_target_dataset(df)


def test_build_residual_target_dataset_rejects_slim_selected_baseline_table() -> None:
    df = _slim_baseline_frame()

    with pytest.raises(ValueError, match="完整历史/日历特征|核心历史统计特征|瘦身"):
        build_residual_target_dataset(df)
