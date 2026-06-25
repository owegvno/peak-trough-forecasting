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
                "目标峰值": 10.0,
                "baseline_peak": 8.5,
                TARGET_PEAK_RESIDUAL_COLUMN: 999.0,
                "peak_value_baseline_name": "old",
                "目标峰值小时": 5,
                "baseline_peak_hour": 4,
                "peak_hour_baseline_name": "mode_last_4",
                "peak_hour_ordinary_error": 1,
                "peak_hour_circular_error": 1,
                "peak_hour_within_2h": True,
            },
            {
                "样本ID": "S2",
                "预测起点日期": "2017-01-02",
                "数据集划分": "测试",
                "目标变量": "HUFL",
                "预测天数": 1,
                "目标峰值": 12.0,
                "baseline_peak": 11.0,
                TARGET_PEAK_RESIDUAL_COLUMN: 999.0,
                "peak_value_baseline_name": "old",
                "目标峰值小时": 6,
                "baseline_peak_hour": 8,
                "peak_hour_baseline_name": "mode_last_4",
                "peak_hour_ordinary_error": 2,
                "peak_hour_circular_error": 2,
                "peak_hour_within_2h": True,
            },
            {
                "样本ID": "S3",
                "预测起点日期": "2017-01-03",
                "数据集划分": "测试",
                "目标变量": "OT",
                "预测天数": 2,
                "目标峰值": 5.0,
                "baseline_peak": 7.0,
                TARGET_PEAK_RESIDUAL_COLUMN: 999.0,
                "peak_value_baseline_name": "old",
                "目标峰值小时": 11,
                "baseline_peak_hour": 12,
                "peak_hour_baseline_name": "global_mode",
                "peak_hour_ordinary_error": 1,
                "peak_hour_circular_error": 1,
                "peak_hour_within_2h": True,
            },
        ]
    )


def test_build_residual_target_dataset_recomputes_residual_and_preserves_training_fields() -> None:
    df = _baseline_frame()

    result = build_residual_target_dataset(df)

    assert list(result.columns) == list(df.columns)
    assert result[TARGET_PEAK_RESIDUAL_COLUMN].tolist() == pytest.approx([1.5, 1.0, -2.0])
    restored = result["baseline_peak"] + result[TARGET_PEAK_RESIDUAL_COLUMN]
    assert restored.tolist() == pytest.approx(result["目标峰值"].tolist())
    assert result[TARGET_PEAK_RESIDUAL_COLUMN].isna().sum() == 0


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
    assert "| 目标变量 | 预测天数 | row_count | mean | std | min | max |" in report
    assert report_path.read_text(encoding="utf-8") == report


def test_build_residual_target_dataset_rejects_nan_residual_sources() -> None:
    df = _baseline_frame()
    df.loc[0, "baseline_peak"] = None

    with pytest.raises(ValueError, match="baseline_peak"):
        build_residual_target_dataset(df)
