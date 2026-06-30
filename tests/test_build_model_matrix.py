from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wave_experiments.features.build_model_matrix import (
    FEATURE_COLUMNS_OUTPUT_PATH,
    FORBIDDEN_FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MODEL_MATRIX_OUTPUT_PATH,
    REPORT_PATH,
    build_model_matrix,
    run_build_model_matrix,
)


def _row(
    sample_id: str,
    split: str,
    target_variable: str,
    horizon: int,
    value_baseline_name: str,
    hour_baseline_name: str,
) -> dict:
    row = {
        "样本ID": sample_id,
        "预测起点日期": f"2017-01-0{horizon}",
        "数据集划分": split,
        "目标变量": target_variable,
        "预测天数": horizon,
        "目标峰值": 10.0 + horizon,
        "target_peak_residual": 2.0 + horizon,
        "目标峰值小时": 5 + horizon,
        "baseline_peak": 8.0 + horizon,
        "peak_value_baseline_name": value_baseline_name,
        "baseline_peak_hour": 4 + horizon,
        "peak_hour_baseline_name": hour_baseline_name,
        "目标谷值": -1.0,
        "目标谷值残差": -2.0,
        "目标谷值小时": 3,
        "目标峰值残差": 99.0,
        "基线峰值": 77.0,
        "peak_hour_ordinary_error": 1,
        "peak_hour_circular_error": 1,
        "peak_hour_within_2h": True,
        "日历_星期": horizon,
        "日历_月份": 1,
        "日历_年内日序": horizon,
        "日历_是否周末": int(horizon >= 2),
    }
    variables = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
    for index, variable in enumerate(variables, start=1):
        base = float(index * 10 + horizon)
        row.update(
            {
                f"{variable}_过去96小时_均值": base,
                f"{variable}_过去96小时_标准差": base + 0.1,
                f"{variable}_过去96小时_最大值": base + 1.0,
                f"{variable}_过去96小时_最小值": base - 1.0,
                f"{variable}_过去96小时_首值": base - 0.5,
                f"{variable}_过去96小时_末首差": 0.5,
                f"{variable}_过去96小时_差分均值": 0.1,
                f"{variable}_过去96小时_差分标准差": 0.2,
                f"{variable}_过去96小时_差分最大值": 0.3,
                f"{variable}_过去96小时_差分最小值": -0.3,
                f"{variable}_过去96小时_趋势斜率": 0.01,
                f"{variable}_过去第1天_均值": base + 2.0,
                f"{variable}_过去第1天_标准差": base + 2.1,
                f"{variable}_过去第1天_最小值": base + 1.0,
                f"{variable}_过去第1天_最大值": base + 3.0,
                f"{variable}_过去第1天_末值": base + 2.5,
                f"{variable}_过去第1天_峰值小时": 6,
                f"{variable}_过去第1天_谷值小时": 2,
                f"{variable}_过去第1天_峰谷差": 4.0,
                f"{variable}_历史峰值_均值4天": base + 4.0,
                f"{variable}_历史峰值_标准差4天": 0.4,
                f"{variable}_历史峰值_最小值4天": base + 3.0,
                f"{variable}_历史峰值_最近差": 0.6,
                f"{variable}_历史峰值小时_均值4天": 7.0,
                f"{variable}_历史峰值小时_标准差4天": 1.1,
                f"{variable}_历史峰值小时_中位数4天": 7.0,
                f"{variable}_历史峰值小时_众数4天": 8.0,
                f"{variable}_峰谷差_均值4天": 5.0,
                f"{variable}_峰谷差_标准差4天": 0.7,
                f"{variable}_峰谷差_最大值4天": 6.0,
                f"{variable}_峰谷差_最近差": 0.8,
            }
        )
    return row


def _full_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("S1", "训练", "HUFL", 1, "mean_last_4", "mode_last_4"),
            _row("S2", "验证", "OT", 2, "cycle_mod_4", "median_last_4"),
            _row("S3", "测试", "OT", 3, "weekday_horizon_mean", "weekday_mode"),
        ]
    )


def _skinny_frame() -> pd.DataFrame:
    return _full_feature_frame().loc[
        :,
        [
            "样本ID",
            "预测起点日期",
            "数据集划分",
            "目标变量",
            "预测天数",
            "目标峰值",
            "baseline_peak",
            "target_peak_residual",
            "peak_value_baseline_name",
            "目标峰值小时",
            "baseline_peak_hour",
            "peak_hour_baseline_name",
            "peak_hour_ordinary_error",
            "peak_hour_circular_error",
            "peak_hour_within_2h",
        ],
    ].copy()


def test_build_model_matrix_includes_full_history_calendar_and_cross_variable_features() -> None:
    matrix, feature_columns, report = build_model_matrix(_full_feature_frame())

    assert len(feature_columns) > 100
    assert "预测天数" in feature_columns
    assert "baseline_peak" in feature_columns
    assert "baseline_peak_hour" in feature_columns
    assert "日历_星期" in feature_columns
    assert "日历_月份" in feature_columns
    assert any("过去96小时" in column for column in feature_columns)
    assert any("过去第1天" in column for column in feature_columns)
    assert any("历史峰值_" in column for column in feature_columns)
    assert any("历史峰值小时_" in column for column in feature_columns)
    assert any("峰谷差" in column for column in feature_columns)
    assert any(column.startswith("目标变量__") for column in feature_columns)
    assert "HUFL_过去96小时_均值" in feature_columns
    assert "OT_过去96小时_均值" in feature_columns
    assert "各类特征数量" in report
    assert "信息泄露检查" in report
    assert matrix.shape[0] == 3


def test_build_model_matrix_excludes_labels_future_information_and_old_baseline_fields() -> None:
    matrix, feature_columns, report = build_model_matrix(_full_feature_frame())

    for column in FORBIDDEN_FEATURE_COLUMNS:
        assert column not in feature_columns
    assert set(LABEL_COLUMNS).issubset(matrix.columns)
    assert "目标谷值" not in matrix.columns
    assert "基线峰值" not in matrix.columns
    assert "peak_hour_ordinary_error" not in matrix.columns
    assert "实际排除字段" in report
    assert "标签字段与特征列交集：无" in report


def test_build_model_matrix_rejects_skinny_input_without_history_or_calendar_features() -> None:
    with pytest.raises(ValueError, match="缺少完整历史/日历特征"):
        build_model_matrix(_skinny_frame())


def test_build_model_matrix_outputs_numeric_finite_features_with_consistent_split_columns() -> None:
    df = _full_feature_frame()
    df.loc[1, "HUFL_过去96小时_均值"] = np.nan
    df.loc[2, "OT_峰谷差_最大值4天"] = np.inf

    matrix, feature_columns, _ = build_model_matrix(df)
    numeric = matrix.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")

    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert any(column.startswith("peak_value_baseline_name__") for column in feature_columns)
    assert any(column.startswith("peak_hour_baseline_name__") for column in feature_columns)
    for _, split_frame in matrix.groupby("数据集划分", sort=False):
        assert list(split_frame.loc[:, feature_columns].columns) == feature_columns


def test_run_build_model_matrix_writes_matrix_feature_columns_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "dataset_with_residual_targets.csv"
    matrix_path = tmp_path / MODEL_MATRIX_OUTPUT_PATH.name
    feature_columns_path = tmp_path / FEATURE_COLUMNS_OUTPUT_PATH.name
    report_path = tmp_path / REPORT_PATH.name
    _full_feature_frame().to_csv(input_path, index=False)

    matrix, feature_columns, report = run_build_model_matrix(
        input_path=input_path,
        matrix_output_path=matrix_path,
        feature_columns_output_path=feature_columns_path,
        report_path=report_path,
    )

    assert matrix_path.exists()
    assert feature_columns_path.exists()
    assert report_path.exists()
    assert pd.read_csv(matrix_path).shape == matrix.shape
    assert feature_columns_path.read_text(encoding="utf-8").splitlines() == feature_columns
    assert "最终特征数" in report
    assert "禁止字段清单" in report
    assert "split 行数和特征一致性" in report
