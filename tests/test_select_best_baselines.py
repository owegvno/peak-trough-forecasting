from __future__ import annotations

import pandas as pd
import pytest

from wave_experiments.baselines.load_baseline_data import (
    HORIZON_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    TEST_SPLIT,
    VAL_SPLIT,
)
from wave_experiments.baselines.select_best_baselines import (
    build_dataset_with_selected_baselines,
    select_best_peak_hour_baselines,
    select_best_peak_value_baselines,
)


def test_select_best_peak_value_baselines_uses_validation_mae_by_target_and_horizon() -> None:
    metrics = pd.DataFrame(
        [
            {
                "eval_level": "target_variable_horizon",
                SPLIT_COLUMN: VAL_SPLIT,
                "baseline_name": "mean_last_4",
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "row_count": 3,
                "MAE": 2.0,
                "RMSE": 2.5,
                "sMAPE": 0.2,
            },
            {
                "eval_level": "target_variable_horizon",
                SPLIT_COLUMN: VAL_SPLIT,
                "baseline_name": "cycle_mod_4",
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "row_count": 3,
                "MAE": 1.0,
                "RMSE": 3.0,
                "sMAPE": 0.3,
            },
            {
                "eval_level": "target_variable_horizon",
                SPLIT_COLUMN: TEST_SPLIT,
                "baseline_name": "mean_last_4",
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "row_count": 3,
                "MAE": 0.1,
                "RMSE": 0.2,
                "sMAPE": 0.1,
            },
        ]
    )

    best = select_best_peak_value_baselines(metrics)

    assert len(best) == 1
    row = best.iloc[0]
    assert row[TARGET_VARIABLE_COLUMN] == "HUFL"
    assert row[HORIZON_COLUMN] == 1
    assert row["best_baseline_name"] == "cycle_mod_4"
    assert row["selection_metric"] == "MAE"
    assert row["validation_MAE"] == pytest.approx(1.0)


def test_select_best_peak_hour_baselines_uses_hit_rate_then_hour_error() -> None:
    metrics = pd.DataFrame(
        [
            {
                "eval_level": "target_variable_horizon",
                SPLIT_COLUMN: VAL_SPLIT,
                "baseline_name": "mode_last_4",
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "row_count": 3,
                "普通小时误差": 3.0,
                "环形小时误差": 2.0,
                "Top-1 accuracy": 0.1,
                "±1h 命中率": 0.5,
                "±2h 命中率": 0.8,
            },
            {
                "eval_level": "target_variable_horizon",
                SPLIT_COLUMN: VAL_SPLIT,
                "baseline_name": "median_last_4",
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "row_count": 3,
                "普通小时误差": 2.0,
                "环形小时误差": 1.5,
                "Top-1 accuracy": 0.2,
                "±1h 命中率": 0.4,
                "±2h 命中率": 0.8,
            },
            {
                "eval_level": "target_variable_horizon",
                SPLIT_COLUMN: VAL_SPLIT,
                "baseline_name": "global_mode",
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "row_count": 3,
                "普通小时误差": 1.0,
                "环形小时误差": 0.8,
                "Top-1 accuracy": 0.3,
                "±1h 命中率": 0.7,
                "±2h 命中率": 0.7,
            },
        ]
    )

    best = select_best_peak_hour_baselines(metrics)

    assert len(best) == 1
    row = best.iloc[0]
    assert row["best_baseline_name"] == "median_last_4"
    assert row["selection_metric"] == "±2h 命中率, then 普通小时误差"
    assert row["validation_±2h 命中率"] == pytest.approx(0.8)
    assert row["validation_普通小时误差"] == pytest.approx(2.0)


def test_build_dataset_with_selected_baselines_outputs_one_row_per_sample_with_baseline_peak() -> None:
    value_predictions = pd.DataFrame(
        [
            {
                "样本ID": "S1",
                "预测起点日期": "2017-01-01",
                SPLIT_COLUMN: VAL_SPLIT,
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "目标峰值": 10.0,
                "baseline_name": "mean_last_4",
                "baseline_peak_value": 8.0,
            },
            {
                "样本ID": "S1",
                "预测起点日期": "2017-01-01",
                SPLIT_COLUMN: VAL_SPLIT,
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "目标峰值": 10.0,
                "baseline_name": "cycle_mod_4",
                "baseline_peak_value": 9.5,
            },
            {
                "样本ID": "S2",
                "预测起点日期": "2017-01-02",
                SPLIT_COLUMN: TEST_SPLIT,
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "目标峰值": 12.0,
                "baseline_name": "cycle_mod_4",
                "baseline_peak_value": 11.0,
            },
        ]
    )
    hour_predictions = pd.DataFrame(
        [
            {
                "样本ID": "S1",
                "预测起点日期": "2017-01-01",
                SPLIT_COLUMN: VAL_SPLIT,
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "目标峰值小时": 5,
                "baseline_name": "mode_last_4",
                "baseline_peak_hour": 4,
            },
            {
                "样本ID": "S2",
                "预测起点日期": "2017-01-02",
                SPLIT_COLUMN: TEST_SPLIT,
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "目标峰值小时": 6,
                "baseline_name": "mode_last_4",
                "baseline_peak_hour": 8,
            },
        ]
    )
    best_value = pd.DataFrame(
        [
            {
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "best_baseline_name": "cycle_mod_4",
                "selection_metric": "MAE",
                "selection_rule": "validation MAE min",
                "validation_row_count": 3,
                "validation_MAE": 1.0,
                "validation_RMSE": 2.0,
                "validation_sMAPE": 0.2,
            }
        ]
    )
    best_hour = pd.DataFrame(
        [
            {
                TARGET_VARIABLE_COLUMN: "HUFL",
                HORIZON_COLUMN: 1,
                "best_baseline_name": "mode_last_4",
                "selection_metric": "±2h 命中率, then 普通小时误差",
                "selection_rule": "validation ±2h hit max, ordinary hour error min",
                "validation_row_count": 3,
                "validation_±2h 命中率": 0.8,
                "validation_普通小时误差": 2.0,
            }
        ]
    )

    dataset = build_dataset_with_selected_baselines(
        value_predictions,
        hour_predictions,
        best_value,
        best_hour,
    )

    assert len(dataset) == 2
    assert dataset["baseline_peak"].isna().sum() == 0
    assert set(dataset["peak_value_baseline_name"]) == {"cycle_mod_4"}
    assert set(dataset["peak_hour_baseline_name"]) == {"mode_last_4"}
    first = dataset.loc[dataset["样本ID"] == "S1"].iloc[0]
    assert first["baseline_peak"] == pytest.approx(9.5)
    assert first["target_peak_residual"] == pytest.approx(0.5)
    second = dataset.loc[dataset["样本ID"] == "S2"].iloc[0]
    assert second["baseline_peak"] == pytest.approx(11.0)
