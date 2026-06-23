from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from wave_experiments.baselines.load_baseline_data import SPLIT_COLUMN, TEST_SPLIT, VAL_SPLIT
from wave_experiments.baselines.peak_value_baselines import (
    BASELINE_NAMES,
    build_peak_value_predictions,
    evaluate_peak_value_baselines,
    run_peak_value_baselines,
)
import wave_experiments.baselines.peak_value_baselines as peak_value_module


def _base_row(
    sample_id: str,
    split: str,
    target_variable: str,
    horizon: int,
    weekday: int,
    target_peak_value: float,
) -> dict:
    return {
        "样本ID": sample_id,
        "预测起点日期": "2016-01-01",
        "目标变量": target_variable,
        "预测天数": horizon,
        "基线峰值": 0.0,
        "数据集划分": split,
        "目标峰值": target_peak_value,
        "目标峰值残差": 0.0,
        "目标峰值小时": 8,
        "目标谷值": 1.0,
        "目标谷值残差": 0.0,
        "目标谷值小时": 4,
        "日历_星期": weekday,
        f"{target_variable}_过去第1天_最大值": 14.0,
        f"{target_variable}_过去第2天_最大值": 12.0,
        f"{target_variable}_过去第3天_最大值": 10.0,
        f"{target_variable}_过去第4天_最大值": 8.0,
        f"{target_variable}_历史峰值_均值4天": 11.0,
    }


def _split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = [
        _base_row("T1", "训练", "HUFL", 1, 2, 100.0),
        _base_row("T2", "训练", "HUFL", 1, 2, 102.0),
        _base_row("T3", "训练", "OT", 4, 1, 50.0),
        _base_row("T4", "训练", "OT", 4, 5, 70.0),
    ]
    val_rows = [
        _base_row("V1", VAL_SPLIT, "HUFL", 1, 2, 13.0),
    ]
    test_rows = [
        _base_row("E1", TEST_SPLIT, "OT", 4, 3, 9.0),
    ]
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def test_build_peak_value_predictions_generates_all_baselines_for_val_and_test() -> None:
    train_df, val_df, test_df = _split_frames()

    predictions = build_peak_value_predictions(train_df, val_df, test_df)

    assert set(predictions[SPLIT_COLUMN]) == {VAL_SPLIT, TEST_SPLIT}
    assert set(predictions["baseline_name"]) == set(BASELINE_NAMES)
    assert predictions["baseline_peak_value"].isna().sum() == 0
    assert {
        "样本ID",
        "目标变量",
        "预测天数",
        "目标峰值",
        "baseline_name",
        "baseline_peak_value",
    }.issubset(predictions.columns)

    by_baseline = (
        predictions.loc[predictions["样本ID"] == "V1"]
        .set_index("baseline_name")["baseline_peak_value"]
        .to_dict()
    )
    assert by_baseline["mean_last_4"] == pytest.approx(11.0)
    assert by_baseline["weighted_mean_last_4"] == pytest.approx(12.0)
    assert by_baseline["cycle_mod_4"] == pytest.approx(14.0)
    assert by_baseline["weekday_horizon_mean"] == pytest.approx(101.0)

    test_cycle_value = predictions.loc[
        (predictions["样本ID"] == "E1") & (predictions["baseline_name"] == "cycle_mod_4"),
        "baseline_peak_value",
    ].iloc[0]
    assert test_cycle_value == pytest.approx(8.0)

    fallback_value = predictions.loc[
        (predictions["样本ID"] == "E1") & (predictions["baseline_name"] == "weekday_horizon_mean"),
        "baseline_peak_value",
    ].iloc[0]
    assert fallback_value == pytest.approx(60.0)


def test_evaluate_peak_value_baselines_outputs_four_metric_levels() -> None:
    train_df, val_df, test_df = _split_frames()
    predictions = build_peak_value_predictions(train_df, val_df, test_df)

    metrics = evaluate_peak_value_baselines(predictions)

    assert {"MAE", "RMSE", "sMAPE"}.issubset(metrics.columns)
    assert set(metrics["eval_level"]) == {
        "global",
        "target_variable",
        "horizon",
        "target_variable_horizon",
    }
    assert set(metrics["baseline_name"]) == set(BASELINE_NAMES)
    assert set(metrics[SPLIT_COLUMN]) == {VAL_SPLIT, TEST_SPLIT}

    row = metrics.loc[
        (metrics[SPLIT_COLUMN] == VAL_SPLIT)
        & (metrics["baseline_name"] == "mean_last_4")
        & (metrics["eval_level"] == "global")
    ].iloc[0]
    assert row["MAE"] == pytest.approx(2.0)
    assert row["RMSE"] == pytest.approx(2.0)
    assert row["sMAPE"] == pytest.approx(2 * 2.0 / (13.0 + 11.0))


def test_evaluate_peak_value_baselines_handles_zero_smape_denominator_without_warning() -> None:
    predictions = pd.DataFrame(
        [
            {
                "样本ID": "Z1",
                "预测起点日期": "2016-01-01",
                SPLIT_COLUMN: VAL_SPLIT,
                "目标变量": "HUFL",
                "预测天数": 1,
                "目标峰值": 0.0,
                "baseline_name": "mean_last_4",
                "baseline_peak_value": 0.0,
            }
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = evaluate_peak_value_baselines(predictions)

    assert caught == []
    assert set(metrics["sMAPE"]) == {0.0}


def test_run_peak_value_baselines_writes_prediction_and_metric_csvs(tmp_path: Path) -> None:
    train_df, val_df, test_df = _split_frames()
    input_path = tmp_path / "mini_long_table.csv"
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(input_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"

    run_peak_value_baselines(input_path, prediction_path, metrics_path, plot_predictions=False)

    written_predictions = pd.read_csv(prediction_path)
    written_metrics = pd.read_csv(metrics_path)
    assert set(written_predictions[SPLIT_COLUMN]) == {VAL_SPLIT, TEST_SPLIT}
    assert written_predictions["baseline_peak_value"].isna().sum() == 0
    assert {"MAE", "RMSE", "sMAPE"}.issubset(written_metrics.columns)


def test_run_peak_value_baselines_triggers_value_visualization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_df, val_df, test_df = _split_frames()
    input_path = tmp_path / "mini_long_table.csv"
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(input_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    calls: list[dict[str, object]] = []

    def fake_plot(**kwargs: object) -> dict:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(peak_value_module, "maybe_plot_peak_baseline_predictions", fake_plot)

    run_peak_value_baselines(
        input_path,
        prediction_path,
        metrics_path,
    )

    assert calls == [
        {
            "value_prediction_csv": prediction_path,
            "plot_hour": False,
        }
    ]
