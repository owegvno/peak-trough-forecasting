from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wave_experiments.baselines.load_baseline_data import SPLIT_COLUMN, TEST_SPLIT, VAL_SPLIT
from wave_experiments.baselines.peak_hour_baselines import (
    BASELINE_NAMES,
    build_peak_hour_predictions,
    evaluate_peak_hour_baselines,
    run_peak_hour_baselines,
)
import wave_experiments.baselines.peak_hour_baselines as peak_hour_module


def _base_row(
    sample_id: str,
    split: str,
    target_variable: str,
    horizon: int,
    weekday: int,
    target_peak_hour: int,
    history_hours: tuple[int, int, int, int] = (8, 8, 5, 5),
) -> dict:
    row = {
        "样本ID": sample_id,
        "预测起点日期": "2016-01-01",
        "目标变量": target_variable,
        "预测天数": horizon,
        "基线峰值": 0.0,
        "数据集划分": split,
        "目标峰值": 10.0,
        "目标峰值残差": 0.0,
        "目标峰值小时": target_peak_hour,
        "目标谷值": 1.0,
        "目标谷值残差": 0.0,
        "目标谷值小时": 4,
        "日历_星期": weekday,
    }
    for day, hour in enumerate(history_hours, start=1):
        row[f"{target_variable}_过去第{day}天_峰值小时"] = hour
    return row


def _split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = [
        _base_row("T1", "训练", "HUFL", 1, 2, 5),
        _base_row("T2", "训练", "HUFL", 1, 2, 7),
        _base_row("T3", "训练", "HUFL", 1, 3, 7),
        _base_row("T4", "训练", "OT", 4, 1, 23),
        _base_row("T5", "训练", "OT", 4, 5, 22),
    ]
    val_rows = [
        _base_row("V1", VAL_SPLIT, "HUFL", 1, 2, 7, (8, 8, 5, 5)),
    ]
    test_rows = [
        _base_row("E1", TEST_SPLIT, "OT", 4, 3, 0, (23, 0, 0, 23)),
    ]
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def test_build_peak_hour_predictions_generates_all_baselines_for_val_and_test() -> None:
    train_df, val_df, test_df = _split_frames()

    predictions = build_peak_hour_predictions(train_df, val_df, test_df)

    assert set(predictions[SPLIT_COLUMN]) == {VAL_SPLIT, TEST_SPLIT}
    assert set(predictions["baseline_name"]) == set(BASELINE_NAMES)
    assert predictions["baseline_peak_hour"].between(0, 23).all()
    assert predictions["baseline_peak_hour"].isna().sum() == 0
    assert {
        "样本ID",
        "目标变量",
        "预测天数",
        "目标峰值小时",
        "baseline_name",
        "baseline_peak_hour",
    }.issubset(predictions.columns)

    val_by_baseline = (
        predictions.loc[predictions["样本ID"] == "V1"]
        .set_index("baseline_name")["baseline_peak_hour"]
        .to_dict()
    )
    assert val_by_baseline["mode_last_4"] == 5
    assert val_by_baseline["median_last_4"] == 7
    assert val_by_baseline["global_mode"] == 7
    assert val_by_baseline["weekday_mode"] == 5

    test_by_baseline = (
        predictions.loc[predictions["样本ID"] == "E1"]
        .set_index("baseline_name")["baseline_peak_hour"]
        .to_dict()
    )
    assert test_by_baseline["mode_last_4"] == 0
    assert test_by_baseline["median_last_4"] == 12
    assert test_by_baseline["global_mode"] == 22
    assert test_by_baseline["weekday_mode"] == 22


def test_evaluate_peak_hour_baselines_outputs_four_metric_levels() -> None:
    predictions = pd.DataFrame(
        [
            {
                "样本ID": "A",
                "预测起点日期": "2016-01-01",
                SPLIT_COLUMN: VAL_SPLIT,
                "目标变量": "HUFL",
                "预测天数": 1,
                "目标峰值小时": 0,
                "baseline_name": "mode_last_4",
                "baseline_peak_hour": 23,
            },
            {
                "样本ID": "B",
                "预测起点日期": "2016-01-02",
                SPLIT_COLUMN: VAL_SPLIT,
                "目标变量": "HUFL",
                "预测天数": 1,
                "目标峰值小时": 3,
                "baseline_name": "mode_last_4",
                "baseline_peak_hour": 5,
            },
        ]
    )

    metrics = evaluate_peak_hour_baselines(predictions)

    assert {
        "普通小时误差",
        "环形小时误差",
        "Top-1 accuracy",
        "±1h 命中率",
        "±2h 命中率",
    }.issubset(metrics.columns)
    assert set(metrics["eval_level"]) == {
        "global",
        "target_variable",
        "horizon",
        "target_variable_horizon",
    }

    row = metrics.loc[
        (metrics[SPLIT_COLUMN] == VAL_SPLIT)
        & (metrics["baseline_name"] == "mode_last_4")
        & (metrics["eval_level"] == "global")
    ].iloc[0]
    assert row["普通小时误差"] == pytest.approx(12.5)
    assert row["环形小时误差"] == pytest.approx(1.5)
    assert row["Top-1 accuracy"] == pytest.approx(0.0)
    assert row["±1h 命中率"] == pytest.approx(0.5)
    assert row["±2h 命中率"] == pytest.approx(1.0)


def test_run_peak_hour_baselines_writes_prediction_and_metric_csvs(tmp_path: Path) -> None:
    train_df, val_df, test_df = _split_frames()
    input_path = tmp_path / "mini_long_table.csv"
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(input_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"

    run_peak_hour_baselines(input_path, prediction_path, metrics_path, plot_predictions=False)

    written_predictions = pd.read_csv(prediction_path)
    written_metrics = pd.read_csv(metrics_path)
    assert set(written_predictions[SPLIT_COLUMN]) == {VAL_SPLIT, TEST_SPLIT}
    assert set(written_predictions["baseline_name"]) == set(BASELINE_NAMES)
    assert written_predictions["baseline_peak_hour"].between(0, 23).all()
    assert {"普通小时误差", "环形小时误差", "Top-1 accuracy", "±1h 命中率", "±2h 命中率"}.issubset(
        written_metrics.columns
    )


def test_run_peak_hour_baselines_triggers_hour_visualization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_df, val_df, test_df = _split_frames()
    input_path = tmp_path / "mini_long_table.csv"
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(input_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    calls: list[dict[str, object]] = []

    def fake_plot(**kwargs: object) -> dict:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(peak_hour_module, "maybe_plot_peak_baseline_predictions", fake_plot)

    run_peak_hour_baselines(
        input_path,
        prediction_path,
        metrics_path,
    )

    assert calls == [
        {
            "hour_prediction_csv": prediction_path,
            "plot_value": False,
        }
    ]
