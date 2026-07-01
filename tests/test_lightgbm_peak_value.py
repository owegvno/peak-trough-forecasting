from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from wave_dataset import visualization as visualization_module
import wave_experiments.models.train_lightgbm_peak_value as lightgbm_module


def _prediction_row(
    sample_id: str,
    split: str,
    target_variable: str,
    horizon: int,
    pred_peak_value: float,
) -> dict[str, object]:
    return {
        "样本ID": sample_id,
        "预测起点日期": "2017-01-01",
        "数据集划分": split,
        "目标变量": target_variable,
        "预测天数": horizon,
        "目标峰值": 10.0 + horizon,
        "baseline_peak": 8.0 + horizon,
        "target_peak_residual": 2.0,
        "pred_peak_residual": pred_peak_value - (8.0 + horizon),
        "pred_peak_value": pred_peak_value,
        "model_name": "lightgbm_peak_value",
    }


def test_plot_lightgbm_peak_value_prediction_batch_uses_val_and_test_split_folders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            _prediction_row("V1", "验证", "HUFL", 1, 11.5),
            _prediction_row("V1", "验证", "HUFL", 2, 12.5),
            _prediction_row("T1", "测试", "HUFL", 1, 21.5),
            _prediction_row("T1", "测试", "HUFL", 2, 22.5),
        ]
    ).to_csv(prediction_path, index=False)

    captured: list[dict[str, object]] = []

    monkeypatch.setattr(visualization_module, "parse_hourly_csv", lambda _: [{"date": object()}])
    monkeypatch.setattr(
        visualization_module,
        "_true_daily_peak_rows",
        lambda records, forecast_start, target_col, pred_days: [
            {"预测天数": 1, "目标峰值小时": 3, "目标峰值": 11.0},
            {"预测天数": 2, "目标峰值小时": 4, "目标峰值": 12.0},
        ],
    )

    def fake_plot_peak_prediction_rows(**kwargs: object) -> Path:
        captured.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        path = output_dir / f"{kwargs['target_col']}_{len(captured)}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("fake image", encoding="utf-8")
        return path

    monkeypatch.setattr(
        visualization_module,
        "plot_peak_prediction_rows",
        fake_plot_peak_prediction_rows,
    )

    paths = visualization_module.plot_lightgbm_peak_value_prediction_batch(
        hourly_csv="ETTh1.csv",
        prediction_csv=prediction_path,
        output_root=tmp_path,
        dataset_name="ETTH1_pred14_seq4",
        target_cols=["HUFL"],
        splits=("验证", "测试"),
        sample_count=1,
    )

    assert sorted(paths) == ["测试", "验证"]
    assert [call["task_name"] for call in captured] == [
        "LightGBM_波峰残差预测_验证",
        "LightGBM_波峰残差预测_测试",
    ]
    assert captured[0]["output_dir"] == tmp_path / "ETTH1_pred14_seq4" / "LightGBM_波峰残差预测_验证" / "HUFL"
    assert captured[1]["output_dir"] == tmp_path / "ETTH1_pred14_seq4" / "LightGBM_波峰残差预测_测试" / "HUFL"
    assert captured[0]["prediction_label"] == "LightGBM波峰残差预测"
    assert captured[0]["filename_suffix"] == "LightGBM波峰残差预测"
    first_rows = captured[0]["prediction_rows"]
    assert isinstance(first_rows, list)
    assert first_rows[0]["baseline_peak_value"] == pytest.approx(11.5)
    assert first_rows[0]["baseline_peak_hour"] == 3


def test_run_lightgbm_peak_value_training_triggers_visualization_after_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace())
    monkeypatch.setattr(
        lightgbm_module,
        "load_model_matrix",
        lambda matrix_path, feature_columns_path: (
            pd.DataFrame(
                {
                    lightgbm_module.TARGET_VARIABLE_COLUMN: list(
                        lightgbm_module.EXPECTED_TARGET_VARIABLES
                    )
                }
            ),
            ["feature_a"],
        ),
    )

    def fake_train_one_target(
        lgb: object,
        target_variable: str,
        matrix: pd.DataFrame,
        feature_columns: list[str],
        model_dir: Path,
        num_boost_round: int,
        early_stopping_rounds: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
        rows = []
        for split in lightgbm_module.EVAL_SPLITS:
            frame = pd.DataFrame(
                [
                    {
                        lightgbm_module.SAMPLE_ID_COLUMN: f"{split}_{target_variable}",
                        lightgbm_module.START_DATE_COLUMN: "2017-01-01",
                        lightgbm_module.SPLIT_COLUMN: split,
                        lightgbm_module.TARGET_VARIABLE_COLUMN: target_variable,
                        lightgbm_module.HORIZON_COLUMN: 1,
                        lightgbm_module.TARGET_PEAK_COLUMN: 10.0,
                        lightgbm_module.BASELINE_PEAK_COLUMN: 8.0,
                        lightgbm_module.TARGET_RESIDUAL_COLUMN: 2.0,
                    }
                ]
            )
            rows.append(lightgbm_module.add_peak_value_predictions(frame, [1.5]))
        importance = pd.DataFrame(
            {
                "target_variable": [target_variable],
                "feature": ["feature_a"],
                "importance_gain": [1.0],
                "importance_split": [1],
                "best_iteration": [1],
            }
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{target_variable}_peak_value.txt"
        model_path.write_text("fake model", encoding="utf-8")
        summary = {
            "target_variable": target_variable,
            "model_path": str(model_path),
            "train_rows": 1,
            "val_rows": 1,
            "test_rows": 1,
            "feature_count": 1,
            "best_iteration": 1,
            "best_validation_l1": 0.5,
        }
        return pd.concat(rows, ignore_index=True), importance, summary, model_path

    def fake_comparison(metrics: pd.DataFrame, best_rule: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "eval_level": "global",
                    lightgbm_module.SPLIT_COLUMN: split,
                    "model_name": lightgbm_module.MODEL_NAME,
                    "baseline_name": lightgbm_module.BASELINE_NAME,
                    "best_rule_baseline_name": pd.NA,
                    lightgbm_module.TARGET_VARIABLE_COLUMN: pd.NA,
                    lightgbm_module.HORIZON_COLUMN: pd.NA,
                    "row_count": 7,
                    "MAE": 0.5,
                    "baseline_MAE": 1.0,
                    "MAE_improvement": 0.5,
                    "MAE_exceeds_best_rule": True,
                    "RMSE": 0.5,
                    "baseline_RMSE": 1.0,
                    "RMSE_improvement": 0.5,
                    "RMSE_exceeds_best_rule": True,
                    "sMAPE": 0.1,
                    "baseline_sMAPE": 0.2,
                    "sMAPE_improvement": 0.5,
                    "sMAPE_exceeds_best_rule": True,
                }
                for split in lightgbm_module.EVAL_SPLITS
            ]
        )

    plot_calls: list[dict[str, object]] = []

    def fake_plot_lightgbm_peak_value_predictions(**kwargs: object) -> tuple[Path, ...]:
        plot_calls.append(kwargs)
        return (tmp_path / "plot.png",)

    monkeypatch.setattr(lightgbm_module, "_train_one_target", fake_train_one_target)
    monkeypatch.setattr(lightgbm_module, "_load_best_rule_baseline", lambda *args: pd.DataFrame())
    monkeypatch.setattr(lightgbm_module, "compare_with_best_rule_baseline", fake_comparison)
    monkeypatch.setattr(
        lightgbm_module,
        "plot_lightgbm_peak_value_predictions",
        fake_plot_lightgbm_peak_value_predictions,
    )

    outputs = lightgbm_module.run_lightgbm_peak_value_training(
        matrix_path=tmp_path / "matrix.csv",
        feature_columns_path=tmp_path / "features.txt",
        best_rule_baseline_path=tmp_path / "best.csv",
        output_dir=tmp_path,
        model_dir=tmp_path / "models",
        num_boost_round=1,
        early_stopping_rounds=1,
    )

    assert len(plot_calls) == 1
    assert plot_calls[0]["prediction_csv"] == outputs.prediction_path
    assert plot_calls[0]["splits"] == lightgbm_module.EVAL_SPLITS
    assert outputs.plot_paths == (tmp_path / "plot.png",)
