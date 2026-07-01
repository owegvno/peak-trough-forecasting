from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wave_dataset import visualization as visualization_module
import wave_experiments.models.train_lightgbm_peak_value as lightgbm_module
import wave_experiments.models.train_xgboost_peak_value as xgboost_module


class FakeBooster:
    def __init__(self) -> None:
        self.best_iteration = 3
        self.best_score = 0.25

    def predict(self, matrix: object, iteration_range: tuple[int, int] | None = None) -> np.ndarray:
        data = matrix.data
        return np.full(len(data), 1.5, dtype=float)

    def get_score(self, importance_type: str) -> dict[str, float]:
        values = {
            "gain": {"f0": 2.0, "f1": 1.0},
            "weight": {"f0": 4.0, "f1": 2.0},
        }
        return values[importance_type]

    def save_model(self, path: str) -> None:
        Path(path).write_text("fake xgboost model", encoding="utf-8")


class FakeDMatrix:
    def __init__(
        self,
        data: pd.DataFrame,
        label: pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.data = data
        self.label = label
        self.feature_names = feature_names


class FakeXGBoost:
    DMatrix = FakeDMatrix

    @staticmethod
    def train(**kwargs: object) -> FakeBooster:
        return FakeBooster()


def _minimal_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_variable in xgboost_module.EXPECTED_TARGET_VARIABLES:
        for split in (xgboost_module.TRAIN_SPLIT, *xgboost_module.EVAL_SPLITS):
            rows.append(
                {
                    xgboost_module.SAMPLE_ID_COLUMN: f"{split}_{target_variable}",
                    xgboost_module.START_DATE_COLUMN: "2017-01-01",
                    xgboost_module.SPLIT_COLUMN: split,
                    xgboost_module.TARGET_VARIABLE_COLUMN: target_variable,
                    xgboost_module.HORIZON_COLUMN: 1,
                    xgboost_module.TARGET_PEAK_COLUMN: 10.0,
                    xgboost_module.BASELINE_PEAK_COLUMN: 8.0,
                    xgboost_module.TARGET_RESIDUAL_COLUMN: 2.0,
                    "feature_a": 1.0,
                    "feature_b": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_xgboost_prediction_contract_matches_lightgbm_peak_value() -> None:
    frame = _minimal_matrix().head(1)

    predictions = xgboost_module.add_peak_value_predictions(frame, [1.5])

    assert list(predictions.columns) == list(lightgbm_module.PREDICTION_COLUMNS)
    assert predictions["model_name"].tolist() == ["xgboost_peak_value"]
    assert predictions["pred_peak_value"].tolist() == pytest.approx([9.5])


def test_xgboost_params_use_cuda_hist_when_gpu_runtime_is_installed() -> None:
    params = xgboost_module._xgboost_params()

    assert params["tree_method"] == "hist"
    assert params["device"] == "cuda"


def test_plot_xgboost_peak_value_prediction_batch_uses_val_and_test_split_folders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                xgboost_module.SAMPLE_ID_COLUMN: "V1",
                xgboost_module.START_DATE_COLUMN: "2017-01-01",
                xgboost_module.SPLIT_COLUMN: "验证",
                xgboost_module.TARGET_VARIABLE_COLUMN: "HUFL",
                xgboost_module.HORIZON_COLUMN: 1,
                xgboost_module.TARGET_PEAK_COLUMN: 11.0,
                xgboost_module.BASELINE_PEAK_COLUMN: 9.0,
                xgboost_module.TARGET_RESIDUAL_COLUMN: 2.0,
                "pred_peak_residual": 2.5,
                "pred_peak_value": 11.5,
                "model_name": "xgboost_peak_value",
            },
            {
                xgboost_module.SAMPLE_ID_COLUMN: "T1",
                xgboost_module.START_DATE_COLUMN: "2017-01-01",
                xgboost_module.SPLIT_COLUMN: "测试",
                xgboost_module.TARGET_VARIABLE_COLUMN: "HUFL",
                xgboost_module.HORIZON_COLUMN: 1,
                xgboost_module.TARGET_PEAK_COLUMN: 21.0,
                xgboost_module.BASELINE_PEAK_COLUMN: 19.0,
                xgboost_module.TARGET_RESIDUAL_COLUMN: 2.0,
                "pred_peak_residual": 2.5,
                "pred_peak_value": 21.5,
                "model_name": "xgboost_peak_value",
            },
        ]
    ).to_csv(prediction_path, index=False)

    captured: list[dict[str, object]] = []

    monkeypatch.setattr(visualization_module, "parse_hourly_csv", lambda _: [{"date": object()}])
    monkeypatch.setattr(
        visualization_module,
        "_true_daily_peak_rows",
        lambda records, forecast_start, target_col, pred_days: [
            {"预测天数": 1, "目标峰值小时": 3, "目标峰值": 11.0},
        ],
    )

    def fake_plot_peak_prediction_rows(**kwargs: object) -> Path:
        captured.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        path = output_dir / f"{kwargs['target_col']}_{len(captured)}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("fake image", encoding="utf-8")
        return path

    monkeypatch.setattr(visualization_module, "plot_peak_prediction_rows", fake_plot_peak_prediction_rows)

    paths = visualization_module.plot_lightgbm_peak_value_prediction_batch(
        hourly_csv="ETTh1.csv",
        prediction_csv=prediction_path,
        output_root=tmp_path,
        dataset_name="ETTH1_pred14_seq4",
        target_cols=["HUFL"],
        splits=("验证", "测试"),
        sample_count=1,
        plot_group_prefix="XGBoost_波峰残差预测",
        prediction_label="XGBoost波峰残差预测",
        filename_suffix="XGBoost波峰残差预测",
    )

    assert sorted(paths) == ["测试", "验证"]
    assert [call["task_name"] for call in captured] == [
        "XGBoost_波峰残差预测_验证",
        "XGBoost_波峰残差预测_测试",
    ]
    assert captured[0]["output_dir"] == tmp_path / "ETTH1_pred14_seq4" / "XGBoost_波峰残差预测_验证" / "HUFL"
    assert captured[1]["output_dir"] == tmp_path / "ETTH1_pred14_seq4" / "XGBoost_波峰残差预测_测试" / "HUFL"
    assert captured[0]["prediction_label"] == "XGBoost波峰残差预测"
    assert captured[0]["filename_suffix"] == "XGBoost波峰残差预测"
    first_rows = captured[0]["prediction_rows"]
    assert isinstance(first_rows, list)
    assert first_rows[0]["baseline_peak_value"] == pytest.approx(11.5)
    assert first_rows[0]["baseline_peak_hour"] == 3


def test_run_xgboost_peak_value_training_writes_separate_outputs_and_seven_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_path = tmp_path / "model_matrix_seq96_pred336.csv"
    feature_columns_path = tmp_path / "feature_columns_seq96_pred336.txt"
    best_rule_path = tmp_path / "best_peak_value_baseline_by_group.csv"
    matrix = _minimal_matrix()
    matrix.to_csv(matrix_path, index=False)
    feature_columns_path.write_text("feature_a\nfeature_b\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                xgboost_module.TARGET_VARIABLE_COLUMN: target_variable,
                xgboost_module.HORIZON_COLUMN: 1,
                "best_baseline_name": "mean_last_4",
                "validation_MAE": 1.0,
                "validation_RMSE": 1.0,
                "validation_sMAPE": 0.2,
            }
            for target_variable in xgboost_module.EXPECTED_TARGET_VARIABLES
        ]
    ).to_csv(best_rule_path, index=False)

    monkeypatch.setitem(sys.modules, "xgboost", FakeXGBoost)

    outputs = xgboost_module.run_xgboost_peak_value_training(
        matrix_path=matrix_path,
        feature_columns_path=feature_columns_path,
        best_rule_baseline_path=best_rule_path,
        baseline_metrics_path=None,
        output_dir=tmp_path / "xgboost_peak_value",
        model_dir=tmp_path / "xgboost_peak_value" / "models",
        num_boost_round=5,
        early_stopping_rounds=2,
    )

    predictions = pd.read_csv(outputs.prediction_path)
    metrics = pd.read_csv(outputs.metrics_path)
    comparison = pd.read_csv(outputs.comparison_path)
    importance = pd.read_csv(outputs.feature_importance_path)

    assert len(outputs.model_paths) == 7
    assert all(path.suffix == ".json" for path in outputs.model_paths)
    assert all(path.exists() for path in outputs.model_paths)
    assert outputs.prediction_path.name == "xgboost_peak_value_predictions.csv"
    assert list(predictions.columns) == list(lightgbm_module.PREDICTION_COLUMNS)
    assert set(predictions["model_name"]) == {"xgboost_peak_value"}
    assert predictions["pred_peak_value"].tolist() == pytest.approx([9.5] * 14)
    assert {"MAE", "RMSE", "sMAPE"}.issubset(metrics.columns)
    assert {
        "MAE_improvement",
        "RMSE_improvement",
        "sMAPE_improvement",
        "imp",
    }.issubset(comparison.columns)
    assert set(importance["target_variable"]) == set(xgboost_module.EXPECTED_TARGET_VARIABLES)


def test_run_xgboost_peak_value_training_triggers_visualization_after_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "xgboost", FakeXGBoost)
    monkeypatch.setattr(
        xgboost_module,
        "load_model_matrix",
        lambda matrix_path, feature_columns_path: (
            pd.DataFrame(
                {
                    xgboost_module.TARGET_VARIABLE_COLUMN: list(
                        xgboost_module.EXPECTED_TARGET_VARIABLES
                    )
                }
            ),
            ["feature_a"],
        ),
    )

    def fake_train_one_target(
        xgb: object,
        target_variable: str,
        matrix: pd.DataFrame,
        feature_columns: list[str],
        model_dir: Path,
        num_boost_round: int,
        early_stopping_rounds: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
        rows = []
        for split in xgboost_module.EVAL_SPLITS:
            frame = pd.DataFrame(
                [
                    {
                        xgboost_module.SAMPLE_ID_COLUMN: f"{split}_{target_variable}",
                        xgboost_module.START_DATE_COLUMN: "2017-01-01",
                        xgboost_module.SPLIT_COLUMN: split,
                        xgboost_module.TARGET_VARIABLE_COLUMN: target_variable,
                        xgboost_module.HORIZON_COLUMN: 1,
                        xgboost_module.TARGET_PEAK_COLUMN: 10.0,
                        xgboost_module.BASELINE_PEAK_COLUMN: 8.0,
                        xgboost_module.TARGET_RESIDUAL_COLUMN: 2.0,
                    }
                ]
            )
            rows.append(xgboost_module.add_peak_value_predictions(frame, [1.5]))
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
        model_path = model_dir / f"{target_variable}_peak_value.json"
        model_path.write_text("fake model", encoding="utf-8")
        summary = {
            "target_variable": target_variable,
            "model_path": str(model_path),
            "train_rows": 1,
            "val_rows": 1,
            "test_rows": 1,
            "feature_count": 1,
            "best_iteration": 1,
            "best_validation_mae": 0.5,
        }
        return pd.concat(rows, ignore_index=True), importance, summary, model_path

    def fake_comparison(metrics: pd.DataFrame, best_rule: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "eval_level": "global",
                    xgboost_module.SPLIT_COLUMN: split,
                    "model_name": xgboost_module.MODEL_NAME,
                    "baseline_name": xgboost_module.BASELINE_NAME,
                    "best_rule_baseline_name": pd.NA,
                    xgboost_module.TARGET_VARIABLE_COLUMN: pd.NA,
                    xgboost_module.HORIZON_COLUMN: pd.NA,
                    "row_count": 7,
                    "MAE": 0.5,
                    "baseline_MAE": 1.0,
                    "MAE_improvement": 0.5,
                    "imp": 0.5,
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
                for split in xgboost_module.EVAL_SPLITS
            ]
        )

    plot_calls: list[dict[str, object]] = []

    def fake_plot_xgboost_peak_value_predictions(**kwargs: object) -> tuple[Path, ...]:
        plot_calls.append(kwargs)
        return (tmp_path / "plot.png",)

    monkeypatch.setattr(xgboost_module, "_train_one_target", fake_train_one_target)
    monkeypatch.setattr(xgboost_module, "_load_best_rule_baseline", lambda *args: pd.DataFrame())
    monkeypatch.setattr(xgboost_module, "compare_with_best_rule_baseline", fake_comparison)
    monkeypatch.setattr(
        xgboost_module,
        "plot_xgboost_peak_value_predictions",
        fake_plot_xgboost_peak_value_predictions,
    )

    outputs = xgboost_module.run_xgboost_peak_value_training(
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
    assert plot_calls[0]["splits"] == xgboost_module.EVAL_SPLITS
    assert outputs.plot_paths == (tmp_path / "plot.png",)
