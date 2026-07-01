"""Train one XGBoost peak-value residual regressor per target variable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from wave_dataset.visualization import plot_lightgbm_peak_value_prediction_batch


MODEL_MATRIX_PATH = Path("实验输出/results/features/model_matrix_seq96_pred336.csv")
FEATURE_COLUMNS_PATH = Path("实验输出/results/features/feature_columns_seq96_pred336.txt")
BEST_RULE_BASELINE_PATH = Path("实验输出/results/baselines/best_peak_value_baseline_by_group.csv")
BASELINE_METRICS_PATH = Path("实验输出/results/baselines/peak_value_baseline_metrics.csv")
OUTPUT_DIR = Path("实验输出/results/models/xgboost_peak_value")
MODEL_DIR = OUTPUT_DIR / "models"

PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_predictions.csv"
VAL_PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_val_predictions.csv"
TEST_PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_test_predictions.csv"
METRICS_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_metrics.csv"
COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_vs_best_rule_baseline.csv"
FEATURE_IMPORTANCE_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_feature_importance.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_model_summary.csv"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "xgboost_peak_value_report.md"

SAMPLE_ID_COLUMN = "样本ID"
START_DATE_COLUMN = "预测起点日期"
SPLIT_COLUMN = "数据集划分"
TARGET_VARIABLE_COLUMN = "目标变量"
HORIZON_COLUMN = "预测天数"
TARGET_PEAK_COLUMN = "目标峰值"
BASELINE_PEAK_COLUMN = "baseline_peak"
TARGET_RESIDUAL_COLUMN = "target_peak_residual"

TRAIN_SPLIT = "训练"
VAL_SPLIT = "验证"
TEST_SPLIT = "测试"
EVAL_SPLITS = (VAL_SPLIT, TEST_SPLIT)
EXPECTED_TARGET_VARIABLES = ("HUFL", "HULL", "LUFL", "LULL", "MUFL", "MULL", "OT")

MODEL_NAME = "xgboost_peak_value"
BASELINE_NAME = "best_rule_peak_value"
METRIC_COLUMNS = ("MAE", "RMSE", "sMAPE")
PREDICTION_COLUMNS = (
    SAMPLE_ID_COLUMN,
    START_DATE_COLUMN,
    SPLIT_COLUMN,
    TARGET_VARIABLE_COLUMN,
    HORIZON_COLUMN,
    TARGET_PEAK_COLUMN,
    BASELINE_PEAK_COLUMN,
    TARGET_RESIDUAL_COLUMN,
    "pred_peak_residual",
    "pred_peak_value",
    "model_name",
)


@dataclass(frozen=True)
class XGBoostPeakValueOutputs:
    prediction_path: Path
    val_prediction_path: Path
    test_prediction_path: Path
    metrics_path: Path
    comparison_path: Path
    feature_importance_path: Path
    summary_path: Path
    report_path: Path
    model_paths: Tuple[Path, ...]
    plot_paths: Tuple[Path, ...]


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = _missing_columns(df, columns)
    if missing:
        raise ValueError(f"{label} 缺少必要字段：{', '.join(missing)}")


def read_feature_columns(path: Union[Path, str]) -> List[str]:
    """Read the fixed feature list used by the LightGBM experiment."""

    feature_columns = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not feature_columns:
        raise ValueError(f"特征列清单为空：{path}")
    return feature_columns


def load_model_matrix(
    matrix_path: Union[Path, str],
    feature_columns_path: Union[Path, str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load the fixed model matrix and validate the training contract."""

    matrix = pd.read_csv(matrix_path)
    feature_columns = read_feature_columns(feature_columns_path)
    required_columns = (
        SAMPLE_ID_COLUMN,
        START_DATE_COLUMN,
        SPLIT_COLUMN,
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        TARGET_PEAK_COLUMN,
        BASELINE_PEAK_COLUMN,
        TARGET_RESIDUAL_COLUMN,
        *feature_columns,
    )
    _require_columns(matrix, required_columns, "模型矩阵")

    matrix[HORIZON_COLUMN] = pd.to_numeric(matrix[HORIZON_COLUMN], errors="coerce")
    if matrix[HORIZON_COLUMN].isna().any():
        raise ValueError(f"{HORIZON_COLUMN} 存在缺失或非数值")
    matrix[HORIZON_COLUMN] = matrix[HORIZON_COLUMN].astype(int)

    for column in (TARGET_PEAK_COLUMN, BASELINE_PEAK_COLUMN, TARGET_RESIDUAL_COLUMN, *feature_columns):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        if matrix[column].isna().any():
            raise ValueError(f"{column} 存在缺失或非数值")
        if not np.isfinite(matrix[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} 存在 Inf")

    split_values = set(matrix[SPLIT_COLUMN].dropna().astype(str))
    missing_splits = [split for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT) if split not in split_values]
    if missing_splits:
        raise ValueError(f"{SPLIT_COLUMN} 缺少必要划分：{', '.join(missing_splits)}")

    return matrix, feature_columns


def _safe_feature_names(feature_columns: Sequence[str]) -> List[str]:
    return [f"f{index}" for index, _ in enumerate(feature_columns)]


def _feature_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    safe_names = _safe_feature_names(feature_columns)
    frame = df.loc[:, list(feature_columns)].astype("float64").copy()
    frame.columns = safe_names
    return frame


def add_peak_value_predictions(
    frame: pd.DataFrame,
    pred_peak_residual: Sequence[float],
) -> pd.DataFrame:
    """Attach residual and restored peak-value predictions to an evaluation frame."""

    _require_columns(
        frame,
        (
            SAMPLE_ID_COLUMN,
            START_DATE_COLUMN,
            SPLIT_COLUMN,
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            TARGET_PEAK_COLUMN,
            BASELINE_PEAK_COLUMN,
            TARGET_RESIDUAL_COLUMN,
        ),
        "预测输入",
    )
    output = frame.loc[
        :,
        [
            SAMPLE_ID_COLUMN,
            START_DATE_COLUMN,
            SPLIT_COLUMN,
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            TARGET_PEAK_COLUMN,
            BASELINE_PEAK_COLUMN,
            TARGET_RESIDUAL_COLUMN,
        ],
    ].copy()
    residual = np.asarray(pred_peak_residual, dtype=float)
    if len(residual) != len(output):
        raise ValueError(f"预测残差长度不匹配：期望 {len(output)}，实际 {len(residual)}")
    output["pred_peak_residual"] = residual
    output["pred_peak_value"] = output[BASELINE_PEAK_COLUMN].astype(float) + residual
    output["model_name"] = MODEL_NAME
    return output.loc[:, list(PREDICTION_COLUMNS)]


def _metric_summary(group: pd.DataFrame) -> pd.Series:
    true = group[TARGET_PEAK_COLUMN].astype(float).to_numpy()
    pred = group["pred_peak_value"].astype(float).to_numpy()
    abs_error = np.abs(pred - true)
    squared_error = np.square(pred - true)
    denominator = np.abs(pred) + np.abs(true)
    smape_values = np.zeros_like(abs_error, dtype=float)
    np.divide(2.0 * abs_error, denominator, out=smape_values, where=denominator != 0.0)
    return pd.Series(
        {
            "row_count": int(len(group)),
            "MAE": float(np.mean(abs_error)),
            "RMSE": float(np.sqrt(np.mean(squared_error))),
            "sMAPE": float(np.mean(smape_values)),
        }
    )


def _evaluate_level(
    predictions: pd.DataFrame,
    eval_level: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for keys, group in predictions.groupby(list(group_columns), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, key_tuple))
        record.update(_metric_summary(group).to_dict())
        records.append(record)

    grouped = pd.DataFrame.from_records(records)
    grouped.insert(0, "eval_level", eval_level)
    grouped["model_name"] = MODEL_NAME
    if TARGET_VARIABLE_COLUMN not in grouped.columns:
        grouped[TARGET_VARIABLE_COLUMN] = pd.NA
    if HORIZON_COLUMN not in grouped.columns:
        grouped[HORIZON_COLUMN] = pd.NA
    return grouped[
        [
            "eval_level",
            SPLIT_COLUMN,
            "model_name",
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            "row_count",
            *METRIC_COLUMNS,
        ]
    ]


def evaluate_peak_value_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate model predictions at global, variable, horizon, and combined levels."""

    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "eval_level",
                SPLIT_COLUMN,
                "model_name",
                TARGET_VARIABLE_COLUMN,
                HORIZON_COLUMN,
                "row_count",
                *METRIC_COLUMNS,
            ]
        )

    _require_columns(predictions, (SPLIT_COLUMN, TARGET_PEAK_COLUMN, "pred_peak_value"), "预测结果")
    levels = (
        ("global", [SPLIT_COLUMN]),
        ("target_variable", [SPLIT_COLUMN, TARGET_VARIABLE_COLUMN]),
        ("horizon", [SPLIT_COLUMN, HORIZON_COLUMN]),
        ("target_variable_horizon", [SPLIT_COLUMN, TARGET_VARIABLE_COLUMN, HORIZON_COLUMN]),
    )
    metrics = [_evaluate_level(predictions, eval_level, columns) for eval_level, columns in levels]
    output = pd.concat(metrics, ignore_index=True)
    output[HORIZON_COLUMN] = pd.to_numeric(output[HORIZON_COLUMN], errors="ignore")
    return output


def _aggregate_best_rule_baseline(
    best_rule: pd.DataFrame,
    split: str,
    eval_level: str,
) -> pd.DataFrame:
    metric_map = {
        "MAE": f"{split}_MAE",
        "RMSE": f"{split}_RMSE",
        "sMAPE": f"{split}_sMAPE",
    }
    if eval_level == "target_variable_horizon":
        group_columns = [TARGET_VARIABLE_COLUMN, HORIZON_COLUMN]
    elif eval_level == "target_variable":
        group_columns = [TARGET_VARIABLE_COLUMN]
    elif eval_level == "horizon":
        group_columns = [HORIZON_COLUMN]
    elif eval_level == "global":
        group_columns = []
    else:
        raise ValueError(f"未知评估层级：{eval_level}")

    value_columns = [metric_map[metric] for metric in METRIC_COLUMNS]
    _require_columns(best_rule, [*group_columns, *value_columns], "最佳规则基线表")
    if best_rule.empty:
        return pd.DataFrame(columns=[*group_columns, "baseline_name", *value_columns])

    if group_columns:
        aggregated = best_rule.groupby(group_columns, dropna=False)[value_columns].mean().reset_index()
    else:
        aggregated = pd.DataFrame([{column: float(best_rule[column].mean()) for column in value_columns}])
    aggregated["baseline_name"] = BASELINE_NAME
    return aggregated


def _load_best_rule_baseline(
    best_rule_baseline_path: Union[Path, str],
    baseline_metrics_path: Optional[Union[Path, str]] = BASELINE_METRICS_PATH,
) -> pd.DataFrame:
    best_rule = pd.read_csv(best_rule_baseline_path)
    _require_columns(
        best_rule,
        (
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            "best_baseline_name",
            "validation_MAE",
            "validation_RMSE",
            "validation_sMAPE",
        ),
        "最佳规则基线表",
    )
    best_rule[HORIZON_COLUMN] = pd.to_numeric(best_rule[HORIZON_COLUMN], errors="coerce")
    if best_rule[HORIZON_COLUMN].isna().any():
        raise ValueError("最佳规则基线表的预测天数存在缺失或非数值")
    best_rule[HORIZON_COLUMN] = best_rule[HORIZON_COLUMN].astype(int)
    best_rule = best_rule.rename(
        columns={
            "validation_MAE": f"{VAL_SPLIT}_MAE",
            "validation_RMSE": f"{VAL_SPLIT}_RMSE",
            "validation_sMAPE": f"{VAL_SPLIT}_sMAPE",
        }
    )

    if baseline_metrics_path is not None and Path(baseline_metrics_path).exists():
        baseline_metrics = pd.read_csv(baseline_metrics_path)
        required_columns = (
            "eval_level",
            SPLIT_COLUMN,
            "baseline_name",
            TARGET_VARIABLE_COLUMN,
            HORIZON_COLUMN,
            *METRIC_COLUMNS,
        )
        _require_columns(baseline_metrics, required_columns, "规则基线指标表")
        test_metrics = baseline_metrics.loc[
            (baseline_metrics["eval_level"] == "target_variable_horizon")
            & (baseline_metrics[SPLIT_COLUMN] == TEST_SPLIT)
        ].copy()
        test_metrics[HORIZON_COLUMN] = pd.to_numeric(test_metrics[HORIZON_COLUMN], errors="coerce")
        test_metrics = test_metrics.rename(
            columns={
                "baseline_name": "best_baseline_name",
                "MAE": f"{TEST_SPLIT}_MAE",
                "RMSE": f"{TEST_SPLIT}_RMSE",
                "sMAPE": f"{TEST_SPLIT}_sMAPE",
            }
        )
        best_rule = best_rule.merge(
            test_metrics[
                [
                    TARGET_VARIABLE_COLUMN,
                    HORIZON_COLUMN,
                    "best_baseline_name",
                    f"{TEST_SPLIT}_MAE",
                    f"{TEST_SPLIT}_RMSE",
                    f"{TEST_SPLIT}_sMAPE",
                ]
            ],
            on=[TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, "best_baseline_name"],
            how="left",
        )

    for metric in METRIC_COLUMNS:
        test_column = f"{TEST_SPLIT}_{metric}"
        val_column = f"{VAL_SPLIT}_{metric}"
        if test_column not in best_rule.columns:
            best_rule[test_column] = np.nan
        best_rule[test_column] = best_rule[test_column].fillna(best_rule[val_column])
    return best_rule


def compare_with_best_rule_baseline(
    model_metrics: pd.DataFrame,
    best_rule_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model metrics to validation-selected rule baselines."""

    _require_columns(
        model_metrics,
        ("eval_level", SPLIT_COLUMN, TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, *METRIC_COLUMNS),
        "模型指标表",
    )
    best_rule = best_rule_baseline.copy()
    best_rule = best_rule.rename(
        columns={
            "validation_MAE": f"{VAL_SPLIT}_MAE",
            "validation_RMSE": f"{VAL_SPLIT}_RMSE",
            "validation_sMAPE": f"{VAL_SPLIT}_sMAPE",
        }
    )
    for metric in METRIC_COLUMNS:
        val_column = f"{VAL_SPLIT}_{metric}"
        test_column = f"{TEST_SPLIT}_{metric}"
        if val_column in best_rule.columns and test_column not in best_rule.columns:
            best_rule[test_column] = best_rule[val_column]
    best_rule[HORIZON_COLUMN] = pd.to_numeric(best_rule[HORIZON_COLUMN], errors="coerce").astype("Int64")

    comparison_frames: List[pd.DataFrame] = []
    for split in EVAL_SPLITS:
        split_metrics = model_metrics.loc[model_metrics[SPLIT_COLUMN] == split].copy()
        for eval_level in ("global", "target_variable", "horizon", "target_variable_horizon"):
            level_metrics = split_metrics.loc[split_metrics["eval_level"] == eval_level].copy()
            if level_metrics.empty:
                continue
            if HORIZON_COLUMN in level_metrics.columns:
                level_metrics[HORIZON_COLUMN] = pd.to_numeric(
                    level_metrics[HORIZON_COLUMN], errors="coerce"
                ).astype("Int64")

            aggregated = _aggregate_best_rule_baseline(best_rule, split, eval_level)
            if HORIZON_COLUMN in aggregated.columns:
                aggregated[HORIZON_COLUMN] = pd.to_numeric(
                    aggregated[HORIZON_COLUMN], errors="coerce"
                ).astype("Int64")

            join_columns = [
                column
                for column in (TARGET_VARIABLE_COLUMN, HORIZON_COLUMN)
                if column in aggregated.columns
                and column in level_metrics.columns
                and not level_metrics[column].isna().all()
            ]
            baseline_columns = [
                *join_columns,
                "baseline_name",
                *[f"{split}_{metric}" for metric in METRIC_COLUMNS],
            ]
            if join_columns:
                merged = level_metrics.merge(
                    aggregated.loc[:, baseline_columns],
                    on=join_columns,
                    how="left",
                )
            else:
                merged = level_metrics.copy()
                baseline_row = aggregated.iloc[0]
                for column in baseline_columns:
                    merged[column] = baseline_row[column]
            for metric in METRIC_COLUMNS:
                baseline_column = f"{split}_{metric}"
                merged = merged.rename(columns={baseline_column: f"baseline_{metric}"})
                merged[f"{metric}_improvement"] = (
                    merged[f"baseline_{metric}"] - merged[metric]
                ) / merged[f"baseline_{metric}"]
                merged[f"{metric}_exceeds_best_rule"] = merged[f"{metric}_improvement"] > 0
            comparison_frames.append(merged)

    comparison = pd.concat(comparison_frames, ignore_index=True)
    if "best_baseline_name" in best_rule.columns:
        best_names = best_rule.loc[
            :,
            [TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, "best_baseline_name"],
        ].rename(columns={"best_baseline_name": "best_rule_baseline_name"})
        best_names[HORIZON_COLUMN] = pd.to_numeric(best_names[HORIZON_COLUMN], errors="coerce").astype(
            "Int64"
        )
        comparison[HORIZON_COLUMN] = pd.to_numeric(
            comparison[HORIZON_COLUMN], errors="coerce"
        ).astype("Int64")
        comparison = comparison.merge(
            best_names,
            on=[TARGET_VARIABLE_COLUMN, HORIZON_COLUMN],
            how="left",
        )
    if "baseline_name" not in comparison.columns:
        comparison["baseline_name"] = BASELINE_NAME
    comparison["baseline_name"] = comparison["baseline_name"].fillna(BASELINE_NAME)
    if "best_rule_baseline_name" not in comparison.columns:
        comparison["best_rule_baseline_name"] = pd.NA

    output_columns = [
        "eval_level",
        SPLIT_COLUMN,
        "model_name",
        "baseline_name",
        "best_rule_baseline_name",
        TARGET_VARIABLE_COLUMN,
        HORIZON_COLUMN,
        "row_count",
        "MAE",
        "baseline_MAE",
        "MAE_improvement",
        "MAE_exceeds_best_rule",
        "RMSE",
        "baseline_RMSE",
        "RMSE_improvement",
        "RMSE_exceeds_best_rule",
        "sMAPE",
        "baseline_sMAPE",
        "sMAPE_improvement",
        "sMAPE_exceeds_best_rule",
    ]
    comparison["imp"] = comparison["MAE_improvement"]
    output_columns.insert(output_columns.index("MAE_exceeds_best_rule"), "imp")
    return comparison.loc[:, output_columns]


def _xgboost_params() -> Dict[str, object]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 2026,
        "tree_method": "hist",
        "device": "cuda",
        "nthread": 0,
    }


def _predict_with_best_iteration(model: object, matrix: object) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and int(best_iteration) >= 0:
        try:
            return model.predict(matrix, iteration_range=(0, int(best_iteration) + 1))
        except TypeError:
            return model.predict(matrix)
    return model.predict(matrix)


def _feature_importance_frame(
    model: object,
    target_variable: str,
    feature_columns: Sequence[str],
    best_iteration: int,
) -> pd.DataFrame:
    safe_names = _safe_feature_names(feature_columns)
    gain_map = model.get_score(importance_type="gain")
    weight_map = model.get_score(importance_type="weight")
    importance = pd.DataFrame(
        {
            "target_variable": target_variable,
            "feature": list(feature_columns),
            "importance_gain": [float(gain_map.get(name, 0.0)) for name in safe_names],
            "importance_split": [float(weight_map.get(name, 0.0)) for name in safe_names],
            "best_iteration": best_iteration,
        }
    )
    return importance.sort_values(
        ["target_variable", "importance_gain", "importance_split", "feature"],
        ascending=[True, False, False, True],
    )


def _train_one_target(
    xgb: object,
    target_variable: str,
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    model_dir: Path,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Path]:
    target_frame = matrix.loc[matrix[TARGET_VARIABLE_COLUMN] == target_variable].copy()
    train_df = target_frame.loc[target_frame[SPLIT_COLUMN] == TRAIN_SPLIT].copy()
    val_df = target_frame.loc[target_frame[SPLIT_COLUMN] == VAL_SPLIT].copy()
    test_df = target_frame.loc[target_frame[SPLIT_COLUMN] == TEST_SPLIT].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"{target_variable} 的训练/验证/测试数据不能为空")

    feature_names = _safe_feature_names(feature_columns)
    train_x = _feature_frame(train_df, feature_columns)
    val_x = _feature_frame(val_df, feature_columns)
    test_x = _feature_frame(test_df, feature_columns)
    train_y = train_df[TARGET_RESIDUAL_COLUMN].astype("float64")
    val_y = val_df[TARGET_RESIDUAL_COLUMN].astype("float64")

    train_matrix = xgb.DMatrix(train_x, label=train_y, feature_names=feature_names)
    val_matrix = xgb.DMatrix(val_x, label=val_y, feature_names=feature_names)
    test_matrix = xgb.DMatrix(test_x, feature_names=feature_names)
    model = xgb.train(
        params=_xgboost_params(),
        dtrain=train_matrix,
        num_boost_round=num_boost_round,
        evals=[(val_matrix, VAL_SPLIT)],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    val_predictions = add_peak_value_predictions(
        val_df,
        _predict_with_best_iteration(model, val_matrix),
    )
    test_predictions = add_peak_value_predictions(
        test_df,
        _predict_with_best_iteration(model, test_matrix),
    )
    predictions = pd.concat([val_predictions, test_predictions], ignore_index=True)

    best_iteration = int(getattr(model, "best_iteration", num_boost_round - 1) or 0)
    importance = _feature_importance_frame(
        model=model,
        target_variable=target_variable,
        feature_columns=feature_columns,
        best_iteration=best_iteration,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{target_variable}_peak_value.json"
    model.save_model(str(model_path))

    summary = {
        "target_variable": target_variable,
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_count": len(feature_columns),
        "best_iteration": best_iteration,
        "best_validation_mae": float(getattr(model, "best_score", np.nan)),
    }
    return predictions, importance, summary, model_path


def build_report(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    output_paths: Sequence[Path],
) -> str:
    """Build a compact markdown report for the XGBoost peak-value experiment."""

    global_rows = comparison.loc[
        (comparison["eval_level"] == "global") & comparison[SPLIT_COLUMN].isin(EVAL_SPLITS)
    ].copy()
    status_lines = []
    for split in EVAL_SPLITS:
        row = global_rows.loc[global_rows[SPLIT_COLUMN] == split]
        if row.empty:
            status_lines.append(f"- {split}：无全局对比结果")
            continue
        item = row.iloc[0]
        direction = "超过" if bool(item["MAE_exceeds_best_rule"]) else "未超过"
        status_lines.append(
            f"- {split}：{direction}最佳规则基线，"
            f"XGBoost MAE={item['MAE']:.6g}，"
            f"规则基线 MAE={item['baseline_MAE']:.6g}，"
            f"improvement={item['MAE_improvement']:.2%}"
        )

    summary_rows = "\n".join(
        f"- {row['target_variable']}：train={int(row['train_rows'])}, "
        f"val={int(row['val_rows'])}, test={int(row['test_rows'])}, "
        f"best_iteration={int(row['best_iteration'])}"
        for _, row in summary.iterrows()
    )
    metric_rows = metrics.loc[
        (metrics["eval_level"] == "global") & metrics[SPLIT_COLUMN].isin(EVAL_SPLITS)
    ]
    metric_text = "\n".join(
        f"- {row[SPLIT_COLUMN]}：MAE={row['MAE']:.6g}, RMSE={row['RMSE']:.6g}, sMAPE={row['sMAPE']:.6g}"
        for _, row in metric_rows.iterrows()
    )
    return "\n".join(
        [
            "# XGBoost peak_value 残差回归报告",
            "",
            "## 是否超过最佳规则基线",
            "",
            *status_lines,
            "",
            "## 全局指标",
            "",
            metric_text,
            "",
            "## 模型摘要",
            "",
            summary_rows,
            "",
            "## 输出文件",
            "",
            *[f"- `{path}`" for path in output_paths],
            "",
        ]
    )


def plot_xgboost_peak_value_predictions(
    prediction_csv: Union[Path, str] = PREDICTION_OUTPUT_PATH,
    hourly_csv: Union[Path, str] = "ETTh1.csv",
    output_root: Union[Path, str] = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    splits: Sequence[str] = EVAL_SPLITS,
    sample_count: int = 6,
    target_cols: Optional[Sequence[str]] = None,
) -> Tuple[Path, ...]:
    """Plot XGBoost residual peak-value predictions for validation and test splits."""

    path_map = plot_lightgbm_peak_value_prediction_batch(
        hourly_csv=hourly_csv,
        prediction_csv=prediction_csv,
        output_root=output_root,
        dataset_name=dataset_name,
        target_cols=list(target_cols) if target_cols is not None else None,
        splits=tuple(splits),
        sample_count=sample_count,
        plot_group_prefix="XGBoost_波峰残差预测",
        prediction_label="XGBoost波峰残差预测",
        filename_suffix="XGBoost波峰残差预测",
    )
    paths: List[Path] = []
    for split_paths in path_map.values():
        for target_paths in split_paths.values():
            paths.extend(target_paths)
    return tuple(paths)


def run_xgboost_peak_value_training(
    matrix_path: Union[Path, str] = MODEL_MATRIX_PATH,
    feature_columns_path: Union[Path, str] = FEATURE_COLUMNS_PATH,
    best_rule_baseline_path: Union[Path, str] = BEST_RULE_BASELINE_PATH,
    baseline_metrics_path: Optional[Union[Path, str]] = BASELINE_METRICS_PATH,
    output_dir: Union[Path, str] = OUTPUT_DIR,
    model_dir: Union[Path, str] = MODEL_DIR,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    plot_predictions: bool = True,
) -> XGBoostPeakValueOutputs:
    """Train seven XGBoost residual regressors and write all experiment artifacts."""

    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "当前 Python 环境缺少 xgboost；请在运行环境中安装 xgboost 后重试。"
        ) from exc

    resolved_output_dir = Path(output_dir)
    resolved_model_dir = Path(model_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_dir.mkdir(parents=True, exist_ok=True)

    matrix, feature_columns = load_model_matrix(matrix_path, feature_columns_path)
    target_variables = sorted(matrix[TARGET_VARIABLE_COLUMN].dropna().astype(str).unique())
    if target_variables != sorted(EXPECTED_TARGET_VARIABLES):
        raise ValueError(
            f"目标变量必须为 7 个固定变量：期望 {sorted(EXPECTED_TARGET_VARIABLES)}，实际 {target_variables}"
        )

    prediction_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    summary_records: List[Dict[str, object]] = []
    model_paths: List[Path] = []
    for target_variable in target_variables:
        predictions, importance, summary, model_path = _train_one_target(
            xgb=xgb,
            target_variable=target_variable,
            matrix=matrix,
            feature_columns=feature_columns,
            model_dir=resolved_model_dir,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        prediction_frames.append(predictions)
        importance_frames.append(importance)
        summary_records.append(summary)
        model_paths.append(model_path)

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        [SPLIT_COLUMN, TARGET_VARIABLE_COLUMN, HORIZON_COLUMN, SAMPLE_ID_COLUMN],
        kind="mergesort",
    )
    val_predictions = predictions.loc[predictions[SPLIT_COLUMN] == VAL_SPLIT].copy()
    test_predictions = predictions.loc[predictions[SPLIT_COLUMN] == TEST_SPLIT].copy()
    metrics = evaluate_peak_value_predictions(predictions)
    best_rule = _load_best_rule_baseline(best_rule_baseline_path, baseline_metrics_path)
    comparison = compare_with_best_rule_baseline(metrics, best_rule)
    importance = pd.concat(importance_frames, ignore_index=True)
    summary = pd.DataFrame.from_records(summary_records).sort_values("target_variable")

    prediction_path = resolved_output_dir / PREDICTION_OUTPUT_PATH.name
    val_prediction_path = resolved_output_dir / VAL_PREDICTION_OUTPUT_PATH.name
    test_prediction_path = resolved_output_dir / TEST_PREDICTION_OUTPUT_PATH.name
    metrics_path = resolved_output_dir / METRICS_OUTPUT_PATH.name
    comparison_path = resolved_output_dir / COMPARISON_OUTPUT_PATH.name
    feature_importance_path = resolved_output_dir / FEATURE_IMPORTANCE_OUTPUT_PATH.name
    summary_path = resolved_output_dir / SUMMARY_OUTPUT_PATH.name
    report_path = resolved_output_dir / REPORT_OUTPUT_PATH.name

    predictions.to_csv(prediction_path, index=False)
    val_predictions.to_csv(val_prediction_path, index=False)
    test_predictions.to_csv(test_prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    importance.to_csv(feature_importance_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_paths: Tuple[Path, ...] = ()
    if plot_predictions:
        plot_paths = plot_xgboost_peak_value_predictions(
            prediction_csv=prediction_path,
            splits=EVAL_SPLITS,
        )

    report = build_report(
        summary=summary,
        metrics=metrics,
        comparison=comparison,
        output_paths=[
            prediction_path,
            val_prediction_path,
            test_prediction_path,
            metrics_path,
            comparison_path,
            feature_importance_path,
            summary_path,
            report_path,
            *model_paths,
            *plot_paths,
        ],
    )
    report_path.write_text(report, encoding="utf-8")

    return XGBoostPeakValueOutputs(
        prediction_path=prediction_path,
        val_prediction_path=val_prediction_path,
        test_prediction_path=test_prediction_path,
        metrics_path=metrics_path,
        comparison_path=comparison_path,
        feature_importance_path=feature_importance_path,
        summary_path=summary_path,
        report_path=report_path,
        model_paths=tuple(model_paths),
        plot_paths=plot_paths,
    )


def main() -> None:
    outputs = run_xgboost_peak_value_training()
    comparison = pd.read_csv(outputs.comparison_path)
    global_comparison = comparison.loc[
        (comparison["eval_level"] == "global") & comparison[SPLIT_COLUMN].isin(EVAL_SPLITS)
    ]
    print(f"prediction_path: {outputs.prediction_path}")
    print(f"val_prediction_path: {outputs.val_prediction_path}")
    print(f"test_prediction_path: {outputs.test_prediction_path}")
    print(f"metrics_path: {outputs.metrics_path}")
    print(f"comparison_path: {outputs.comparison_path}")
    print(f"feature_importance_path: {outputs.feature_importance_path}")
    print(f"summary_path: {outputs.summary_path}")
    print(f"report_path: {outputs.report_path}")
    print(f"model_count: {len(outputs.model_paths)}")
    print(f"plot_count: {len(outputs.plot_paths)}")
    for _, row in global_comparison.iterrows():
        direction = "YES" if bool(row["MAE_exceeds_best_rule"]) else "NO"
        print(
            f"{row[SPLIT_COLUMN]} exceeds_best_rule_by_MAE: {direction}; "
            f"MAE={row['MAE']:.6g}; baseline_MAE={row['baseline_MAE']:.6g}; "
            f"improvement={row['MAE_improvement']:.2%}"
        )


if __name__ == "__main__":
    main()
