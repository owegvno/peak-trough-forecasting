"""Helpers for optional baseline prediction visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from wave_dataset.visualization import (
    plot_peak_hour_prediction_batch,
    plot_peak_value_prediction_batch,
)


def maybe_plot_peak_baseline_predictions(
    hourly_csv: Union[str, Path] = "ETTh1.csv",
    value_prediction_csv: Union[str, Path] = "实验输出/results/baselines/peak_value_baseline_predictions.csv",
    hour_prediction_csv: Union[str, Path] = "实验输出/results/baselines/peak_hour_baseline_predictions.csv",
    output_root: Union[str, Path] = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    split: str = "验证",
    sample_count: int = 6,
    plot_value: bool = True,
    plot_hour: bool = True,
) -> dict[str, dict[str, dict[str, list[Path]]]]:
    """Plot available peak-value and peak-hour predictions for every baseline_name."""

    value_path = Path(value_prediction_csv)
    hour_path = Path(hour_prediction_csv)
    plot_paths: dict[str, dict[str, dict[str, list[Path]]]] = {}
    if plot_value and value_path.exists():
        plot_paths["波峰值"] = plot_peak_value_prediction_batch(
            hourly_csv=hourly_csv,
            value_prediction_csv=value_path,
            output_root=output_root,
            dataset_name=dataset_name,
            split=split,
            sample_count=sample_count,
        )
    elif plot_value:
        print(f"跳过波峰值可视化，缺少文件: {value_path}")

    if plot_hour and hour_path.exists():
        plot_paths["波峰小时"] = plot_peak_hour_prediction_batch(
            hourly_csv=hourly_csv,
            hour_prediction_csv=hour_path,
            output_root=output_root,
            dataset_name=dataset_name,
            split=split,
            sample_count=sample_count,
        )
    elif plot_hour:
        print(f"跳过波峰小时可视化，缺少文件: {hour_path}")
    return plot_paths
