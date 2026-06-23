"""Helpers for optional baseline prediction visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from wave_dataset.visualization import (
    DEFAULT_PEAK_HOUR_BASELINE_NAME,
    DEFAULT_PEAK_PLOT_GROUP_NAME,
    DEFAULT_PEAK_VALUE_BASELINE_NAME,
    plot_baseline_peak_prediction_batch,
)


def maybe_plot_peak_baseline_predictions(
    hourly_csv: Union[str, Path] = "ETTh1.csv",
    value_prediction_csv: Union[str, Path] = "实验输出/results/baselines/peak_value_baseline_predictions.csv",
    hour_prediction_csv: Union[str, Path] = "实验输出/results/baselines/peak_hour_baseline_predictions.csv",
    output_root: Union[str, Path] = "数据集可视化",
    dataset_name: str = "ETTH1_pred14_seq4",
    split: str = "验证",
    sample_count: int = 20,
    value_baseline_name: Optional[str] = DEFAULT_PEAK_VALUE_BASELINE_NAME,
    hour_baseline_name: Optional[str] = DEFAULT_PEAK_HOUR_BASELINE_NAME,
    plot_group_name: str = DEFAULT_PEAK_PLOT_GROUP_NAME,
) -> dict[str, list[Path]]:
    """Plot peak baseline predictions when both value and hour CSVs are available."""

    value_path = Path(value_prediction_csv)
    hour_path = Path(hour_prediction_csv)
    missing = [str(path) for path in (value_path, hour_path) if not path.exists()]
    if missing:
        print("跳过波峰预测可视化，缺少文件: " + ", ".join(missing))
        return {}

    return plot_baseline_peak_prediction_batch(
        hourly_csv=hourly_csv,
        value_prediction_csv=value_path,
        hour_prediction_csv=hour_path,
        output_root=output_root,
        dataset_name=dataset_name,
        split=split,
        sample_count=sample_count,
        value_baseline_name=value_baseline_name,
        hour_baseline_name=hour_baseline_name,
        plot_group_name=plot_group_name,
    )
