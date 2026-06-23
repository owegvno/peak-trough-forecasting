from __future__ import annotations

import argparse
from pathlib import Path

from .visualization import (
    DEFAULT_BASELINE_HOUR_PREDICTION_CSV,
    DEFAULT_BASELINE_VALUE_PREDICTION_CSV,
    plot_peak_hour_prediction_batch,
    plot_peak_value_prediction_batch,
    plot_dataset_visualizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制数据集样本窗口的波峰/波谷和拐点图")
    parser.add_argument("--input", default="ETTh1.csv", help="输入 ETTh1 小时级 CSV 路径")
    parser.add_argument("--data-root", default="数据集", help="数据集根目录")
    parser.add_argument("--dataset-name", default="ETTH1_pred14_seq4", help="数据集文件夹名")
    parser.add_argument("--target-col", default="", help="目标变量；不传则绘制全部变量")
    parser.add_argument("--split", default="训练", help="样本划分，默认 训练")
    parser.add_argument("--output-root", default="数据集可视化", help="图片输出根目录")
    parser.add_argument("--baseline-peaks", action="store_true", help="绘制波峰预测结果，而不是标签可视化")
    parser.add_argument("--value-prediction", default=str(DEFAULT_BASELINE_VALUE_PREDICTION_CSV), help="波峰值预测 CSV")
    parser.add_argument("--hour-prediction", default=str(DEFAULT_BASELINE_HOUR_PREDICTION_CSV), help="波峰小时预测 CSV")
    parser.add_argument("--sample-count", type=int, default=6, help="每个变量均匀采样样本数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_cols = [args.target_col] if args.target_col else None
    if args.baseline_peaks:
        if Path(args.value_prediction).exists():
            value_paths = plot_peak_value_prediction_batch(
                Path(args.input),
                value_prediction_csv=Path(args.value_prediction),
                output_root=Path(args.output_root),
                dataset_name=args.dataset_name,
                target_cols=target_cols,
                split=args.split,
                sample_count=args.sample_count,
            )
            for baseline_name, baseline_paths in value_paths.items():
                print(f"波峰值 {baseline_name}: " + ", ".join(f"{col}={len(paths)}" for col, paths in baseline_paths.items()))
        if Path(args.hour_prediction).exists():
            hour_paths = plot_peak_hour_prediction_batch(
                Path(args.input),
                hour_prediction_csv=Path(args.hour_prediction),
                output_root=Path(args.output_root),
                dataset_name=args.dataset_name,
                target_cols=target_cols,
                split=args.split,
                sample_count=args.sample_count,
            )
            for baseline_name, baseline_paths in hour_paths.items():
                print(f"波峰小时 {baseline_name}: " + ", ".join(f"{col}={len(paths)}" for col, paths in baseline_paths.items()))
        return

    paths = plot_dataset_visualizations(
        Path(args.input),
        data_root=Path(args.data_root),
        dataset_name=args.dataset_name,
        output_root=Path(args.output_root),
        target_cols=target_cols,
        split=args.split,
    )
    for col, col_paths in paths.items():
        print(f"{col} 波峰波谷图: {col_paths['peaks_troughs']}")
        print(f"{col} 拐点图: {col_paths['turning_points']}")


if __name__ == "__main__":
    main()
