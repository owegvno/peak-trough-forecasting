from __future__ import annotations

import argparse
from pathlib import Path

from .visualization import plot_dataset_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制数据集样本窗口的波峰/波谷和拐点图")
    parser.add_argument("--input", default="ETTh1.csv", help="输入 ETTh1 小时级 CSV 路径")
    parser.add_argument("--data-root", default="数据集", help="数据集根目录")
    parser.add_argument("--dataset-name", default="ETTH1_pred14_seq4", help="数据集文件夹名")
    parser.add_argument("--target-col", default="", help="目标变量；不传则绘制全部变量")
    parser.add_argument("--split", default="训练", help="样本划分，默认 训练")
    parser.add_argument("--output-root", default="数据集可视化", help="图片输出根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_cols = [args.target_col] if args.target_col else None
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
