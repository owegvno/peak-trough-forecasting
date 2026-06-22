from __future__ import annotations

import argparse
from pathlib import Path

from .peak_dataset import build_peak_dataset
from .turning_points import build_turning_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ETTh1 波峰预测样本 CSV 和拐点标签 CSV")
    parser.add_argument("--input", default="ETTh1.csv", help="输入 ETTh1 CSV 路径")
    parser.add_argument("--output-dir", default="数据集", help="输出数据集目录")
    parser.add_argument("--seq-days", type=int, default=4, help="输入完整自然日数量")
    parser.add_argument("--pred-days", type=int, default=14, help="预测完整自然日数量")
    parser.add_argument("--smooth-window", type=int, default=5, help="拐点平滑窗口")
    parser.add_argument("--mad-window", type=int, default=168, help="拐点 MAD 窗口")
    parser.add_argument("--prominence-multiplier", type=float, default=0.5, help="拐点显著性倍数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    peak_result = build_peak_dataset(input_path, output_dir, seq_days=args.seq_days, pred_days=args.pred_days)
    actual_output_dir = Path(peak_result["output_dir"])
    turning_result = build_turning_dataset(
        input_path,
        actual_output_dir,
        smooth_window=args.smooth_window,
        mad_window=args.mad_window,
        prominence_multiplier=args.prominence_multiplier,
    )
    print("峰值数据集生成完成")
    print(f"  完整自然日数量: {peak_result['complete_days']}")
    print(f"  样本行数: {peak_result['sample_rows']}")
    print(f"  划分行数: {peak_result['split_counts']}")
    print("拐点数据集生成完成")
    print(f"  原始小时行数: {turning_result['raw_rows']}")
    print(f"输出目录: {actual_output_dir}")


if __name__ == "__main__":
    main()
