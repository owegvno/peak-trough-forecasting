#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Savitzky-Golay 平滑、峰谷显著性和自适应 MAD 阈值检测长程时序拐点。

运行示例：
conda run -n Ada-MSHyper python 自适应显著性拐点检测.py
conda run -n Ada-MSHyper python 自适应显著性拐点检测.py --列名 OT
conda run -n Ada-MSHyper python 自适应显著性拐点检测.py --列名 HUFL,HULL OT

输出说明：
每个处理列会单独生成一个 CSV 文件，文件名默认为“列名拐点.csv”。
拐点类型：1 表示峰值型拐点，-1 表示谷值型拐点，0 表示不是拐点。
拐点显著性：当前峰或谷相对周边形态的突出程度，用于衡量拐点强弱。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


默认输入文件 = Path("ETTh1.csv")
默认输出目录 = Path("自适应拐点类型结果")
默认时间列 = "date"
默认平滑窗口 = 5
默认多项式阶数 = 2
默认显著性倍数 = 0.5
默认最小间隔 = 2
默认最小宽度 = 1.0
默认MAD窗口 = 168
默认最小显著性 = 0.0
默认文件名后缀 = "拐点"
输出列名 = ["日期", "对应数值", "平滑数值", "拐点类型", "拐点显著性", "显著性阈值", "拐点有效标记"]


def 规范化列名(列名参数: Iterable[str] | None) -> list[str] | None:
    """把命令行传入的列名整理为列表；未传入时返回 None。"""
    if 列名参数 is None:
        return None

    结果: list[str] = []
    for 原始列名 in 列名参数:
        for 列名 in str(原始列名).split(","):
            列名 = 列名.strip()
            if 列名:
                结果.append(列名)

    return 结果 or None


def 获取待处理列名(数据表: pd.DataFrame, 时间列: str, 处理列名: list[str] | None) -> list[str]:
    """确定需要处理的列；未指定时处理除时间列外的所有数值列。"""
    if 时间列 not in 数据表.columns:
        raise ValueError(f"未找到时间列：{时间列}")

    if 处理列名:
        缺失列 = [列名 for 列名 in 处理列名 if 列名 not in 数据表.columns]
        if 缺失列:
            raise ValueError(f"输入文件中不存在这些列：{', '.join(缺失列)}")
        if 时间列 in 处理列名:
            raise ValueError(f"时间列 {时间列} 不能作为待处理列")
        return 处理列名

    待处理列名: list[str] = []
    for 列名 in 数据表.columns:
        if 列名 == 时间列:
            continue
        数值列 = pd.to_numeric(数据表[列名], errors="coerce")
        if 数值列.notna().any():
            待处理列名.append(列名)

    if not 待处理列名:
        raise ValueError("未找到可处理的数值列")

    return 待处理列名


def 补齐缺失值(数值序列: pd.Series) -> pd.Series:
    """用线性插值补齐缺失值，便于平滑和峰谷检测。"""
    return 数值序列.interpolate(method="linear", limit_direction="both")


def 标准化平滑窗口(数据行数: int, 平滑窗口: int, 多项式阶数: int) -> int:
    """把平滑窗口整理为 Savitzky-Golay 可用的奇数窗口。"""
    if 平滑窗口 <= 1:
        return 1
    if 多项式阶数 < 0:
        raise ValueError("多项式阶数不能小于 0")
    if 数据行数 <= 0:
        raise ValueError("数据行数必须大于 0")

    窗口 = int(平滑窗口)
    if 窗口 % 2 == 0:
        窗口 += 1

    最大窗口 = 数据行数 if 数据行数 % 2 == 1 else 数据行数 - 1
    窗口 = min(窗口, 最大窗口)
    最小窗口 = 多项式阶数 + 2
    if 最小窗口 % 2 == 0:
        最小窗口 += 1

    if 窗口 < 最小窗口:
        return 1

    return 窗口


def 计算平滑序列(数值序列: pd.Series, 平滑窗口: int, 多项式阶数: int) -> np.ndarray:
    """对数值序列做 Savitzky-Golay 平滑；窗口不足时返回补齐后的原序列。"""
    补齐序列 = 补齐缺失值(数值序列)
    平滑窗口 = 标准化平滑窗口(len(补齐序列), 平滑窗口, 多项式阶数)
    if 平滑窗口 <= 1:
        return 补齐序列.to_numpy(dtype=float)

    实际阶数 = min(int(多项式阶数), 平滑窗口 - 1)
    return savgol_filter(补齐序列.to_numpy(dtype=float), window_length=平滑窗口, polyorder=实际阶数, mode="interp")


def 计算滚动MAD阈值(
    原始数值: pd.Series,
    平滑数值: np.ndarray,
    MAD窗口: int,
    显著性倍数: float,
    最小显著性: float,
) -> np.ndarray:
    """用平滑残差的滚动 MAD 估计局部噪声阈值。"""
    if MAD窗口 <= 0:
        raise ValueError("MAD窗口必须大于 0")
    if 显著性倍数 < 0:
        raise ValueError("显著性倍数不能小于 0")
    if 最小显著性 < 0:
        raise ValueError("最小显著性不能小于 0")

    补齐序列 = 补齐缺失值(原始数值)
    残差绝对值 = (补齐序列 - pd.Series(平滑数值, index=原始数值.index)).abs()
    滚动中位数 = 残差绝对值.rolling(window=MAD窗口, min_periods=1, center=True).median()
    滚动MAD = (残差绝对值 - 滚动中位数).abs().rolling(window=MAD窗口, min_periods=1, center=True).median()

    全局MAD = float((残差绝对值 - 残差绝对值.median()).abs().median())
    if not np.isfinite(全局MAD):
        全局MAD = 0.0

    阈值 = 滚动MAD.fillna(全局MAD).to_numpy(dtype=float) * 1.4826 * 显著性倍数
    return np.maximum(阈值, float(最小显著性))


def 回贴候选索引到原始极值(
    候选索引: np.ndarray,
    原始数值: pd.Series,
    候选类型: int,
    搜索半径: int,
) -> np.ndarray:
    """把平滑序列上的候选峰谷回贴到原始序列窗口内最近的真实极值点。"""
    原始数组 = 原始数值.to_numpy(dtype=float)
    回贴索引: list[int] = []

    for 行号 in 候选索引:
        左边界 = max(0, int(行号) - 搜索半径)
        右边界 = min(len(原始数组), int(行号) + 搜索半径 + 1)
        局部极值位置: list[int] = []

        for 位置 in range(左边界, 右边界):
            if not np.isfinite(原始数组[位置]):
                continue
            if 位置 == 0 or 位置 == len(原始数组) - 1:
                continue
            左值 = 原始数组[位置 - 1]
            当前值 = 原始数组[位置]
            右值 = 原始数组[位置 + 1]
            if not (np.isfinite(左值) and np.isfinite(右值)):
                continue

            if 候选类型 == 1 and 当前值 >= 左值 and 当前值 >= 右值:
                局部极值位置.append(位置)
            elif 候选类型 == -1 and 当前值 <= 左值 and 当前值 <= 右值:
                局部极值位置.append(位置)

        if 局部极值位置:
            极值位置数组 = np.array(局部极值位置, dtype=int)
            最近位置 = int(极值位置数组[np.argmin(np.abs(极值位置数组 - int(行号)))])
            回贴索引.append(最近位置)
            continue

        窗口值 = 原始数组[左边界:右边界]
        有效位置 = np.flatnonzero(np.isfinite(窗口值))
        if len(有效位置) == 0:
            回贴索引.append(int(行号))
            continue

        有效值 = 窗口值[有效位置]
        极值 = np.max(有效值) if 候选类型 == 1 else np.min(有效值)
        极值相对位置 = 有效位置[有效值 == 极值]
        极值绝对位置 = 左边界 + 极值相对位置
        最近位置 = int(极值绝对位置[np.argmin(np.abs(极值绝对位置 - int(行号)))])
        回贴索引.append(最近位置)

    return np.array(回贴索引, dtype=int)


def 写入候选拐点(
    候选索引: np.ndarray,
    显著性: np.ndarray,
    候选类型: int,
    拐点类型: np.ndarray,
    拐点显著性: np.ndarray,
    显著性阈值: np.ndarray,
    有效掩码: np.ndarray,
) -> None:
    """把通过显著性阈值过滤的候选峰谷写入结果数组。"""
    for 序号, 行号 in enumerate(候选索引):
        当前显著性 = float(显著性[序号])
        if not 有效掩码[行号]:
            continue
        if 当前显著性 < 显著性阈值[行号]:
            continue
        if 当前显著性 <= 拐点显著性[行号]:
            continue

        拐点类型[行号] = 候选类型
        拐点显著性[行号] = 当前显著性


def 计算单列自适应拐点(
    数据表: pd.DataFrame,
    列名: str,
    时间列: str = 默认时间列,
    平滑窗口: int = 默认平滑窗口,
    多项式阶数: int = 默认多项式阶数,
    显著性倍数: float = 默认显著性倍数,
    最小间隔: int = 默认最小间隔,
    最小宽度: float = 默认最小宽度,
    MAD窗口: int = 默认MAD窗口,
    最小显著性: float = 默认最小显著性,
) -> pd.DataFrame:
    """按自适应显著性峰谷规则计算单列拐点类型。"""
    if 时间列 not in 数据表.columns:
        raise ValueError(f"未找到时间列：{时间列}")
    if 列名 not in 数据表.columns:
        raise ValueError(f"输入文件中不存在列：{列名}")
    if 列名 == 时间列:
        raise ValueError(f"时间列 {时间列} 不能作为待处理列")
    if 最小间隔 <= 0:
        raise ValueError("最小间隔必须大于 0")
    if 最小宽度 <= 0:
        raise ValueError("最小宽度必须大于 0")

    数值序列 = pd.to_numeric(数据表[列名], errors="coerce")
    if 数值序列.isna().all():
        raise ValueError(f"列 {列名} 没有可处理的数值")

    有效掩码 = 数值序列.notna().to_numpy()
    平滑数值 = 计算平滑序列(数值序列, 平滑窗口, 多项式阶数)
    显著性阈值 = 计算滚动MAD阈值(数值序列, 平滑数值, MAD窗口, 显著性倍数, 最小显著性)
    搜索半径 = max(1, int(平滑窗口))

    拐点类型 = np.zeros(len(数据表), dtype=int)
    拐点显著性 = np.zeros(len(数据表), dtype=float)

    峰值索引, 峰值属性 = find_peaks(平滑数值, prominence=0, distance=int(最小间隔), width=float(最小宽度))
    谷值索引, 谷值属性 = find_peaks(-平滑数值, prominence=0, distance=int(最小间隔), width=float(最小宽度))
    峰值索引 = 回贴候选索引到原始极值(峰值索引, 数值序列, 1, 搜索半径)
    谷值索引 = 回贴候选索引到原始极值(谷值索引, 数值序列, -1, 搜索半径)

    写入候选拐点(
        候选索引=峰值索引,
        显著性=峰值属性.get("prominences", np.array([], dtype=float)),
        候选类型=1,
        拐点类型=拐点类型,
        拐点显著性=拐点显著性,
        显著性阈值=显著性阈值,
        有效掩码=有效掩码,
    )
    写入候选拐点(
        候选索引=谷值索引,
        显著性=谷值属性.get("prominences", np.array([], dtype=float)),
        候选类型=-1,
        拐点类型=拐点类型,
        拐点显著性=拐点显著性,
        显著性阈值=显著性阈值,
        有效掩码=有效掩码,
    )

    return pd.DataFrame(
        {
            "日期": 数据表[时间列].tolist(),
            "对应数值": 数值序列.tolist(),
            "平滑数值": 平滑数值.tolist(),
            "拐点类型": 拐点类型.tolist(),
            "拐点显著性": 拐点显著性.tolist(),
            "显著性阈值": 显著性阈值.tolist(),
            "拐点有效标记": 有效掩码.astype(int).tolist(),
        },
        columns=输出列名,
    )


def 处理数据文件(
    输入文件: str | Path = 默认输入文件,
    输出目录: str | Path = 默认输出目录,
    处理列名: list[str] | None = None,
    时间列: str = 默认时间列,
    平滑窗口: int = 默认平滑窗口,
    多项式阶数: int = 默认多项式阶数,
    显著性倍数: float = 默认显著性倍数,
    最小间隔: int = 默认最小间隔,
    最小宽度: float = 默认最小宽度,
    MAD窗口: int = 默认MAD窗口,
    最小显著性: float = 默认最小显著性,
    文件名后缀: str = 默认文件名后缀,
) -> list[Path]:
    """读取输入 CSV，并为每个处理列分别生成自适应拐点 CSV 文件。"""
    输入路径 = Path(输入文件)
    输出路径 = Path(输出目录)

    if not 输入路径.exists():
        raise FileNotFoundError(f"输入文件不存在：{输入路径}")
    if not 文件名后缀:
        raise ValueError("文件名后缀不能为空")

    数据表 = pd.read_csv(输入路径)
    待处理列名 = 获取待处理列名(数据表, 时间列, 处理列名)
    输出路径.mkdir(parents=True, exist_ok=True)

    生成文件: list[Path] = []
    for 列名 in 待处理列名:
        拐点数据 = 计算单列自适应拐点(
            数据表=数据表,
            列名=列名,
            时间列=时间列,
            平滑窗口=平滑窗口,
            多项式阶数=多项式阶数,
            显著性倍数=显著性倍数,
            最小间隔=最小间隔,
            最小宽度=最小宽度,
            MAD窗口=MAD窗口,
            最小显著性=最小显著性,
        )
        输出文件 = 输出路径 / f"{列名}{文件名后缀}.csv"
        拐点数据.to_csv(输出文件, index=False, encoding="utf-8")
        生成文件.append(输出文件)

    return 生成文件


def 构建命令行参数() -> argparse.Namespace:
    """构建命令行参数，支持选择输入文件、输出目录、处理列和检测参数。"""
    解析器 = argparse.ArgumentParser(description="使用平滑和自适应显著性阈值检测 ETTh1.csv 每列拐点")
    解析器.add_argument("--输入文件", default=str(默认输入文件), help="输入 CSV 文件路径，默认 ETTh1.csv")
    解析器.add_argument("--输出目录", default=str(默认输出目录), help="输出目录，默认 自适应拐点类型结果")
    解析器.add_argument("--列名", nargs="+", help="要处理的列名；可写多个，也可用逗号分隔；未传入时处理所有数值列")
    解析器.add_argument("--时间列", default=默认时间列, help="时间列列名，默认 date")
    解析器.add_argument("--平滑窗口", type=int, default=默认平滑窗口, help="Savitzky-Golay 平滑窗口，默认 5")
    解析器.add_argument("--多项式阶数", type=int, default=默认多项式阶数, help="Savitzky-Golay 多项式阶数，默认 2")
    解析器.add_argument("--显著性倍数", type=float, default=默认显著性倍数, help="滚动 MAD 噪声阈值倍数，默认 0.5")
    解析器.add_argument("--最小间隔", type=int, default=默认最小间隔, help="相邻同类峰谷候选点的最小间隔，默认 2")
    解析器.add_argument("--最小宽度", type=float, default=默认最小宽度, help="峰谷最小宽度，默认 1.0")
    解析器.add_argument("--MAD窗口", type=int, default=默认MAD窗口, help="滚动 MAD 窗口，默认 168")
    解析器.add_argument("--最小显著性", type=float, default=默认最小显著性, help="绝对最小显著性阈值，默认 0.0")
    解析器.add_argument("--文件名后缀", default=默认文件名后缀, help="输出文件名后缀，默认 拐点")
    return 解析器.parse_args()


def main() -> None:
    """命令行入口。"""
    参数 = 构建命令行参数()
    生成文件 = 处理数据文件(
        输入文件=参数.输入文件,
        输出目录=参数.输出目录,
        处理列名=规范化列名(参数.列名),
        时间列=参数.时间列,
        平滑窗口=参数.平滑窗口,
        多项式阶数=参数.多项式阶数,
        显著性倍数=参数.显著性倍数,
        最小间隔=参数.最小间隔,
        最小宽度=参数.最小宽度,
        MAD窗口=参数.MAD窗口,
        最小显著性=参数.最小显著性,
        文件名后缀=参数.文件名后缀,
    )

    print("已生成以下文件：")
    for 文件 in 生成文件:
        print(文件)


if __name__ == "__main__":
    main()
