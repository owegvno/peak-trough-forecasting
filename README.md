# ETTh1 波峰、波谷与拐点数据集生成

本目录用于从项目根目录的 `ETTh1.csv` 生成两类 CSV：

1. 波峰/波谷预测样本长表。
2. 每小时拐点标签表。

当前版本主要完成：

1. 波峰/波谷训练数据准备。
2. 拐点标签训练数据准备。
3. 按数据集文件夹批量生成波峰/波谷和拐点可视化图。

本阶段不运行规则基线、LightGBM、XGBoost 或模型评估。

## 环境

在 WSL 中使用 `Ada-MSHyper` 环境：

```bash
cd /home/wmb/test/波峰波谷预测
PYTHONPATH='src' /home/wmb/anaconda3/envs/Ada-MSHyper/bin/python -m wave_dataset.build_datasets
```

## 生成数据集

默认配置：

```text
seq_len = 96 小时
pred_len = 336 小时
目标列 = HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
```

运行：

```bash
cd /home/wmb/test/波峰波谷预测
PYTHONPATH='src' /home/wmb/anaconda3/envs/Ada-MSHyper/bin/python -m wave_dataset.build_datasets \
  --input ETTh1.csv \
  --output-dir '数据集' \
  --seq-days 4 \
  --pred-days 14
```

`--seq-days` 和 `--pred-days` 可省略，省略时默认分别为 4 和 14。实际输出目录会放在 `数据集` 文件夹下，并自动带上数据集名和窗口参数，例如 `数据集/ETTH1_pred14_seq4`。

主要输出：

```text
数据集/ETTH1_pred14_seq4/完整自然日清单.csv
数据集/ETTH1_pred14_seq4/日周期峰值标签.csv
数据集/ETTH1_pred14_seq4/峰谷数据集生成报告.md
数据集/ETTH1_pred14_seq4/HUFL/峰谷预测样本_seq96_pred336_HUFL.csv
数据集/ETTH1_pred14_seq4/HUFL/峰谷预测样本_seq96_pred336_HUFL_训练.csv
数据集/ETTH1_pred14_seq4/HUFL/峰谷预测样本_seq96_pred336_HUFL_验证.csv
数据集/ETTH1_pred14_seq4/HUFL/峰谷预测样本_seq96_pred336_HUFL_测试.csv
...
数据集/ETTH1_pred14_seq4/OT/峰谷预测样本_seq96_pred336_OT.csv
数据集/ETTH1_pred14_seq4/拐点标签/HUFL拐点.csv
...
数据集/ETTH1_pred14_seq4/拐点标签/OT拐点.csv
```

## 波峰与波谷字段

每日标签表中，每个变量同时输出：

```text
变量_峰值
变量_峰值小时
变量_谷值
变量_谷值小时
```

预测样本中，每行表示：

```text
一个样本ID + 一个预测起点日期 + 一个目标变量 + 一个完整预测窗口
```

每个目标变量单独写入一个子文件夹。子文件夹中的样本只包含该变量自身的历史特征和通用日历特征；如果需要多变量输入，可以在训练阶段读取多个变量 CSV 后自行合并。未来多天的目标字段按宽表展开，例如 `第1天_目标谷值` 到 `第14天_目标谷值`。

核心目标字段：

```text
目标峰值
目标峰值残差
目标峰值小时
目标谷值
目标谷值残差
目标谷值小时
```

字段名输出为中文；代码内部保留英文字段到中文字段的映射，见 `src/wave_dataset/config.py`。

## 切分规则

按第一个完整自然日起算：

```text
训练区间 = 前 12 个月
验证区间 = 后续 4 个月
测试区间 = 后续 4 个月
```

样本的预测窗口必须完整落在某个区间内，否则直接丢弃。输入窗口可以使用区间之前的历史数据，因为它只包含预测起点之前的信息。

## 拐点标签

每个变量单独输出一个小时级拐点文件，字段为：

```text
日期
原始值
平滑值
拐点类型
拐点显著性
拐点阈值
有效标记
```

拐点类型：

```text
1  = 峰型拐点
-1 = 谷型拐点
0  = 非拐点
```

检测方法：

```text
scipy.signal.savgol_filter 平滑
scipy.signal.find_peaks 检测峰谷候选
滚动 MAD 自适应阈值
回贴到原始序列附近真实局部极值
```

## 数据可视化

可视化会读取指定数据集文件夹中的第一个训练样本，为每个变量生成两张图：

1. 原始序列 + 已计算好的每日波峰 + 已计算好的每日波谷。
2. 原始序列 + 已生成拐点标签中的峰型拐点 + 谷型拐点。

点数由数据集文件夹名决定。例如 `ETTH1_pred14_seq4` 表示历史 4 天、预测 14 天，总点数为 `(4 + 14) * 24 = 432`。图中会在第 `4 * 24 = 96` 个点处画一条更明显的竖直虚线，用于区分历史输入窗口和预测窗口。

默认不传变量名时，会绘制全部变量：

```text
HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
```

运行：

```bash
cd /home/wmb/test/波峰波谷预测
PYTHONPATH='src' /home/wmb/anaconda3/envs/Ada-MSHyper/bin/python -m wave_dataset.plot_sample \
  --input ETTh1.csv \
  --data-root '数据集' \
  --dataset-name ETTH1_pred14_seq4 \
  --output-root '数据集可视化'
```

输出：

```text
数据集可视化/ETTH1_pred14_seq4/HUFL_S000001_波峰波谷.png
数据集可视化/ETTH1_pred14_seq4/HUFL_S000001_拐点.png
...
数据集可视化/ETTH1_pred14_seq4/OT_S000001_波峰波谷.png
数据集可视化/ETTH1_pred14_seq4/OT_S000001_拐点.png
```

只绘制某个变量：

```bash
cd /home/wmb/test/波峰波谷预测
PYTHONPATH='src' /home/wmb/anaconda3/envs/Ada-MSHyper/bin/python -m wave_dataset.plot_sample \
  --dataset-name ETTH1_pred14_seq4 \
  --target-col OT
```

也可以在 Python 中直接批量调用：

```python
from wave_dataset.peak_dataset import parse_hourly_csv
from wave_dataset.visualization import plot_dataset_visualizations

records = parse_hourly_csv("ETTh1.csv")
plot_dataset_visualizations(records, dataset_name="ETTH1_pred14_seq4")
```

## 测试

```bash
cd /home/wmb/test/波峰波谷预测
PYTHONPATH='src' /home/wmb/anaconda3/envs/Ada-MSHyper/bin/python -m unittest discover -s 'tests' -v
```
