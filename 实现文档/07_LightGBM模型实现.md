# LightGBM 模型实现

## 1. 目标

使用 LightGBM 完成第一版峰值预测实验。

预测目标：

```text
peak_value：预测残差，回归任务
peak_hour：预测 0-23 小时，分类任务
```

第一版不加入小波特征。

## 2. 模型拆分

目标列：

```text
HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
```

每列训练两个模型：

```text
1 个 peak_value residual 回归模型
1 个 peak_hour 分类模型
```

总计：

```text
7 个回归模型 + 7 个分类模型 = 14 个 LightGBM 模型
```

`horizon=1..14` 作为输入特征，而不是为每个 horizon 单独训练模型。

## 3. 输入数据

使用峰值预测样本长表：

```text
[当前实现] 宽表按变量保存于 数据集/ETTH1_pred14_seq4/{变量}/；训练建议优先使用长表 数据集/ETTH1_pred14_seq4/长表/峰谷预测长表_seq96_pred336_全部变量.csv
```

每次训练时按 `target_col` 过滤：

```text
target_col == 当前变量
```

回归目标：

```text
target_peak_residual
```

分类目标：

```text
target_peak_hour
```

## 4. 回归模型

LightGBM 回归模型用于预测：

```text
future_peak_value - baseline_peak
```

推荐初始参数：

```text
objective = regression
metric = l1
learning_rate = 0.03
num_leaves = 31
max_depth = -1
min_data_in_leaf = 20
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 1
lambda_l1 = 0.0
lambda_l2 = 1.0
num_boost_round = 2000
early_stopping_rounds = 100
```

主评估指标使用验证集 MAE。

## 5. 分类模型

LightGBM 分类模型用于预测：

```text
target_peak_hour ∈ {0,1,...,23}
```

推荐初始参数：

```text
objective = multiclass
num_class = 24
metric = multi_logloss
learning_rate = 0.03
num_leaves = 31
min_data_in_leaf = 20
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 1
lambda_l2 = 1.0
num_boost_round = 2000
early_stopping_rounds = 100
```

训练时仍记录：

```text
平均小时误差
±1h 命中率
±2h 命中率
```

这些指标比 `multi_logloss` 更贴近业务目标。

## 6. 预测还原

回归模型输出的是残差：

```text
pred_residual
```

最终峰值预测：

```text
pred_peak_value = baseline_peak + pred_residual
```

最终峰值小时预测：

```text
pred_peak_hour = 分类概率最大的小时
```

## 7. 输出文件

建议输出：

```text
[待确认] 实验输出/models/lightgbm/HUFL_peak_value.txt
[待确认] 实验输出/models/lightgbm/HUFL_peak_hour.txt
...
[待确认] 实验输出/results/lightgbm/预测结果_seq96_pred336.csv
[待确认] 实验输出/results/lightgbm/评估结果_seq96_pred336.csv
[待确认] 实验输出/results/lightgbm/特征重要性_seq96_pred336.csv
```

预测结果表建议字段：

```text
sample_id
forecast_start_date
target_col
horizon
target_date
baseline_peak
target_peak_value
pred_peak_value
target_peak_hour
pred_peak_hour
split
model_name
```

## 8. 特征重要性

每个回归模型和分类模型都应输出特征重要性。

重点查看：

```text
horizon 是否重要
目标变量自身历史峰值是否重要
其他变量历史特征是否有贡献
日历特征是否有贡献
```

如果模型主要依赖 `horizon` 和少数均值特征，说明可能没有学到复杂规律。

## 9. 常见问题

| 问题 | 可能原因 | 处理 |
| --- | --- | --- |
| 训练集好、测试集差 | 过拟合 | 增大正则、减少叶子数、检查泄露 |
| 不如基线 | 特征无效或任务难 | 先检查残差相关性 |
| peak_hour 准确率低 | 峰值时间随机性强 | 看峰值小时分布 |
| 预测峰值过平滑 | 模型退化到均值 | 检查分布和残差目标 |

## 10. 当前实现路径待确认

[待确认] 当前项目已有数据集代码位于 `src/wave_dataset/`，但 LightGBM 训练代码尚未确认落位。建议后续新增模型相关代码到 `src/wave_experiments/`，输出结果到 `实验输出/`。如果你希望继续使用 `代码实现/` 目录，需要先统一调整所有执行提示词和文档路径。
