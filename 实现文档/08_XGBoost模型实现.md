# XGBoost 模型实现

## 1. 目标

使用 XGBoost 与 LightGBM 做公平对照。

两者必须使用：

```text
同一份训练数据
同一份特征
同一份规则基线
同一套切分
同一套评估指标
```

这样才能判断模型差异，而不是数据或特征差异。

## 2. 模型拆分

与 LightGBM 相同：

```text
7 个 peak_value residual 回归模型
7 个 peak_hour 分类模型
```

总计 14 个 XGBoost 模型。

## 3. 回归模型

XGBoost 回归模型预测：

```text
target_peak_residual
```

推荐初始参数：

```text
objective = reg:squarederror
eval_metric = mae
eta = 0.03
max_depth = 4
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
lambda = 1.0
alpha = 0.0
n_estimators = 2000
early_stopping_rounds = 100
```

如果过拟合，优先调整：

```text
max_depth 降低
min_child_weight 增大
lambda 增大
subsample 降低
```

## 4. 分类模型

XGBoost 分类模型预测：

```text
target_peak_hour ∈ {0,1,...,23}
```

推荐初始参数：

```text
objective = multi:softprob
num_class = 24
eval_metric = mlogloss
eta = 0.03
max_depth = 4
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
lambda = 1.0
n_estimators = 2000
early_stopping_rounds = 100
```

预测时取概率最大的小时：

```text
pred_peak_hour = argmax(pred_probability)
```

## 5. 与 LightGBM 的公平比较

比较时固定：

| 项目 | 要求 |
| --- | --- |
| 数据文件 | 完全一致 |
| 特征列 | 完全一致 |
| 训练验证测试切分 | 完全一致 |
| baseline_peak | 完全一致 |
| 残差目标 | 完全一致 |
| 评价指标 | 完全一致 |

比较结果应按以下层级记录：

```text
全局平均
按变量
按 horizon
按 peak_value 与 peak_hour 分开
```

## 6. 输出文件

建议输出：

```text
模型/XGBoost/HUFL_peak_value.json
模型/XGBoost/HUFL_peak_hour.json
...
结果/XGBoost/预测结果_seq96_pred336.csv
结果/XGBoost/评估结果_seq96_pred336.csv
结果/XGBoost/特征重要性_seq96_pred336.csv
```

## 7. 速度与资源注意事项

XGBoost 可能比 LightGBM 慢。若训练较慢，可以：

```text
减少 n_estimators
调大 early_stopping 依赖验证集自动停止
降低 max_depth
先只跑 OT 或一个变量做冒烟测试
再扩展到 7 个变量
```

## 8. 常见问题

| 问题 | 可能原因 | 处理 |
| --- | --- | --- |
| 训练太慢 | 树太多或深度太大 | 降低 `max_depth` 和 `n_estimators` |
| 验证集不提升 | 特征弱或学习率不合适 | 检查基线和残差 |
| 比 LightGBM 差很多 | 参数不适配 | 先调整深度和正则 |
| 分类模型偏向常见小时 | 类别不平衡 | 查看 peak_hour 分布 |

