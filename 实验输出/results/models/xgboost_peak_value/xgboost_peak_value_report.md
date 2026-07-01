# XGBoost peak_value 残差回归报告

## 是否超过最佳规则基线

- 验证：超过最佳规则基线，XGBoost MAE=1.65371，规则基线 MAE=1.7206，improvement=3.89%
- 测试：未超过最佳规则基线，XGBoost MAE=1.43157，规则基线 MAE=1.25578，improvement=-14.00%

## 全局指标

- 测试：MAE=1.43157, RMSE=2.1799, sMAPE=0.253339
- 验证：MAE=1.65371, RMSE=2.35025, sMAPE=0.343206

## 模型摘要

- HUFL：train=4872, val=1540, test=1498, best_iteration=13
- HULL：train=4872, val=1540, test=1498, best_iteration=6
- LUFL：train=4872, val=1540, test=1498, best_iteration=12
- LULL：train=4872, val=1540, test=1498, best_iteration=0
- MUFL：train=4872, val=1540, test=1498, best_iteration=4
- MULL：train=4872, val=1540, test=1498, best_iteration=11
- OT：train=4872, val=1540, test=1498, best_iteration=139

## 输出文件

- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_predictions.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_val_predictions.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_test_predictions.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_metrics.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_vs_best_rule_baseline.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_feature_importance.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_model_summary.csv`
- `实验输出/results/models/xgboost_peak_value/xgboost_peak_value_report.md`
- `实验输出/results/models/xgboost_peak_value/models/HUFL_peak_value.json`
- `实验输出/results/models/xgboost_peak_value/models/HULL_peak_value.json`
- `实验输出/results/models/xgboost_peak_value/models/LUFL_peak_value.json`
- `实验输出/results/models/xgboost_peak_value/models/LULL_peak_value.json`
- `实验输出/results/models/xgboost_peak_value/models/MUFL_peak_value.json`
- `实验输出/results/models/xgboost_peak_value/models/MULL_peak_value.json`
- `实验输出/results/models/xgboost_peak_value/models/OT_peak_value.json`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HUFL/HUFL_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HUFL/HUFL_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HUFL/HUFL_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HUFL/HUFL_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HUFL/HUFL_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HUFL/HUFL_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HULL/HULL_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HULL/HULL_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HULL/HULL_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HULL/HULL_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HULL/HULL_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/HULL/HULL_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MUFL/MUFL_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MUFL/MUFL_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MUFL/MUFL_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MUFL/MUFL_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MUFL/MUFL_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MUFL/MUFL_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MULL/MULL_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MULL/MULL_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MULL/MULL_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MULL/MULL_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MULL/MULL_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/MULL/MULL_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LUFL/LUFL_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LUFL/LUFL_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LUFL/LUFL_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LUFL/LUFL_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LUFL/LUFL_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LUFL/LUFL_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LULL/LULL_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LULL/LULL_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LULL/LULL_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LULL/LULL_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LULL/LULL_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/LULL/LULL_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/OT/OT_S000349_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/OT/OT_S000370_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/OT/OT_S000392_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/OT/OT_S000414_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/OT/OT_S000436_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_验证/OT/OT_S000458_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HUFL/HUFL_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HUFL/HUFL_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HUFL/HUFL_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HUFL/HUFL_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HUFL/HUFL_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HUFL/HUFL_S000565_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HULL/HULL_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HULL/HULL_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HULL/HULL_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HULL/HULL_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HULL/HULL_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/HULL/HULL_S000565_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MUFL/MUFL_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MUFL/MUFL_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MUFL/MUFL_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MUFL/MUFL_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MUFL/MUFL_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MUFL/MUFL_S000565_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MULL/MULL_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MULL/MULL_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MULL/MULL_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MULL/MULL_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MULL/MULL_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/MULL/MULL_S000565_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LUFL/LUFL_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LUFL/LUFL_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LUFL/LUFL_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LUFL/LUFL_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LUFL/LUFL_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LUFL/LUFL_S000565_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LULL/LULL_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LULL/LULL_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LULL/LULL_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LULL/LULL_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LULL/LULL_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/LULL/LULL_S000565_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/OT/OT_S000459_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/OT/OT_S000480_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/OT/OT_S000501_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/OT/OT_S000522_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/OT/OT_S000543_XGBoost波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/XGBoost_波峰残差预测_测试/OT/OT_S000565_XGBoost波峰残差预测.png`
