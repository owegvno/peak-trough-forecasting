# LightGBM peak_value 残差回归报告

## 是否超过最佳规则基线

- 验证：超过最佳规则基线，LightGBM MAE=1.65268，规则基线 MAE=1.7206，improvement=3.95%
- 测试：未超过最佳规则基线，LightGBM MAE=1.37184，规则基线 MAE=1.25578，improvement=-9.24%

## 全局指标

- 测试：MAE=1.37184, RMSE=2.0536, sMAPE=0.248489
- 验证：MAE=1.65268, RMSE=2.34891, sMAPE=0.343508

## 模型摘要

- HUFL：train=4872, val=1540, test=1498, best_iteration=27
- HULL：train=4872, val=1540, test=1498, best_iteration=11
- LUFL：train=4872, val=1540, test=1498, best_iteration=8
- LULL：train=4872, val=1540, test=1498, best_iteration=1
- MUFL：train=4872, val=1540, test=1498, best_iteration=8
- MULL：train=4872, val=1540, test=1498, best_iteration=4
- OT：train=4872, val=1540, test=1498, best_iteration=68

## 输出文件

- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_predictions.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_val_predictions.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_test_predictions.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_metrics.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_vs_best_rule_baseline.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_feature_importance.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_model_summary.csv`
- `实验输出/results/models/lightgbm_peak_value/lightgbm_peak_value_report.md`
- `实验输出/results/models/lightgbm_peak_value/models/HUFL_peak_value.txt`
- `实验输出/results/models/lightgbm_peak_value/models/HULL_peak_value.txt`
- `实验输出/results/models/lightgbm_peak_value/models/LUFL_peak_value.txt`
- `实验输出/results/models/lightgbm_peak_value/models/LULL_peak_value.txt`
- `实验输出/results/models/lightgbm_peak_value/models/MUFL_peak_value.txt`
- `实验输出/results/models/lightgbm_peak_value/models/MULL_peak_value.txt`
- `实验输出/results/models/lightgbm_peak_value/models/OT_peak_value.txt`
