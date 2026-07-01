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
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HUFL/HUFL_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HUFL/HUFL_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HUFL/HUFL_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HUFL/HUFL_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HUFL/HUFL_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HUFL/HUFL_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HULL/HULL_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HULL/HULL_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HULL/HULL_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HULL/HULL_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HULL/HULL_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/HULL/HULL_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MUFL/MUFL_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MUFL/MUFL_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MUFL/MUFL_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MUFL/MUFL_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MUFL/MUFL_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MUFL/MUFL_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MULL/MULL_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MULL/MULL_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MULL/MULL_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MULL/MULL_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MULL/MULL_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/MULL/MULL_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LUFL/LUFL_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LUFL/LUFL_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LUFL/LUFL_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LUFL/LUFL_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LUFL/LUFL_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LUFL/LUFL_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LULL/LULL_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LULL/LULL_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LULL/LULL_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LULL/LULL_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LULL/LULL_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/LULL/LULL_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/OT/OT_S000349_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/OT/OT_S000370_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/OT/OT_S000392_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/OT/OT_S000414_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/OT/OT_S000436_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_验证/OT/OT_S000458_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HUFL/HUFL_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HUFL/HUFL_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HUFL/HUFL_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HUFL/HUFL_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HUFL/HUFL_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HUFL/HUFL_S000565_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HULL/HULL_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HULL/HULL_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HULL/HULL_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HULL/HULL_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HULL/HULL_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/HULL/HULL_S000565_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MUFL/MUFL_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MUFL/MUFL_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MUFL/MUFL_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MUFL/MUFL_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MUFL/MUFL_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MUFL/MUFL_S000565_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MULL/MULL_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MULL/MULL_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MULL/MULL_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MULL/MULL_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MULL/MULL_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/MULL/MULL_S000565_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LUFL/LUFL_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LUFL/LUFL_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LUFL/LUFL_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LUFL/LUFL_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LUFL/LUFL_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LUFL/LUFL_S000565_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LULL/LULL_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LULL/LULL_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LULL/LULL_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LULL/LULL_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LULL/LULL_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/LULL/LULL_S000565_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/OT/OT_S000459_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/OT/OT_S000480_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/OT/OT_S000501_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/OT/OT_S000522_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/OT/OT_S000543_LightGBM波峰残差预测.png`
- `数据集可视化/ETTH1_pred14_seq4/LightGBM_波峰残差预测_测试/OT/OT_S000565_LightGBM波峰残差预测.png`
