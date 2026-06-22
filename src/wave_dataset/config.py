from __future__ import annotations

TARGET_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

DATE_COLUMN = "date"

SEQ_DAYS = 4
PRED_DAYS = 14

COLUMN_EN_TO_ZH = {
    "sample_id": "样本ID",
    "forecast_start_date": "预测起点日期",
    "target_col": "目标变量",
    "horizon": "预测天数",
    "target_date": "目标日期",
    "baseline_peak": "基线峰值",
    "target_peak_value": "目标峰值",
    "target_peak_residual": "目标峰值残差",
    "target_peak_hour": "目标峰值小时",
    "target_trough_value": "目标谷值",
    "target_trough_residual": "目标谷值残差",
    "target_trough_hour": "目标谷值小时",
    "split": "数据集划分",
    "date": "日期",
    "value": "原始值",
    "smooth_value": "平滑值",
    "turning_type": "拐点类型",
    "turning_prominence": "拐点显著性",
    "turning_threshold": "拐点阈值",
    "valid_mask": "有效标记",
}

PEAK_BASE_COLUMNS_ZH = [
    COLUMN_EN_TO_ZH["sample_id"],
    COLUMN_EN_TO_ZH["forecast_start_date"],
    COLUMN_EN_TO_ZH["target_col"],
    COLUMN_EN_TO_ZH["horizon"],
    COLUMN_EN_TO_ZH["target_date"],
    COLUMN_EN_TO_ZH["baseline_peak"],
    COLUMN_EN_TO_ZH["target_peak_value"],
    COLUMN_EN_TO_ZH["target_peak_residual"],
    COLUMN_EN_TO_ZH["target_peak_hour"],
    COLUMN_EN_TO_ZH["target_trough_value"],
    COLUMN_EN_TO_ZH["target_trough_residual"],
    COLUMN_EN_TO_ZH["target_trough_hour"],
    COLUMN_EN_TO_ZH["split"],
]

TURNING_COLUMNS_ZH = [
    COLUMN_EN_TO_ZH["date"],
    COLUMN_EN_TO_ZH["value"],
    COLUMN_EN_TO_ZH["smooth_value"],
    COLUMN_EN_TO_ZH["turning_type"],
    COLUMN_EN_TO_ZH["turning_prominence"],
    COLUMN_EN_TO_ZH["turning_threshold"],
    COLUMN_EN_TO_ZH["valid_mask"],
]
