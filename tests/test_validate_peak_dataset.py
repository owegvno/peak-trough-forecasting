from pathlib import Path

import pandas as pd

from wave_experiments.datasets.validate_peak_dataset import (
    EXPECTED_HORIZONS,
    EXPECTED_TARGET_VARIABLES,
    validate_dataset,
    write_markdown_report,
    write_summary_csv,
)


def _valid_frame() -> pd.DataFrame:
    rows = []
    split_dates = {
        "训练": "2016-01-01",
        "验证": "2017-01-01",
        "测试": "2018-01-01",
    }
    for split, date in split_dates.items():
        for variable in EXPECTED_TARGET_VARIABLES:
            for horizon in EXPECTED_HORIZONS:
                rows.append(
                    {
                        "样本ID": f"{split}-{variable}-{horizon}",
                        "预测起点日期": date,
                        "目标变量": variable,
                        "预测天数": horizon,
                        "数据集划分": split,
                        "目标峰值": float(horizon),
                        "目标峰值小时": horizon % 24,
                        "目标谷值": float(-horizon),
                        "目标谷值小时": (horizon + 1) % 24,
                        "HUFL_过去96小时_均值": 1.0,
                        "OT_过去第1天_峰值小时": 6.0,
                    }
                )
    return pd.DataFrame(rows)


def test_validate_dataset_accepts_complete_dataset(tmp_path: Path) -> None:
    df = _valid_frame()

    result = validate_dataset(df)
    report_path = tmp_path / "report.md"
    summary_path = tmp_path / "summary.csv"
    write_markdown_report(result, report_path)
    write_summary_csv(result, summary_path)

    assert result.allow_step3 is True
    assert result.horizon_coverage.ok is True
    assert result.target_variable_coverage.ok is True
    assert result.peak_hour_range.ok is True
    assert result.leakage_check.ok is True
    report = report_path.read_text(encoding="utf-8")
    assert "预测天数覆盖 1 到 14" in report
    assert "覆盖 7 个目标变量" in report
    assert "`目标峰值小时` 全部在 0 到 23" in report
    assert "未发现明显泄露" in report
    assert "允许进入第 3 步：是" in report
    summary = pd.read_csv(summary_path)
    assert {"check", "passed", "details"}.issubset(summary.columns)


def test_validate_dataset_flags_leakage_and_time_overlap() -> None:
    df = _valid_frame()
    df["未来真实目标峰值"] = 1.0
    df.loc[df["数据集划分"] == "验证", "预测起点日期"] = "2015-12-31"

    result = validate_dataset(df)

    assert result.allow_step3 is False
    assert result.leakage_check.ok is False
    assert "未来真实目标峰值" in result.leakage_check.details
    assert result.time_order_check.ok is False


def test_validate_dataset_rejects_shared_boundary_date_between_splits() -> None:
    df = _valid_frame()
    df.loc[df["数据集划分"] == "验证", "预测起点日期"] = "2016-01-01"

    result = validate_dataset(df)

    assert result.time_order_check.ok is False
    assert result.allow_step3 is False
