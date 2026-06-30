from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from wave_dataset.config import COLUMN_EN_TO_ZH, TARGET_COLUMNS
from wave_dataset.peak_dataset import (
    assign_split,
    build_daily_peak_rows,
    build_peak_dataset,
    build_peak_sample_rows,
    complete_day_records,
    parse_hourly_csv,
    resolve_output_dir,
)
from wave_dataset.turning_points import detect_turning_points
from wave_dataset.visualization import (
    evenly_sample_sample_ids,
    dataset_window_from_name,
    merge_baseline_peak_predictions,
    plot_peak_hour_prediction_batch,
    plot_peak_prediction_batch,
    plot_peak_prediction_rows,
    plot_peak_value_prediction_batch,
    plot_selected_best_baseline_prediction_batch,
    plot_dataset_visualizations,
    plot_sample_peaks_troughs,
    plot_sample_turning_points,
    read_first_sample,
    sample_csv_for_dataset,
)


def make_records(start: datetime, hours: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for i in range(hours):
        current = start + timedelta(hours=i)
        row = {"date": f"{current.year}/{current.month}/{current.day} {current:%H:%M}"}
        hour = current.hour
        for col_index, col in enumerate(TARGET_COLUMNS):
            row[col] = str(hour + col_index * 0.1)
        records.append(row)
    return records


def parse_hourly_csv_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for raw in rows:
        row: dict[str, object] = {"date": datetime.strptime(raw["date"], "%Y/%m/%d %H:%M")}
        for col in TARGET_COLUMNS:
            row[col] = float(raw[col])
        parsed.append(row)
    return parsed


class RecordingFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path) -> None:
        Path(path).write_bytes(b"fake-png")


class RecordingAxes:
    def __init__(self) -> None:
        self.plots: list[dict[str, object]] = []
        self.scatter_summaries: list[dict[str, object]] = []
        self.vertical_lines: list[int] = []
        self.xlim: tuple[int, int] | None = None

    @staticmethod
    def _list(values: object) -> list[float]:
        if hasattr(values, "tolist"):
            raw_values = values.tolist()
        else:
            raw_values = list(values)  # type: ignore[arg-type]
        output: list[float] = []
        for value in raw_values:
            current = float(value)
            output.append(int(current) if current.is_integer() else current)
        return output

    def plot(self, x: object, y: object, **kwargs: object) -> None:
        self.plots.append({"x": self._list(x), "y": self._list(y), "label": kwargs.get("label")})

    def scatter(self, x: object, y: object, **kwargs: object) -> None:
        self.scatter_summaries.append(
            {"label": kwargs.get("label"), "x": self._list(x), "y": self._list(y)}
        )

    def axvline(self, x: int, **kwargs: object) -> None:
        self.vertical_lines.append(int(x))

    def set_title(self, title: str) -> None:
        return None

    def set_xlabel(self, label: str) -> None:
        return None

    def set_ylabel(self, label: str) -> None:
        return None

    def set_xlim(self, left: int, right: int) -> None:
        self.xlim = (left, right)

    def set_xticks(self, ticks: list[int]) -> None:
        return None

    def grid(self, *args: object, **kwargs: object) -> None:
        return None

    def legend(self, *args: object, **kwargs: object) -> None:
        return None


def write_wide_rows(dataset_dir: Path, target_col: str, rows: list[dict[str, object]], pred_days: int = 2) -> None:
    target_dir = dataset_dir / target_col
    target_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["样本ID", "预测起点日期", "目标变量", "预测天数", "基线峰值", "数据集划分"]
    for day in range(1, pred_days + 1):
        fieldnames.extend(
            [
                f"第{day}天_目标峰值",
                f"第{day}天_目标峰值残差",
                f"第{day}天_目标峰值小时",
                f"第{day}天_目标谷值",
                f"第{day}天_目标谷值残差",
                f"第{day}天_目标谷值小时",
                f"第{day}天_日历_星期",
                f"第{day}天_日历_月份",
                f"第{day}天_日历_年内日序",
                f"第{day}天_日历_是否周末",
            ]
        )
    fieldnames.append(f"{target_col}_过去96小时_均值")
    path = target_dir / f"峰谷预测样本_seq48_pred48_{target_col}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DatasetGenerationTests(unittest.TestCase):
    def test_chinese_column_mapping_keeps_internal_and_output_names(self) -> None:
        self.assertEqual(COLUMN_EN_TO_ZH["sample_id"], "样本ID")
        self.assertEqual(COLUMN_EN_TO_ZH["forecast_start_date"], "预测起点日期")
        self.assertEqual(COLUMN_EN_TO_ZH["target_peak_residual"], "目标峰值残差")

    def test_complete_day_records_discards_half_day(self) -> None:
        records = make_records(datetime(2020, 1, 1), 60)
        complete_days, discarded_days = complete_day_records(records)

        self.assertEqual(list(complete_days), ["2020-01-01", "2020-01-02"])
        self.assertEqual(discarded_days, ["2020-01-03"])

    def test_daily_peak_uses_first_hour_when_max_repeats(self) -> None:
        records = make_records(datetime(2020, 1, 1), 24)
        records[10]["OT"] = "99"
        records[15]["OT"] = "99"
        records[3]["OT"] = "-5"
        complete_days, _ = complete_day_records(records)

        daily_rows = build_daily_peak_rows(complete_days)

        self.assertEqual(daily_rows[0]["OT_peak_value"], 99.0)
        self.assertEqual(daily_rows[0]["OT_peak_hour"], 10)
        self.assertEqual(daily_rows[0]["OT_trough_value"], -5.0)
        self.assertEqual(daily_rows[0]["OT_trough_hour"], 3)

    def test_assign_split_drops_cross_boundary_windows(self) -> None:
        train_start = datetime(2020, 1, 1).date()
        train_end = datetime(2020, 1, 6).date()
        val_end = datetime(2020, 1, 10).date()
        test_end = datetime(2020, 1, 14).date()

        self.assertEqual(assign_split(datetime(2020, 1, 1).date(), 2, train_start, train_end, val_end, test_end), "训练")
        self.assertEqual(assign_split(datetime(2020, 1, 5).date(), 2, train_start, train_end, val_end, test_end), "")
        self.assertEqual(assign_split(datetime(2020, 1, 6).date(), 2, train_start, train_end, val_end, test_end), "验证")
        self.assertEqual(assign_split(datetime(2020, 1, 10).date(), 2, train_start, train_end, val_end, test_end), "测试")

    def test_build_peak_sample_rows_requires_enough_complete_days(self) -> None:
        records = make_records(datetime(2020, 1, 1), 17 * 24)
        complete_days, _ = complete_day_records(records)
        daily_rows = build_daily_peak_rows(complete_days)

        samples = build_peak_sample_rows(complete_days, daily_rows, seq_days=4, pred_days=14)

        self.assertEqual(samples, [])

    def test_build_peak_sample_rows_outputs_chinese_long_table(self) -> None:
        records = make_records(datetime(2020, 1, 1), 18 * 24)
        complete_days, _ = complete_day_records(records)
        daily_rows = build_daily_peak_rows(complete_days)

        samples = build_peak_sample_rows(complete_days, daily_rows, seq_days=4, pred_days=14)

        self.assertEqual(len(samples), len(TARGET_COLUMNS) * 14)
        first = samples[0]
        self.assertIn("样本ID", first)
        self.assertEqual(first["预测起点日期"], "2020-01-05")
        self.assertEqual(first["目标变量"], "HUFL")
        self.assertEqual(first["预测天数"], 1)
        self.assertEqual(first["目标日期"], "2020-01-05")
        self.assertEqual(first["目标峰值小时"], 23)
        self.assertIn("目标谷值", first)
        self.assertIn("目标谷值小时", first)

    def test_build_peak_sample_rows_drops_cross_boundary_samples(self) -> None:
        records = make_records(datetime(2020, 1, 1), 16 * 24)
        complete_days, _ = complete_day_records(records)
        daily_rows = build_daily_peak_rows(complete_days)
        split_boundaries = (
            datetime(2020, 1, 1).date(),
            datetime(2020, 1, 8).date(),
            datetime(2020, 1, 13).date(),
            datetime(2020, 1, 18).date(),
        )

        samples = build_peak_sample_rows(
            complete_days,
            daily_rows,
            seq_days=4,
            pred_days=3,
            split_boundaries=split_boundaries,
        )

        self.assertTrue(samples)
        self.assertNotIn("", {row["数据集划分"] for row in samples})

    def test_parse_hourly_csv_accepts_project_date_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 2):
                    writer.writerow(row)

            rows = parse_hourly_csv(path)

        self.assertEqual(rows[0]["date"].year, 2020)
        self.assertEqual(rows[1]["date"].hour, 1)

    def test_resolve_output_dir_adds_dataset_and_window_parameters(self) -> None:
        output = resolve_output_dir("数据集", "ETTh1.csv", seq_days=5, pred_days=8)

        self.assertEqual(output, Path("数据集") / "ETTH1_pred8_seq5")

    def test_build_peak_dataset_writes_one_single_variable_folder_per_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ETTh1.csv"
            output_base = tmp_path / "数据集"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 8 * 24):
                    writer.writerow(row)

            result = build_peak_dataset(input_path, output_base, seq_days=2, pred_days=2)

            output_dir = tmp_path / "数据集" / "ETTH1_pred2_seq2"
            self.assertEqual(result["output_dir"], output_dir)
            self.assertTrue((output_dir / "完整自然日清单.csv").exists())
            self.assertTrue((output_dir / "日周期峰值标签.csv").exists())
            self.assertFalse((output_dir / "峰谷预测样本_seq48_pred48.csv").exists())
            for col in TARGET_COLUMNS:
                self.assertTrue((output_dir / col).is_dir())

            hufl_path = output_dir / "HUFL" / "峰谷预测样本_seq48_pred48_HUFL.csv"
            self.assertTrue(hufl_path.exists())
            with hufl_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                rows = list(reader)

            self.assertTrue(rows)
            self.assertEqual({row["目标变量"] for row in rows}, {"HUFL"})
            self.assertEqual(len(rows), result["kept_anchor_count"])
            self.assertEqual(rows[0]["预测天数"], "2")
            self.assertIn("第1天_目标谷值", fieldnames)
            self.assertIn("第2天_目标谷值", fieldnames)
            self.assertNotIn("目标谷值", fieldnames)
            self.assertTrue(any(name.startswith("HUFL_") for name in fieldnames))
            self.assertFalse(any(name.startswith("HULL_") for name in fieldnames))
            self.assertFalse(any(name.startswith("OT_") for name in fieldnames))
            self.assertNotIn("日历_预测天数", fieldnames)
            self.assertNotIn("第1天_目标日期", fieldnames)
            self.assertNotIn("HUFL_过去第4天_峰值", fieldnames)
            self.assertNotIn("HUFL_过去第4天_谷值", fieldnames)
            self.assertNotIn("HUFL_过去96小时_末值", fieldnames)
            self.assertNotIn("HUFL_历史峰值_最大值4天", fieldnames)
            self.assertNotIn("HUFL_历史峰值_最近值", fieldnames)
            self.assertNotIn("HUFL_历史峰值_加权均值4天", fieldnames)
            self.assertNotIn("HUFL_历史峰值小时_最近值", fieldnames)
            self.assertNotIn("HUFL_峰谷差_最近值", fieldnames)
            self.assertNotIn("第8天_日历_星期", fieldnames)

            seen_values: dict[tuple[str, ...], str] = {}
            duplicates: list[tuple[str, str]] = []
            for name in fieldnames:
                if "目标" in name or name in {"基线峰值"}:
                    continue
                values = tuple(row[name] for row in rows)
                if values in seen_values:
                    duplicates.append((seen_values[values], name))
                else:
                    seen_values[values] = name
            self.assertEqual(duplicates, [])

    def test_detect_turning_points_labels_peak_and_trough(self) -> None:
        values = [0.0, 1.0, 3.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0]
        rows = detect_turning_points(values, smooth_window=3, mad_window=3, prominence_multiplier=0.0)
        types = [row.turning_type for row in rows]

        self.assertIn(1, types)
        self.assertIn(-1, types)
        self.assertTrue(set(types).issubset({-1, 0, 1}))

    def test_visualization_functions_write_png_files(self) -> None:
        records = parse_hourly_csv(Path("ETTh1.csv"))
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            sample_csv = output_dir / "sample.csv"
            with sample_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["样本ID", "预测起点日期", "目标变量"])
                writer.writeheader()
                writer.writerow({"样本ID": "S000001", "预测起点日期": "2016-07-05", "目标变量": "HUFL"})
                writer.writerow({"样本ID": "S000001", "预测起点日期": "2016-07-05", "目标变量": "OT"})

            self.assertEqual(read_first_sample(sample_csv)["目标变量"], "OT")
            peak_path = plot_sample_peaks_troughs(records, sample_csv=sample_csv, output_dir=output_dir, seq_len=96)
            turning_path = plot_sample_turning_points(records, sample_csv=sample_csv, output_dir=output_dir, seq_len=96)

            self.assertTrue(peak_path.exists())
            self.assertTrue(turning_path.exists())
            self.assertIn("OT", peak_path.name)
            self.assertIn("OT", turning_path.name)
            self.assertGreater(peak_path.stat().st_size, 1000)
            self.assertGreater(turning_path.stat().st_size, 1000)

    def test_dataset_window_from_name_parses_seq_and_pred_days(self) -> None:
        window = dataset_window_from_name("ETTH1_pred14_seq4")

        self.assertEqual(window.seq_days, 4)
        self.assertEqual(window.pred_days, 14)
        self.assertEqual(window.seq_len, 96)
        self.assertEqual(window.pred_len, 336)
        self.assertEqual(window.total_len, 432)

    def test_sample_csv_for_dataset_uses_dataset_folder_and_column(self) -> None:
        path = sample_csv_for_dataset("数据集", "ETTH1_pred14_seq4", "HUFL", split="训练")

        self.assertEqual(
            path,
            Path("数据集")
            / "ETTH1_pred14_seq4"
            / "HUFL"
            / "峰谷预测样本_seq96_pred336_HUFL_训练.csv",
        )

    def test_plot_dataset_visualizations_defaults_to_all_columns(self) -> None:
        records = parse_hourly_csv(Path("ETTh1.csv"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "数据集" / "ETTH1_pred2_seq1"
            label_dir = dataset_dir / "拐点标签"
            label_dir.mkdir(parents=True)
            forecast_start = datetime(2016, 7, 2)

            for col in TARGET_COLUMNS:
                col_dir = dataset_dir / col
                col_dir.mkdir()
                sample_csv = col_dir / f"峰谷预测样本_seq24_pred48_{col}_训练.csv"
                with sample_csv.open("w", encoding="utf-8", newline="") as handle:
                    fieldnames = [
                        "样本ID",
                        "预测起点日期",
                        "目标变量",
                        f"{col}_过去第1天_最大值",
                        f"{col}_过去第1天_峰值小时",
                        f"{col}_过去第1天_最小值",
                        f"{col}_过去第1天_谷值小时",
                    ]
                    for day in range(1, 3):
                        fieldnames.extend(
                            [
                                f"第{day}天_目标峰值",
                                f"第{day}天_目标峰值小时",
                                f"第{day}天_目标谷值",
                                f"第{day}天_目标谷值小时",
                            ]
                        )
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "样本ID": "S000001",
                            "预测起点日期": forecast_start.strftime("%Y-%m-%d"),
                            "目标变量": col,
                            f"{col}_过去第1天_最大值": "10",
                            f"{col}_过去第1天_峰值小时": "5",
                            f"{col}_过去第1天_最小值": "1",
                            f"{col}_过去第1天_谷值小时": "2",
                            "第1天_目标峰值": "11",
                            "第1天_目标峰值小时": "6",
                            "第1天_目标谷值": "2",
                            "第1天_目标谷值小时": "3",
                            "第2天_目标峰值": "12",
                            "第2天_目标峰值小时": "7",
                            "第2天_目标谷值": "3",
                            "第2天_目标谷值小时": "4",
                        }
                    )

                turning_csv = label_dir / f"{col}拐点.csv"
                with turning_csv.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["日期", "原始值", "平滑值", "拐点类型", "拐点显著性", "拐点阈值", "有效标记"],
                    )
                    writer.writeheader()
                    for hour in range(72):
                        current = forecast_start - timedelta(hours=24) + timedelta(hours=hour)
                        writer.writerow(
                            {
                                "日期": current.strftime("%Y-%m-%d %H:%M"),
                                "原始值": "0",
                                "平滑值": "0",
                                "拐点类型": "1" if hour == 10 else "-1" if hour == 30 else "0",
                                "拐点显著性": "1",
                                "拐点阈值": "0",
                                "有效标记": "1",
                            }
                        )

            paths = plot_dataset_visualizations(
                records,
                data_root=tmp_path / "数据集",
                dataset_name="ETTH1_pred2_seq1",
                output_root=tmp_path / "数据集可视化",
            )

            self.assertEqual(set(paths), set(TARGET_COLUMNS))
            for col, col_paths in paths.items():
                self.assertTrue(col_paths["peaks_troughs"].exists(), col)
                self.assertTrue(col_paths["turning_points"].exists(), col)
                self.assertIn("ETTH1_pred2_seq1", str(col_paths["peaks_troughs"]))
                self.assertGreater(col_paths["peaks_troughs"].stat().st_size, 1000)
                self.assertGreater(col_paths["turning_points"].stat().st_size, 1000)

    def test_evenly_sample_sample_ids_uses_validation_rows_across_time(self) -> None:
        rows = [
            {"样本ID": f"S{idx:06d}", "预测起点日期": f"2020-01-{idx:02d}", "数据集划分": "验证"}
            for idx in range(1, 11)
        ]
        rows.extend(
            [
                {"样本ID": "S000099", "预测起点日期": "2020-02-01", "数据集划分": "测试"},
                {"样本ID": "S000001", "预测起点日期": "2020-01-01", "数据集划分": "验证"},
            ]
        )

        sample_ids = evenly_sample_sample_ids(rows, split="验证", sample_count=4)

        self.assertEqual(sample_ids, ["S000001", "S000004", "S000007", "S000010"])

    def test_merge_baseline_peak_predictions_combines_value_and_hour_rules(self) -> None:
        value_rows = [
            {
                "样本ID": "S000001",
                "预测起点日期": "2020-01-05",
                "数据集划分": "验证",
                "目标变量": "HUFL",
                "预测天数": "1",
                "目标峰值": "11.5",
                "baseline_name": "weighted_mean_last_4",
                "baseline_peak_value": "10.25",
            },
            {
                "样本ID": "S000001",
                "预测起点日期": "2020-01-05",
                "数据集划分": "验证",
                "目标变量": "HUFL",
                "预测天数": "1",
                "目标峰值": "11.5",
                "baseline_name": "mean_last_4",
                "baseline_peak_value": "9.0",
            },
        ]
        hour_rows = [
            {
                "样本ID": "S000001",
                "预测起点日期": "2020-01-05",
                "数据集划分": "验证",
                "目标变量": "HUFL",
                "预测天数": "1",
                "目标峰值小时": "7",
                "baseline_name": "mode_last_4",
                "baseline_peak_hour": "8",
            },
            {
                "样本ID": "S000001",
                "预测起点日期": "2020-01-05",
                "数据集划分": "验证",
                "目标变量": "HUFL",
                "预测天数": "1",
                "目标峰值小时": "7",
                "baseline_name": "median_last_4",
                "baseline_peak_hour": "6",
            },
        ]

        merged = merge_baseline_peak_predictions(
            value_rows,
            hour_rows,
            sample_id="S000001",
            target_col="HUFL",
            split="验证",
            value_baseline_name="weighted_mean_last_4",
            hour_baseline_name="mode_last_4",
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["预测天数"], 1)
        self.assertEqual(merged[0]["预测起点日期"], "2020-01-05")
        self.assertEqual(merged[0]["目标峰值"], 11.5)
        self.assertEqual(merged[0]["目标峰值小时"], 7)
        self.assertEqual(merged[0]["baseline_peak_value"], 10.25)
        self.assertEqual(merged[0]["baseline_peak_hour"], 8)

    def test_plot_peak_prediction_rows_includes_history_window_and_shifts_prediction_points(self) -> None:
        records = parse_hourly_csv_records(make_records(datetime(2020, 1, 1), 4 * 24))
        prediction_rows = [
            {
                "样本ID": "S000001",
                "预测起点日期": "2020-01-03",
                "数据集划分": "验证",
                "预测天数": 1,
                "目标变量": "HUFL",
                "目标峰值": 23.0,
                "目标峰值小时": 23,
                "baseline_peak_value": 11.0,
                "baseline_peak_hour": 12,
            },
            {
                "样本ID": "S000001",
                "预测起点日期": "2020-01-03",
                "数据集划分": "验证",
                "预测天数": 2,
                "目标变量": "HUFL",
                "目标峰值": 23.0,
                "目标峰值小时": 23,
                "baseline_peak_value": 12.0,
                "baseline_peak_hour": 18,
            },
        ]
        fake_ax = RecordingAxes()
        fake_fig = RecordingFigure()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("wave_dataset.visualization.setup_chinese_font", lambda: None):
                with patch("wave_dataset.visualization.plt.subplots", return_value=(fake_fig, fake_ax)):
                    with patch("wave_dataset.visualization.plt.close", lambda fig: None):
                        path = plot_peak_prediction_rows(
                            records,
                            prediction_rows,
                            target_col="HUFL",
                            dataset_name="ETTH1_pred2_seq2",
                            output_dir=Path(tmp),
                            task_name="测试图",
                            prediction_label="预测",
                            filename_suffix="预测",
                        )

        self.assertTrue(path.name.endswith("_预测.png"))
        self.assertEqual(len(fake_ax.plots[0]["x"]), 96)
        self.assertEqual(fake_ax.xlim, (1, 96))
        self.assertIn(48, fake_ax.vertical_lines)
        self.assertIn({"label": "历史波峰", "x": [24, 48], "y": [23.0, 23.0]}, fake_ax.scatter_summaries)
        self.assertIn({"label": "真实波峰", "x": [72, 96], "y": [23.0, 23.0]}, fake_ax.scatter_summaries)
        self.assertIn({"label": "预测", "x": [61, 91], "y": [11.0, 12.0]}, fake_ax.scatter_summaries)

    def test_plot_peak_value_prediction_batch_writes_all_baseline_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ETTh1.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 8 * 24):
                    writer.writerow(row)

            value_path = tmp_path / "peak_value_predictions.csv"
            value_fieldnames = [
                "样本ID",
                "预测起点日期",
                "数据集划分",
                "目标变量",
                "预测天数",
                "目标峰值",
                "baseline_name",
                "baseline_peak_value",
            ]
            with value_path.open("w", encoding="utf-8", newline="") as value_handle:
                value_writer = csv.DictWriter(value_handle, fieldnames=value_fieldnames)
                value_writer.writeheader()
                for sample_idx, start_day in enumerate(["2020-01-03", "2020-01-05"], start=1):
                    for target_col in ["HUFL", "OT"]:
                        for horizon in range(1, 3):
                            for baseline_name, offset in [("weighted_mean_last_4", 10), ("mean_last_4", 15)]:
                                value_writer.writerow(
                                    {
                                        "样本ID": f"S{sample_idx:06d}",
                                        "预测起点日期": start_day,
                                        "数据集划分": "验证",
                                        "目标变量": target_col,
                                        "预测天数": horizon,
                                        "目标峰值": 20 + horizon,
                                        "baseline_name": baseline_name,
                                        "baseline_peak_value": offset + horizon,
                                    }
                                )

            paths = plot_peak_value_prediction_batch(
                hourly_csv=input_path,
                value_prediction_csv=value_path,
                output_root=tmp_path / "数据集可视化",
                dataset_name="ETTH1_pred2_seq2",
                target_cols=["HUFL", "OT"],
                split="验证",
                sample_count=2,
            )

            self.assertEqual(set(paths), {"weighted_mean_last_4", "mean_last_4"})
            self.assertEqual(set(paths["weighted_mean_last_4"]), {"HUFL", "OT"})
            self.assertEqual(len(paths["weighted_mean_last_4"]["HUFL"]), 2)
            expected_dir = (
                tmp_path
                / "数据集可视化"
                / "ETTH1_pred2_seq2"
                / "波峰值"
                / "weighted_mean_last_4"
                / "HUFL"
            )
            self.assertTrue(expected_dir.is_dir())
            for baseline_paths in paths.values():
                for target_paths in baseline_paths.values():
                    for path in target_paths:
                        self.assertTrue(path.exists())
                        self.assertGreater(path.stat().st_size, 1000)

    def test_plot_peak_value_prediction_batch_defaults_to_six_samples_per_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ETTh1.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 10 * 24):
                    writer.writerow(row)

            value_path = tmp_path / "peak_value_predictions.csv"
            with value_path.open("w", encoding="utf-8", newline="") as value_handle:
                value_writer = csv.DictWriter(
                    value_handle,
                    fieldnames=[
                        "样本ID",
                        "预测起点日期",
                        "数据集划分",
                        "目标变量",
                        "预测天数",
                        "目标峰值",
                        "baseline_name",
                        "baseline_peak_value",
                    ],
                )
                value_writer.writeheader()
                for sample_idx in range(1, 9):
                    value_writer.writerow(
                        {
                            "样本ID": f"S{sample_idx:06d}",
                            "预测起点日期": f"2020-01-{sample_idx + 1:02d}",
                            "数据集划分": "验证",
                            "目标变量": "HUFL",
                            "预测天数": 1,
                            "目标峰值": 23,
                            "baseline_name": "weighted_mean_last_4",
                            "baseline_peak_value": 11,
                        }
                    )

            paths = plot_peak_value_prediction_batch(
                hourly_csv=input_path,
                value_prediction_csv=value_path,
                output_root=tmp_path / "数据集可视化",
                dataset_name="ETTH1_pred1_seq1",
                target_cols=["HUFL"],
                split="验证",
            )

            self.assertEqual(len(paths["weighted_mean_last_4"]["HUFL"]), 6)

    def test_plot_peak_hour_prediction_batch_writes_all_baseline_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ETTh1.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 8 * 24):
                    writer.writerow(row)

            hour_path = tmp_path / "peak_hour_predictions.csv"
            hour_fieldnames = [
                "样本ID",
                "预测起点日期",
                "数据集划分",
                "目标变量",
                "预测天数",
                "目标峰值小时",
                "baseline_name",
                "baseline_peak_hour",
            ]
            with hour_path.open("w", encoding="utf-8", newline="") as hour_handle:
                hour_writer = csv.DictWriter(hour_handle, fieldnames=hour_fieldnames)
                hour_writer.writeheader()
                for sample_idx, start_day in enumerate(["2020-01-03", "2020-01-05"], start=1):
                    for target_col in ["HUFL", "OT"]:
                        for horizon in range(1, 3):
                            for baseline_name, peak_hour in [("mode_last_4", 12), ("median_last_4", 18)]:
                                hour_writer.writerow(
                                    {
                                        "样本ID": f"S{sample_idx:06d}",
                                        "预测起点日期": start_day,
                                        "数据集划分": "验证",
                                        "目标变量": target_col,
                                        "预测天数": horizon,
                                        "目标峰值小时": 23,
                                        "baseline_name": baseline_name,
                                        "baseline_peak_hour": peak_hour,
                                    }
                                )

            paths = plot_peak_hour_prediction_batch(
                hourly_csv=input_path,
                hour_prediction_csv=hour_path,
                output_root=tmp_path / "数据集可视化",
                dataset_name="ETTH1_pred2_seq2",
                target_cols=["HUFL", "OT"],
                split="验证",
                sample_count=2,
            )

            self.assertEqual(set(paths), {"mode_last_4", "median_last_4"})
            self.assertEqual(set(paths["mode_last_4"]), {"HUFL", "OT"})
            self.assertEqual(len(paths["mode_last_4"]["HUFL"]), 2)
            expected_dir = (
                tmp_path
                / "数据集可视化"
                / "ETTH1_pred2_seq2"
                / "波峰小时"
                / "mode_last_4"
                / "HUFL"
            )
            self.assertTrue(expected_dir.is_dir())
            for baseline_paths in paths.values():
                for target_paths in baseline_paths.values():
                    for path in target_paths:
                        self.assertTrue(path.exists())
                        self.assertGreater(path.stat().st_size, 1000)

    def test_plot_peak_prediction_batch_uses_combined_folder_when_value_and_hour_are_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ETTh1.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 5 * 24):
                    writer.writerow(row)

            value_path = tmp_path / "peak_value_predictions.csv"
            hour_path = tmp_path / "peak_hour_predictions.csv"
            with value_path.open("w", encoding="utf-8", newline="") as value_handle:
                value_writer = csv.DictWriter(
                    value_handle,
                    fieldnames=[
                        "样本ID",
                        "预测起点日期",
                        "数据集划分",
                        "目标变量",
                        "预测天数",
                        "目标峰值",
                        "baseline_name",
                        "baseline_peak_value",
                    ],
                )
                value_writer.writeheader()
                value_writer.writerow(
                    {
                        "样本ID": "S000001",
                        "预测起点日期": "2020-01-03",
                        "数据集划分": "验证",
                        "目标变量": "HUFL",
                        "预测天数": 1,
                        "目标峰值": 23,
                        "baseline_name": "weighted_mean_last_4",
                        "baseline_peak_value": 11,
                    }
                )
            with hour_path.open("w", encoding="utf-8", newline="") as hour_handle:
                hour_writer = csv.DictWriter(
                    hour_handle,
                    fieldnames=[
                        "样本ID",
                        "预测起点日期",
                        "数据集划分",
                        "目标变量",
                        "预测天数",
                        "目标峰值小时",
                        "baseline_name",
                        "baseline_peak_hour",
                    ],
                )
                hour_writer.writeheader()
                hour_writer.writerow(
                    {
                        "样本ID": "S000001",
                        "预测起点日期": "2020-01-03",
                        "数据集划分": "验证",
                        "目标变量": "HUFL",
                        "预测天数": 1,
                        "目标峰值小时": 23,
                        "baseline_name": "mode_last_4",
                        "baseline_peak_hour": 12,
                    }
                )

            paths = plot_peak_prediction_batch(
                hourly_csv=input_path,
                value_prediction_csv=value_path,
                hour_prediction_csv=hour_path,
                output_root=tmp_path / "数据集可视化",
                dataset_name="ETTH1_pred2_seq2",
                target_cols=["HUFL"],
                split="验证",
                sample_count=1,
                value_baseline_name="weighted_mean_last_4",
                hour_baseline_name="mode_last_4",
            )

            expected_dir = (
                tmp_path
                / "数据集可视化"
                / "ETTH1_pred2_seq2"
                / "波峰小时图"
                / "weighted_mean_last_4+mode_last_4"
                / "HUFL"
            )
            self.assertTrue(expected_dir.is_dir())
            self.assertEqual(len(paths["weighted_mean_last_4+mode_last_4"]["HUFL"]), 1)

    def test_plot_selected_best_baseline_prediction_batch_writes_target_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ETTh1.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", *TARGET_COLUMNS])
                writer.writeheader()
                for row in make_records(datetime(2020, 1, 1), 8 * 24):
                    writer.writerow(row)

            selected_path = tmp_path / "dataset_with_selected_baselines.csv"
            fieldnames = [
                "样本ID",
                "预测起点日期",
                "数据集划分",
                "目标变量",
                "预测天数",
                "目标峰值",
                "baseline_peak",
                "target_peak_residual",
                "peak_value_baseline_name",
                "目标峰值小时",
                "baseline_peak_hour",
                "peak_hour_baseline_name",
                "peak_hour_ordinary_error",
                "peak_hour_circular_error",
                "peak_hour_within_2h",
            ]
            with selected_path.open("w", encoding="utf-8", newline="") as selected_handle:
                writer = csv.DictWriter(selected_handle, fieldnames=fieldnames)
                writer.writeheader()
                for sample_idx, start_day in enumerate(["2020-01-03", "2020-01-05"], start=1):
                    for target_col in ["HUFL", "OT"]:
                        for horizon in range(1, 3):
                            writer.writerow(
                                {
                                    "样本ID": f"S{sample_idx:06d}",
                                    "预测起点日期": start_day,
                                    "数据集划分": "验证",
                                    "目标变量": target_col,
                                    "预测天数": horizon,
                                    "目标峰值": 20 + horizon,
                                    "baseline_peak": 10 + horizon,
                                    "target_peak_residual": 10,
                                    "peak_value_baseline_name": "mean_last_4" if horizon == 1 else "cycle_mod_4",
                                    "目标峰值小时": 23,
                                    "baseline_peak_hour": 12 if horizon == 1 else 18,
                                    "peak_hour_baseline_name": "mode_last_4" if horizon == 1 else "global_mode",
                                    "peak_hour_ordinary_error": 11,
                                    "peak_hour_circular_error": 6,
                                    "peak_hour_within_2h": "False",
                                }
                            )

            paths = plot_selected_best_baseline_prediction_batch(
                hourly_csv=input_path,
                selected_baseline_csv=selected_path,
                output_root=tmp_path / "数据集可视化",
                dataset_name="ETTH1_pred2_seq2",
                target_cols=["HUFL", "OT"],
                split="验证",
                sample_count=2,
            )

            self.assertEqual(set(paths), {"HUFL", "OT"})
            self.assertEqual(len(paths["HUFL"]), 2)
            expected_dir = (
                tmp_path
                / "数据集可视化"
                / "ETTH1_pred2_seq2"
                / "逐日最佳基线组合"
                / "HUFL"
            )
            self.assertTrue(expected_dir.is_dir())
            for target_paths in paths.values():
                for path in target_paths:
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 1000)

    def test_convert_wide_dataset_to_long_table_outputs_chinese_columns_and_merges_all_features(self) -> None:
        from wave_dataset.long_table import convert_dataset_to_long_tables

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ETTH1_pred2_seq2"
            write_wide_rows(
                dataset_dir,
                "HUFL",
                [
                    {
                        "样本ID": "S000001",
                        "预测起点日期": "2020-01-05",
                        "目标变量": "HUFL",
                        "预测天数": 2,
                        "基线峰值": 10.0,
                        "数据集划分": "训练",
                        "第1天_目标峰值": 11.0,
                        "第1天_目标峰值残差": 1.0,
                        "第1天_目标峰值小时": 3,
                        "第1天_目标谷值": 4.0,
                        "第1天_目标谷值残差": -1.0,
                        "第1天_目标谷值小时": 8,
                        "第1天_日历_星期": 1,
                        "第1天_日历_月份": 1,
                        "第1天_日历_年内日序": 5,
                        "第1天_日历_是否周末": 0,
                        "第2天_目标峰值": 12.0,
                        "第2天_目标峰值残差": 2.0,
                        "第2天_目标峰值小时": 4,
                        "第2天_目标谷值": 5.0,
                        "第2天_目标谷值残差": 0.0,
                        "第2天_目标谷值小时": 9,
                        "第2天_日历_星期": 2,
                        "第2天_日历_月份": 1,
                        "第2天_日历_年内日序": 6,
                        "第2天_日历_是否周末": 0,
                        "HUFL_过去96小时_均值": 7.1,
                    }
                ],
            )
            write_wide_rows(
                dataset_dir,
                "OT",
                [
                    {
                        "样本ID": "S000001",
                        "预测起点日期": "2020-01-05",
                        "目标变量": "OT",
                        "预测天数": 2,
                        "基线峰值": 30.0,
                        "数据集划分": "训练",
                        "第1天_目标峰值": 31.0,
                        "第1天_目标峰值残差": 1.0,
                        "第1天_目标峰值小时": 13,
                        "第1天_目标谷值": 20.0,
                        "第1天_目标谷值残差": -2.0,
                        "第1天_目标谷值小时": 5,
                        "第1天_日历_星期": 1,
                        "第1天_日历_月份": 1,
                        "第1天_日历_年内日序": 5,
                        "第1天_日历_是否周末": 0,
                        "第2天_目标峰值": 32.0,
                        "第2天_目标峰值残差": 2.0,
                        "第2天_目标峰值小时": 14,
                        "第2天_目标谷值": 21.0,
                        "第2天_目标谷值残差": -1.0,
                        "第2天_目标谷值小时": 6,
                        "第2天_日历_星期": 2,
                        "第2天_日历_月份": 1,
                        "第2天_日历_年内日序": 6,
                        "第2天_日历_是否周末": 0,
                        "OT_过去96小时_均值": 27.5,
                    }
                ],
            )

            result = convert_dataset_to_long_tables(dataset_dir, target_cols=["HUFL", "OT"], seq_days=2, pred_days=2)

            self.assertTrue((dataset_dir / "长表" / "OT" / "峰谷预测长表_seq48_pred48_OT.csv").exists())
            self.assertEqual(result["row_counts"]["OT"], 2)
            with (dataset_dir / "长表" / "OT" / "峰谷预测长表_seq48_pred48_OT.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["预测天数"] for row in rows], ["1", "2"])
            self.assertEqual(rows[0]["目标变量"], "OT")
            self.assertEqual(rows[0]["目标峰值"], "31.0")
            self.assertEqual(rows[1]["目标峰值小时"], "14")
            self.assertEqual(rows[0]["日历_星期"], "6")
            self.assertEqual(rows[0]["日历_是否周末"], "1")
            self.assertEqual(rows[1]["日历_星期"], "0")
            self.assertEqual(rows[1]["日历_是否周末"], "0")
            self.assertEqual(rows[0]["OT_过去96小时_均值"], "27.5")
            self.assertEqual(rows[0]["HUFL_过去96小时_均值"], "7.1")
            self.assertNotIn("horizon", rows[0])
            self.assertNotIn("target_peak_value", rows[0])


if __name__ == "__main__":
    unittest.main()
