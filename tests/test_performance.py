"""验证性能组件的计时、重复试验、独立执行和基线比较，不依赖真实权重。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv

from footballer_analyzer.benchmark import compare_results, run_benchmark, summarize
from footballer_analyzer.cli import main
from footballer_analyzer.config import Config
from footballer_analyzer.performance import PerformanceRecorder
from footballer_analyzer.storage import read_json, write_json


class PerformanceTests(unittest.TestCase):
    """用已知时间和短合成视频检查真实流水线的计时接口。"""

    def prepare_config(self, root):
        """创建六帧视频和占位权重，只在检测器被替换时使用。"""
        root = Path(root)
        video = root / "sample.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
        for _ in range(6):
            writer.write(np.full((48, 64, 3), (45, 115, 55), np.uint8))
        writer.release()
        weights = root / "weights.pt"
        weights.write_bytes(b"test-only")
        return Config(video, weights, root / "normal-outputs", root / "players.yaml", "bench", "sample", start_frame=1, end_frame=6, stride=2)

    def test_context_measures_time_and_drops_failed_samples(self):
        """已知两秒内处理五帧，覆盖十源帧；失败调用不被记作成功样本。"""
        recorder = PerformanceRecorder()
        with patch("footballer_analyzer.performance.time.perf_counter", side_effect=[10, 12]):
            with recorder.stage("tracking", 5, 25, source_frames=10):
                pass
        metric = recorder.metrics[0]
        self.assertEqual((metric.elapsed_s, metric.fps, metric.ms_per_frame), (2, 2.5, 400))
        self.assertAlmostEqual(metric.realtime_factor, 0.2)
        with self.assertRaisesRegex(RuntimeError, "失败"):
            with recorder.stage("ball", 10):
                raise RuntimeError("失败")
        self.assertEqual(len(recorder.metrics), 1)

    def test_invalid_numbers_are_rejected(self):
        """非有限值和零帧不能污染 JSON 或产生无意义吞吐。"""
        for frames, seconds, fps in [(0, 1, 25), (True, 1, 25), (1, 0, 25), (1, float("nan"), 25), (1, 1, float("inf"))]:
            with self.subTest(frames=frames, seconds=seconds, fps=fps):
                with self.assertRaises(ValueError):
                    PerformanceRecorder().record("tracking", frames, seconds, fps)

    def test_aggregate_uses_total_work_and_baseline_ratio(self):
        """不同样本的 FPS 应由总工作量计算，基线使用同协议并拒绝异素材。"""
        recorder = PerformanceRecorder()
        recorder.record("tracking", 10, 1)
        recorder.record("tracking", 30, 9)
        summary = summarize(recorder)
        self.assertEqual(summary["tracking"]["fps"], 4)
        protocol = {"stages": ["tracking"], "video_sha256": "same"}
        baseline = {"status": "complete", "protocol": protocol, "summary": {"tracking": {"fps": 2}}}
        current = {"protocol": protocol, "summary": summary}
        self.assertEqual(compare_results(current, baseline)["tracking"]["speedup"], 2)
        with self.assertRaises(ValueError):
            compare_results({**current, "protocol": {**protocol, "video_sha256": "other"}}, baseline)

    def test_compare_allows_different_device_but_rejects_other_protocol_changes(self):
        """CPU 与 GPU 属同一测试协议，设备不同应允许比较并标注设备来源。"""
        protocol = {"stages": ["tracking"], "video_sha256": "same", "device": "cpu",
                    "config": {"video": "a.mp4", "imgsz": 1920, "device": "cpu"}}
        baseline = {"status": "complete", "protocol": protocol, "summary": {"tracking": {"fps": 4}}}
        current = {"protocol": {**protocol, "device": "gpu", "config": {**protocol["config"], "device": "gpu"}},
                   "summary": {"tracking": {"fps": 10}}}
        comparison = compare_results(current, baseline)["tracking"]
        self.assertEqual(comparison["speedup"], 2.5)
        self.assertEqual(comparison["baseline_device"], "cpu")
        self.assertEqual(comparison["current_device"], "gpu")
        with self.assertRaises(ValueError):
            compare_results({**current, "protocol": {**protocol, "device": "gpu", "warmup": 9}}, baseline)
        with self.assertRaises(ValueError):
            compare_results({**current, "protocol": {**current["protocol"], "config": {**protocol["config"], "device": "gpu", "imgsz": 640}}}, baseline)

    def test_warmup_is_excluded_and_each_repeat_reruns(self):
        """预热和正式轮均执行新流水线，统计排除预热且不读历史耗时。"""
        with tempfile.TemporaryDirectory() as temporary:
            config = self.prepare_config(temporary)
            measured = iter([100, 4, 2])
            seen_paths = []

            def fake_track(actual, progress=None, performance=None):
                """为三个独立调用提供可独立核对的耗时。"""
                run_dir = actual.output_root / "run_test"
                run_dir.mkdir(parents=True)
                seen_paths.append(run_dir)
                performance.record("tracking", 3, next(measured), 25, source_frames=5)
                return run_dir

            with patch("footballer_analyzer.benchmark.track", side_effect=fake_track) as track_call:
                result = run_benchmark(config, stages=["tracking"], repeat=2, warmup=1, output_root=Path(temporary) / "bench")
            self.assertEqual(track_call.call_count, 3)
            self.assertEqual(len(set(seen_paths)), 3)
            self.assertEqual([row["elapsed_s"] for row in result["records"]], [4, 2])
            self.assertEqual(result["summary"]["tracking"]["fps"], 1)
            self.assertEqual(read_json(Path(result["result_path"]))["status"], "complete")
            self.assertFalse(config.output_root.exists())

    def test_real_pipelines_record_counts_and_work_without_tracking(self):
        """三条真实流水线使用合成输入；足球可独立运行而不隐式重跑人物和场地。"""
        with tempfile.TemporaryDirectory() as temporary:
            config = self.prepare_config(temporary)
            preset = {"weights": str(config.weights), "imgsz": 640, "tile": 0, "label": "test"}
            with patch("footballer_analyzer.pipeline.YoloDetector") as detector, patch("ultralytics.YOLO"), patch("footballer_analyzer.ball_pipeline.detect", return_value=[]), patch("footballer_analyzer.ball_pipeline.model_preset", return_value=preset):
                detector.return_value.detect.side_effect = lambda image: sv.Detections.empty()
                result = run_benchmark(config, output_root=Path(temporary) / "bench")
                self.assertEqual({row["stage"]: row["frames"] for row in result["records"]}, {"tracking": 3, "pitch": 5, "ball": 5})
                self.assertTrue(all(row["elapsed_s"] > 0 for row in result["records"]))
                self.assertEqual(result["records"][1]["metadata"]["valid_ratio"], 0)
                self.assertEqual(result["records"][2]["metadata"]["observed_frames"], 0)
                with patch("footballer_analyzer.benchmark.track") as tracking, patch("footballer_analyzer.benchmark.calibrate_run") as pitch:
                    independent = run_benchmark(config, stages=["ball"], output_root=Path(temporary) / "bench")
                tracking.assert_not_called()
                pitch.assert_not_called()
                self.assertEqual(independent["summary"]["ball"]["frames"], 5)
            self.assertFalse(config.output_root.exists())

    def test_failures_are_saved_and_invalid_requests_do_not_start_work(self):
        """失败写入独立报告；非法轮数、区间、八人制白线和基线先行拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            config = self.prepare_config(temporary)
            destination = Path(temporary) / "bench"
            with patch("footballer_analyzer.benchmark.track", side_effect=RuntimeError("模拟失败")):
                with self.assertRaisesRegex(RuntimeError, "模拟失败"):
                    run_benchmark(config, stages=["tracking"], output_root=destination)
            report = read_json(next(destination.glob("benchmark_*/result.json")))
            self.assertEqual((report["status"], report["records"]), ("failed", []))
            with patch("footballer_analyzer.benchmark.track") as tracking:
                for options in [{"repeat": 0}, {"stages": ["no"]}, {"stages": ["tracking", "tracking"]}]:
                    with self.assertRaises(ValueError):
                        run_benchmark(config, **options)
                for invalid in [replace(config, start_frame=-1), replace(config, end_frame=7), replace(config, pitch_model="eight", pitch_length_m=68, pitch_width_m=50, calibration_method="manual")]:
                    with self.assertRaises(ValueError):
                        run_benchmark(invalid, stages=["pitch"])
                baseline = Path(temporary) / "bad.json"
                write_json(baseline, {"status": "failed"})
                with self.assertRaises(ValueError):
                    run_benchmark(config, stages=["tracking"], baseline=baseline)
                tracking.assert_not_called()

    def test_cli_passes_selected_stage_and_frame_window(self):
        """CLI 把阶段、区间、重复次数和基线路径交给可复用组件。"""
        with tempfile.TemporaryDirectory() as temporary:
            config = self.prepare_config(temporary)
            with patch("footballer_analyzer.cli.load_config", return_value=config), patch("footballer_analyzer.benchmark.run_benchmark", return_value={}) as benchmark, patch("footballer_analyzer.benchmark.print_benchmark"):
                code = main(["benchmark", "--config", "unused.yaml", "--stages", "ball", "--start-frame", "2", "--end-frame", "4", "--repeat", "3", "--warmup", "1"])
            self.assertEqual(code, 0)
            actual = benchmark.call_args.args
            self.assertEqual((actual[0].start_frame, actual[0].end_frame, actual[1:4]), (2, 4, (["ball"], 3, 1)))


if __name__ == "__main__":
    unittest.main()
