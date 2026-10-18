"""验证独立人物追踪、容器虚报末尾及后台任务的重入与失败行为。"""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from footballer_analyzer.analysis import analyze_clip
from footballer_analyzer.config import config_from_dict
from footballer_analyzer.server import ReviewApp
from footballer_analyzer.storage import write_json
from footballer_analyzer.video import VideoInfo


class AnalysisTests(unittest.TestCase):
    """素材入口与顺序任务的边界回归测试。"""

    def prepare_clip(self, directory):
        """写入三帧可解码素材，返回受限素材目录与路径。"""
        clips = Path(directory) / "clips"
        clips.mkdir()
        video = clips / "sample.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
        for frame in range(3):
            writer.write(np.full((48, 64, 3), frame * 20, np.uint8))
        writer.release()
        return clips, video

    def test_complete_source_uses_decodable_tail_without_calibration(self):
        """容器虚报帧数时使用真实末尾，十一人制也不自动触发场地计算。"""
        with tempfile.TemporaryDirectory() as temporary:
            _, video = self.prepare_clip(temporary)
            config = config_from_dict({"video": str(video), "weights": "models/yolov8n.pt", "match_id": "test", "clip_id": "sample", "start_frame": 1})
            run_dir = Path(temporary) / "run_test"
            progress = []

            def fake_track(actual, callback):
                """检查真实源帧窗口并发出人物进度。"""
                self.assertEqual((actual.start_frame, actual.end_frame, actual.stride), (1, 3, 1))
                callback("tracking", 2, 2)
                return run_dir

            with patch("footballer_analyzer.analysis.inspect_video", return_value=VideoInfo(64, 48, 25, 8)), patch("footballer_analyzer.analysis.track", side_effect=fake_track), patch("footballer_analyzer.white_line_pipeline.calibrate_run") as calibration:
                self.assertEqual(analyze_clip(config, lambda *values: progress.append(values)), run_dir)
                calibration.assert_not_called()
            self.assertEqual([entry[0] for entry in progress], ["inspecting", "tracking", "awaiting_calibration"])
            self.assertEqual(progress[-1], ("awaiting_calibration", 2, 2, run_dir))

    def test_clip_inventory_and_path_restrictions(self):
        """仅白线及失败运行仍显示未完成人物追踪，素材路径不可越界。"""
        with tempfile.TemporaryDirectory() as temporary:
            clips, video = self.prepare_clip(temporary)
            output = Path(temporary) / "outputs"
            app = ReviewApp(output, clips)
            for run_id, status, kind in [("run_lines", "complete", "white_line_review"), ("run_failed", "failed", "tracking")]:
                write_json(output / run_id / "manifest.json", {"status": status, "kind": kind, "config": {"video": str(video)}})
            self.assertEqual(app.clips()[0]["tracking_runs"], 0)
            write_json(output / "run_good/manifest.json", {"status": "complete", "config": {"video": str(video)}})
            self.assertEqual(app.clips()[0]["tracking_runs"], 1)
            for invalid in [None, "../sample.avi", "..\\sample.avi", str(video), "sample.txt", "missing.avi"]:
                with self.assertRaises(ValueError):
                    app.clip_path(invalid)

    def test_eight_manual_analysis_skips_white_lines_and_keeps_run(self):
        """八人制默认等待手动标定，不构造或运行十一人制自动标定。"""
        with tempfile.TemporaryDirectory() as temporary:
            _, video = self.prepare_clip(temporary)
            config = config_from_dict({"video": str(video), "weights": "models/yolov8n.pt", "match_id": "test", "clip_id": "sample", "pitch_model": "eight"})
            self.assertEqual((config.pitch_length_m, config.pitch_width_m, config.calibration_method), (68, 50, "manual"))
            progress = []
            run_dir = Path(temporary) / "run_test"
            with patch("footballer_analyzer.analysis.track", return_value=run_dir), patch("footballer_analyzer.white_line_pipeline.calibrate_run") as calibration:
                self.assertEqual(analyze_clip(config, lambda *values: progress.append(values)), run_dir)
                calibration.assert_not_called()
            self.assertEqual(progress[-1], ("awaiting_calibration", 3, 3, run_dir))

    def test_eight_job_defaults_and_automatic_mode_rejection(self):
        """HTTP 分析入口按八人制默认尺寸创建任务，拒绝八人制自动匹配。"""
        with tempfile.TemporaryDirectory() as temporary:
            clips, _ = self.prepare_clip(temporary)
            app = ReviewApp(Path(temporary) / "outputs", clips)
            finished = threading.Event()

            def fake_analysis(config, progress):
                """记录手动任务配置，返回同一个人物运行。"""
                self.assertEqual((config.pitch_model, config.calibration_method, config.pitch_length_m, config.pitch_width_m), ("eight", "manual", 68, 50))
                finished.set()
                return Path(temporary) / "run_test"

            with patch("footballer_analyzer.analysis.analyze_clip", side_effect=fake_analysis), patch("footballer_analyzer.dynamic_calibration.WhiteLineCalibrator") as automatic:
                app.start_analysis({"clip_id": "sample.avi", "pitch_model": "eight"})
                self.assertTrue(finished.wait(5))
                for thread in threading.enumerate():
                    if thread.name.endswith("(worker)"):
                        thread.join(timeout=5)
                automatic.assert_not_called()
            self.assertEqual(app.analysis_job["stage"], "awaiting_calibration")
            self.assertEqual(app.analysis_job["status"], "complete")
            with self.assertRaisesRegex(ValueError, "仅支持手动"):
                app.start_analysis({"clip_id": "sample.avi", "pitch_model": "eight", "calibration_method": "white_lines"})

    def test_background_job_reuses_duplicate_and_preserves_partial_result(self):
        """相同请求复用任务，不同参数被拒绝；场地失败仍保留可查看的人物运行。"""
        with tempfile.TemporaryDirectory() as temporary:
            clips, _ = self.prepare_clip(temporary)
            app = ReviewApp(Path(temporary) / "outputs", clips)
            entered = threading.Event()
            release = threading.Event()

            def fake_analysis(config, progress):
                """保持后台任务进行中，然后模拟场地计算失败。"""
                progress("calibrating", 1, 3, Path(temporary) / "run_test")
                entered.set()
                release.wait(5)
                raise ValueError("模拟标定失败")

            with patch("footballer_analyzer.analysis.analyze_clip", side_effect=fake_analysis):
                job = app.start_analysis({"clip_id": "sample.avi"})
                self.assertTrue(entered.wait(5))
                duplicate = app.start_analysis({"clip_id": "sample.avi"})
                self.assertEqual(job["job_id"], duplicate["job_id"])
                with self.assertRaisesRegex(ValueError, "正在分析"):
                    app.start_analysis({"clip_id": "sample.avi", "imgsz": 1280})
                write_json(app.output_root / "run_test/manifest.json", {"status": "complete"})
                with self.assertRaisesRegex(ValueError, "人物追踪尚未完成"):
                    app.start_calibration("run_test", {})
                release.set()
                for thread in threading.enumerate():
                    if thread.name.endswith("(worker)"):
                        thread.join(timeout=5)
            self.assertEqual(app.analysis_job["status"], "failed")
            self.assertEqual(app.analysis_job["run_id"], "run_test")
            self.assertEqual(app.analysis_job["error"], "模拟标定失败")
            with self.assertRaises(ValueError):
                app.start_analysis({"clip_id": "sample.avi", "start_frame": -1})
            with self.assertRaises(ValueError):
                app.start_analysis({"clip_id": "sample.avi", "pitch_width_m": 10})


if __name__ == "__main__":
    unittest.main()
