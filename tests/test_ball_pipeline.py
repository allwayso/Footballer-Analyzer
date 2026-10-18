"""验证足球补跑的源帧、独立版本、失败恢复和人物标定原件保护。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from footballer_analyzer.ball_pipeline import ball_versions, load_ball_result, select_in_windows, track_ball_run
from footballer_analyzer.ball_tracking import BallSelectionConfig
from footballer_analyzer.storage import file_hash, read_json, write_json
import test_reporting


class BallPipelineTests(unittest.TestCase):
    """用独立合成源视频检查补跑，不依赖真实模型的精度。"""

    def test_append_keeps_originals_and_missing_source_frames(self):
        """人物 stride 为三时仍逐帧追球，空帧不插值，原件指纹不变。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            manifest["config"]["stride"] = 3
            manifest.update(start_frame=2, end_frame=8)
            write_json(run_dir / "manifest.json", manifest)
            weights = Path(temporary) / "weights.pt"
            weights.write_bytes(b"test")
            preset = {"weights": str(weights), "imgsz": 640, "tile": 640, "label": "test"}
            names = ["manifest.json", "tracks.parquet", "calibration.yaml", "bindings.yaml"]
            before = {name: file_hash(run_dir / name) for name in names}
            ball = [20, 20, 24, 24, 0.9]
            with patch("footballer_analyzer.ball_pipeline.model_preset", return_value=preset), patch("ultralytics.YOLO"), patch("footballer_analyzer.ball_pipeline.detect", side_effect=[[ball], [ball], [], [ball], [ball], [ball]]):
                result = track_ball_run(run_dir)
            frames = load_ball_result(run_dir)["frames"]
            self.assertEqual([row["frame"] for row in frames], list(range(2, 8)))
            self.assertIsNone(frames[2]["selected"])
            self.assertEqual(result["observed_frames"], 5)
            self.assertEqual(before, {name: file_hash(run_dir / name) for name in names})
            self.assertEqual(len(ball_versions(run_dir)), 1)
            self.assertIsNone(load_ball_result(run_dir, "coco_v8n"))
            with patch("footballer_analyzer.ball_pipeline.model_preset", return_value=preset), patch("ultralytics.YOLO"), patch("footballer_analyzer.ball_pipeline.detect", side_effect=RuntimeError("模拟失败")):
                with self.assertRaisesRegex(RuntimeError, "模拟失败"):
                    track_ball_run(run_dir)
            self.assertEqual(load_ball_result(run_dir)["manifest"]["result_id"], result["result_id"])
            self.assertEqual(before, {name: file_hash(run_dir / name) for name in names})
            manifest["video_sha256"] = "changed"
            write_json(run_dir / "manifest.json", manifest)
            with patch("footballer_analyzer.ball_pipeline.model_preset", return_value=preset), self.assertRaisesRegex(ValueError, "源视频已变化"):
                track_ball_run(run_dir)

    def test_window_boundary_keeps_frame_alignment_and_gaps(self):
        """跨筛选窗口的稳定球保留对应帧，边界缺失保持 None。"""
        records = [{"frame": index, "candidates": [] if index in {249, 250, 500} else [[100 + index, 100, 104 + index, 104, 0.9]]} for index in range(620)]
        selected = select_in_windows(records, 25, (1920, 1080), BallSelectionConfig())
        self.assertEqual(len(selected), 620)
        for record, candidate in zip(records, selected):
            self.assertEqual(candidate, record["candidates"][0] if record["candidates"] else None)

    def test_unknown_model_and_old_run(self):
        """旧运行无球文件可读取，模型名不允许越界路径。"""
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(load_ball_result(Path(temporary)))
            with self.assertRaises(ValueError):
                load_ball_result(Path(temporary), "../../other")


if __name__ == "__main__":
    unittest.main()
