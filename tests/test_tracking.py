"""验证低分关联、缓冲恢复及空帧语义，防止丢失后无故换号。"""

import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import cv2
import supervision as sv

from footballer_analyzer.config import Config, config_from_dict
from footballer_analyzer.detection import YoloDetector, resolve_device
from footballer_analyzer.pipeline import track
from footballer_analyzer.storage import read_json, read_rows
from footballer_analyzer.tracking import ByteTracker


def people(boxes, scores):
    """构造人物检测；框为像素 xyxy，分数为检测置信度。"""
    return sv.Detections(xyxy=np.array(boxes, dtype=float).reshape(-1, 4), confidence=np.array(scores), class_id=np.zeros(len(scores), dtype=int))


class TrackingTests(unittest.TestCase):
    """使用实际 ByteTrack 验证关联行为，不依赖神经网络权重。"""

    def test_low_score_keeps_existing_identity_without_starting_false_tracks(self):
        """高分人物降至 0.15 后沿用编号，远处低分框不生成新轨迹。"""
        tracker = ByteTracker(25, 1)
        first = tracker.update(people([[100, 100, 130, 180]], [0.9]), 0)
        for frame_index in range(1, 12):
            tracked = tracker.update(people([[100 + frame_index, 100, 130 + frame_index, 180], [400, 100, 430, 180]], [0.15, 0.15]), frame_index / 25)
            self.assertEqual(tracked.tracker_id.tolist(), first.tracker_id.tolist())
            self.assertAlmostEqual(float(tracked.confidence[0]), 0.15)

    def test_gap_recovery_and_expiry(self):
        """40 帧漏检后可恢复原编号；超过三秒缓冲后创建新编号。"""
        tracker = ByteTracker(25, 1)
        first = tracker.update(people([[100, 100, 130, 180]], [0.9]), 0)
        for frame_index in range(1, 41):
            self.assertEqual(len(tracker.update(people([], []), frame_index / 25)), 0)
        recovered = tracker.update(people([[100, 100, 130, 180]], [0.9]), 41 / 25)
        self.assertEqual(recovered.tracker_id.tolist(), first.tracker_id.tolist())
        for frame_index in range(42, 125):
            self.assertEqual(len(tracker.update(people([], []), frame_index / 25)), 0)
        tracker.update(people([[100, 100, 130, 180]], [0.9]), 5)
        expired = tracker.update(people([[100, 100, 130, 180]], [0.9]), 5.04)
        self.assertEqual(len(expired), 1)
        self.assertNotEqual(expired.tracker_id.tolist(), first.tracker_id.tolist())

    def test_two_nearby_people_keep_separate_ids(self):
        """两个相邻目标在检测顺序反转、一个降分时仍对应各自编号。"""
        tracker = ByteTracker(25, 1)
        first = tracker.update(people([[100, 100, 130, 180], [125, 100, 155, 180]], [0.9, 0.9]), 0)
        second = tracker.update(people([[126, 100, 156, 180], [101, 100, 131, 180]], [0.9, 0.15]), 0.04)
        self.assertEqual(second.tracker_id.tolist(), first.tracker_id[::-1].tolist())

    def test_buffer_uses_actual_sample_rate_and_timestamps_are_valid(self):
        """25 FPS 按三帧采样时三秒缓冲为 25 帧，拒绝倒退或非有限时间。"""
        tracker = ByteTracker(25, 3)
        self.assertEqual(tracker.tracker.max_time_lost, 25)
        tracker.update(people([], []), 0)
        for timestamp in [0, -1, float("nan"), float("inf")]:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                tracker.update(people([], []), timestamp)

    def test_detector_passes_low_floor_and_nms_to_yolo(self):
        """确认实际推理接收低分下限和去重阈值，防止只修改跟踪配置。"""
        with patch("footballer_analyzer.detection.YOLO") as model_type, patch("footballer_analyzer.detection.sv.Detections.from_ultralytics"):
            weights = Mock(spec=Path)
            weights.is_file.return_value = True
            detector = YoloDetector(weights)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            detector.detect(image)
            parameters = model_type.return_value.predict.call_args.kwargs
            self.assertEqual(parameters["conf"], 0.1)
            self.assertEqual(parameters["iou"], 0.5)
            self.assertEqual(parameters["imgsz"], 1280)
            self.assertEqual(parameters["device"], "cpu")

    def test_detector_passes_requested_device_to_yolo(self):
        """CPU 传 cpu，GPU 在可用时传 0；不可用必须报错而不是静默退回 CPU。"""
        weights = Mock(spec=Path)
        weights.is_file.return_value = True
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("footballer_analyzer.detection.YOLO") as model_type, patch("footballer_analyzer.detection.sv.Detections.from_ultralytics"):
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "GPU"):
                    YoloDetector(weights, device="gpu")
        with patch("footballer_analyzer.detection.YOLO") as model_type, patch("footballer_analyzer.detection.sv.Detections.from_ultralytics"), patch("torch.cuda.is_available", return_value=True):
            detector = YoloDetector(weights, device="gpu")
            detector.detect(image)
            self.assertEqual(model_type.return_value.predict.call_args.kwargs["device"], 0)
        for invalid in ["cuda", "dml", "gpu0", ""]:
            with self.subTest(device=invalid), self.assertRaises(ValueError):
                resolve_device(invalid)
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_config_validates_tracking_parameters(self):
        """检测和激活阈值独立保存，非法缓冲与关联参数应被拒绝。"""
        payload = {"video": "video.mp4", "weights": "model.pt", "match_id": "test", "clip_id": "test"}
        config = config_from_dict(payload)
        self.assertLess(config.confidence, config.track_activation_threshold)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config_from_dict({**payload, "device": "gpu"}).device, "gpu")
        with self.assertRaisesRegex(ValueError, "device"):
            config_from_dict({**payload, "device": "dml"})
        for key, invalid in [("nms_iou", 1), ("nms_iou", float("nan")), ("lost_track_buffer_s", 0), ("minimum_matching_threshold", -1), ("track_activation_threshold", 0.1), ("track_activation_threshold", 0.9)]:
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                config_from_dict({**payload, key: invalid})

    def test_pipeline_preserves_unmatched_detections_without_filling_gaps(self):
        """缓存低分噪声供复查；缺帧不伪造轨迹，恢复观测沿用原编号。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "source.avi"
            video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
            for frame_index in range(4):
                video_writer.write(np.full((48, 64, 3), frame_index * 20, dtype=np.uint8))
            video_writer.release()
            weights = root / "unused.pt"
            weights.write_bytes(b"mock local weights")
            strong = people([[10, 5, 25, 40]], [0.9])
            first = people([[10, 5, 25, 40], [40, 5, 55, 40]], [0.9, 0.15])
            low = people([[10, 5, 25, 40], [40, 5, 55, 40]], [0.15, 0.15])
            noise = people([[40, 5, 55, 40]], [0.15])
            with patch("footballer_analyzer.pipeline.YoloDetector") as detector_type:
                detector_type.return_value.detect.side_effect = [strong, first, low, noise, strong]
                run_dir = track(Config(video_path, weights, root, root / "players.yaml", "test", "test",
                                       tracking_max_cost=.75, tracking_position_tolerance=1.5,
                                       tracking_new_threshold=.35))
            raw_rows = list(read_rows(run_dir / "detections.parquet"))
            track_rows = list(read_rows(run_dir / "tracks.parquet"))
            self.assertEqual(len(raw_rows), 6)
            self.assertEqual([row["frame_index"] for row in track_rows], [0, 1, 3])
            self.assertEqual(len({row["track_id"] for row in track_rows}), 1)
            self.assertEqual(read_json(run_dir / "manifest.json")["detection_count"], 6)
            self.assertEqual(read_json(run_dir / "association_job.json")["status"], "complete")
            self.assertEqual(read_json(run_dir / "association_live.json")["status"], "reconciled")
            evidence = read_json(run_dir / "association_evidence.json")
            self.assertEqual(set(evidence["samples"]), {f"{track_rows[0]['track_id']}:0"})
            self.assertEqual(read_json(run_dir / "field_review.json")["analysis"]["nodes"][0]["field_frames"], [0, 1, 3])


if __name__ == "__main__":
    unittest.main()
