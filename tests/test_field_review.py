"""场内片段人工校对与融合历史的回归测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from footballer_analyzer.server import ReviewApp, field_segments, merge_groups
from footballer_analyzer.storage import read_json, write_json


def row(track_id, frame_index, outside=False):
    """构造带源帧与场外标志的合成观测。"""
    return {
        "track_id": track_id,
        "frame_index": frame_index,
        "timestamp_s": frame_index / 10,
        "team_id": "cluster_0",
        "outside": outside,
    }


class FieldReviewTests(unittest.TestCase):
    """验证场内片段筛选、融合校验与持久化撤销。"""
    def test_segments_bridge_short_gaps_split_long_gaps_and_exclude_outside(self):
        """短缺口保持同段，长缺口切段，场外帧不纳入片段。"""
        manifest = {"video": {"fps": 10}, "config": {"stride": 1, "lost_track_buffer_s": 1}}
        rows = [row(1, frame, outside=(frame == 2)) for frame in [0, 1, 2, 4, 30, 31]]

        def project(observation, calibration):
            """模拟场地投影对场外观测的过滤。"""
            return (1, 1, "outside_roi") if observation["outside"] else (1, 1, None)

        with patch("footballer_analyzer.server.project_observation", side_effect=project):
            segments = field_segments(rows, manifest, {"reviewed": True})

        self.assertEqual([(item["start_frame"], item["end_frame"]) for item in segments], [(0, 4), (30, 31)])
        self.assertEqual([item["observation_count"] for item in segments], [3, 2])
        self.assertEqual([item["field_frames"] for item in segments], [[0, 1, 4], [30, 31]])

    def test_unreviewed_calibration_does_not_create_review_records(self):
        """未验收场地不生成可融合片段。"""
        manifest = {"video": {"fps": 10}, "config": {"stride": 1, "lost_track_buffer_s": 1}}
        with patch("footballer_analyzer.server.project_observation", return_value=(1, 1, None)):
            segments = field_segments([row(1, 0)], manifest, {"reviewed": False})
        self.assertEqual(segments, [])

    def test_merge_groups_and_undo_history_are_persisted(self):
        """融合和撤销写入历史，撤销后按剩余操作重算分组。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run_test"
            run_dir.mkdir()
            write_json(run_dir / "manifest.json", {"run_id": "run_test"})
            app = ReviewApp(root)
            segments = [
                {"record_id": "a", "track_id": 1, "merge_group_id": "a", "merge_group_size": 1, "field_frames": [4]},
                {"record_id": "b", "track_id": 2, "merge_group_id": "b", "merge_group_size": 1, "field_frames": [8]},
                {"record_id": "c", "track_id": 3, "merge_group_id": "c", "merge_group_size": 1, "field_frames": [12]},
            ]
            with patch.object(app, "data", return_value={"field_segments": segments}):
                result = app.merge_field_records("run_test", {"source_record_id": "b", "target_record_id": "a", "target_frame": 4})
            saved = read_json(run_dir / "field_review.json")
            self.assertEqual(saved["operations"][0]["target_frame"], 4)
            history = merge_groups([dict(item) for item in segments], saved["operations"])
            self.assertEqual(history[0]["status"], "active")

            with patch.object(app, "data", return_value={"field_segments": segments}):
                app.merge_field_records("run_test", {"source_record_id": "c", "target_record_id": "a", "target_frame": 4})
            app.undo_field_merge("run_test", {"operation_id": result["operation"]["operation_id"]})
            saved = read_json(run_dir / "field_review.json")
            reviewed_segments = [dict(item) for item in segments]
            history = merge_groups(reviewed_segments, saved["operations"])

            self.assertEqual([entry["status"] for entry in history], ["undone", "active"])
            self.assertEqual([item["merge_group_size"] for item in reviewed_segments], [2, 1, 2])

    def test_merge_rejects_same_or_unknown_record(self):
        """拒绝自融合和已不存在的片段。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run_test"
            run_dir.mkdir()
            write_json(run_dir / "manifest.json", {"run_id": "run_test"})
            app = ReviewApp(root)
            segments = [{"record_id": "a", "track_id": 1}]
            with patch.object(app, "data", return_value={"field_segments": segments}):
                with self.assertRaisesRegex(ValueError, "不同"):
                    app.merge_field_records("run_test", {"source_record_id": "a", "target_record_id": "a"})
                with self.assertRaisesRegex(ValueError, "记录已变化"):
                    app.merge_field_records("run_test", {"source_record_id": "a", "target_record_id": "missing"})

    def test_merge_requires_target_frame_present_in_destination_record(self):
        """目标必须在输入源帧实际出现，布尔值不能充当帧号。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run_test"
            run_dir.mkdir()
            write_json(run_dir / "manifest.json", {"run_id": "run_test"})
            app = ReviewApp(root)
            segments = [
                {"record_id": "a", "track_id": 1, "merge_group_id": "a", "field_frames": [4]},
                {"record_id": "b", "track_id": 2, "merge_group_id": "b", "field_frames": [8]},
            ]
            payload = {"source_record_id": "b", "target_record_id": "a", "target_frame": 8}
            with patch.object(app, "data", return_value={"field_segments": segments}):
                with self.assertRaisesRegex(ValueError, "没有所选目标轨迹"):
                    app.merge_field_records("run_test", payload)
                payload["target_frame"] = True
                with self.assertRaisesRegex(ValueError, "有效的融合目标帧"):
                    app.merge_field_records("run_test", payload)


if __name__ == "__main__":
    unittest.main()
