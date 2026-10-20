"""验证可视化副本的融合身份继承、去重与有效时间守恒。"""

import unittest

from footballer_analyzer.identity import resolve_identity
from footballer_analyzer.metrics import calculate_motion, summarize
from footballer_analyzer.motion_visualization import player_payload, resolved_rows


def observation(track_id, frame, confidence=0.8):
    """构造位于可靠区域内、匀速缓慢移动的合成观测。"""
    x = 10 + frame * 0.1
    return {"track_id": track_id, "frame_index": frame, "timestamp_s": frame / 10,
            "bbox_x1": x - 1, "bbox_x2": x + 1, "bbox_y1": 10, "bbox_y2": 20,
            "confidence": confidence, "team_id": "unknown"}


def binding(track_id, start, end, player_id="p"):
    """构造显式半开区间身份绑定。"""
    return {"track_id": track_id, "start_frame": start, "end_frame": end,
            "action": "bind", "player_id": player_id, "team_id": "home"}


def segment(track_id, start, end):
    """构造同一有效融合组内的场内片段，末帧为包含边界。"""
    return {"track_id": track_id, "start_frame": start, "end_frame": end, "merge_group_id": "group"}


class MotionVisualizationTests(unittest.TestCase):
    """覆盖不修改输入的身份派生以及不跨缺口补距离的统计行为。"""

    def test_merged_identity_and_explicit_duplicate_precedence(self):
        """轨迹融合后继承身份，重叠帧即使置信度更高也优先显式绑定。"""
        rows = [observation(1, 0), observation(1, 1, 0.4), observation(2, 1, 0.99), observation(2, 2)]
        original = [binding(1, 0, 2)]
        selected, effective, diagnostics = resolved_rows(rows, original, [segment(1, 0, 1), segment(2, 1, 2)])
        self.assertEqual(len(original), 1)
        self.assertEqual([(row["track_id"], row["frame_index"]) for row in selected], [(1, 0), (1, 1), (2, 2)])
        self.assertEqual(resolve_identity(selected[-1], effective)[0], "p")
        self.assertEqual(len(diagnostics["duplicate_observations"]), 1)
        self.assertEqual(effective[-1]["end_frame"], 3)

    def test_explicit_exclusion_not_overwritten(self):
        """融合继承仅填补未绑定区间，保留中间显式排除。"""
        entries = [binding(1, 0, 1), {"track_id": 2, "start_frame": 3, "end_frame": 4, "action": "exclude"}]
        rows = [observation(2, frame) for frame in [2, 3, 4]]
        selected, effective, _ = resolved_rows(rows, entries, [segment(1, 0, 0), segment(2, 2, 4)])
        self.assertEqual([resolve_identity(row, effective)[0] for row in selected], ["p", None, "p"])
        self.assertEqual(resolve_identity(selected[1], effective)[3], "excluded")

    def test_conflicting_group_never_inherits(self):
        """融合组出现两个显式人物时，第三片段不得猜测身份。"""
        rows = [observation(3, 5)]
        entries = [binding(1, 0, 2, "a"), binding(2, 3, 5, "b")]
        _, effective, diagnostics = resolved_rows(rows, entries, [segment(1, 0, 1), segment(2, 3, 4), segment(3, 5, 7)])
        self.assertEqual(resolve_identity(rows[0], effective)[0], None)
        self.assertEqual(diagnostics["unresolved_groups"][0]["reason"], "identity_conflict")

    def test_gaps_do_not_add_time_and_heat_conserves_time(self):
        """两个分离片段只统计各自有效边，热图和速度分箱与汇总守恒。"""
        manifest = {"start_frame": 0, "end_frame": 10, "video": {"fps": 10},
                    "config": {"stride": 1, "speed_limit_mps": 12, "min_cutoff_hz": 1,
                               "filter_beta": 0, "match_id": "test", "clip_id": "test"}}
        calibration = {"start_frame": 0, "end_frame": 10, "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       "roi": [[0, 0], [100, 0], [100, 100], [0, 100]], "height": 100,
                       "pitch_length_m": 105, "pitch_width_m": 68}
        entries = [binding(1, 0, 2), binding(2, 5, 7)]
        rows = [observation(1, 0), observation(1, 1), observation(2, 5), observation(2, 6)]
        motions = list(calculate_motion(rows, manifest, calibration, entries))
        summary = summarize(motions, manifest, entries)
        self.assertAlmostEqual(summary[0]["valid_seconds"], 0.2)
        self.assertIsNone(motions[2]["delta_t_s"])
        data = player_payload(motions, summary, {"p": {"display_name": "p"}}, manifest, calibration)[0]
        self.assertAlmostEqual(sum(data["heat"]), 0.2)
        self.assertAlmostEqual(sum(row[1] for row in data["speed_bins"]), 0.2)
        self.assertAlmostEqual(data["speed_bins"][0][0], summary[0]["avg_speed_mps"] * 3.6, places=4)


if __name__ == "__main__":
    unittest.main()
