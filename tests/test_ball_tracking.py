"""验证球的跨帧筛选可拒绝孤立跳点，且不将漏检伪装为观测。"""

import unittest

from footballer_analyzer.ball_tracking import select_ball_candidates


def ball(center_x, center_y, confidence=0.6):
    """构造十像素球框。"""
    return [center_x - 5, center_y - 5, center_x + 5, center_y + 5, confidence]


class BallTrackingTests(unittest.TestCase):
    """使用独立已知轨迹检查关联与时间语义。"""

    def test_rejects_isolated_high_confidence_jump(self):
        """真实球连续移动，单帧远处高分杂物不得抢占轨迹。"""
        records = [{"frame": frame, "candidates": [ball(100 + frame * 12, 200)]} for frame in range(12)]
        records[5]["candidates"].insert(0, ball(1200, 800, 0.99))
        selected = select_ball_candidates(records, 25, (1920, 1080))
        self.assertEqual(selected[5], ball(160, 200))
        self.assertTrue(all(selected))

    def test_missing_stays_missing_and_can_recover(self):
        """短缺口保持 None，恢复后使用真实框。"""
        records = [{"frame": frame, "candidates": [] if frame in (4, 5) else [ball(100 + frame * 8, 200)]} for frame in range(10)]
        selected = select_ball_candidates(records, 25, (1920, 1080))
        self.assertEqual(selected[4:6], [None, None])
        self.assertEqual(selected[6], ball(148, 200))

    def test_gap_expiry_reacquires_distant_ball(self):
        """长时间中断后允许远处重新捕获，不把源帧间隔当成一次采样。"""
        records = [{"frame": 0, "candidates": [ball(100, 200)]}, {"frame": 30, "candidates": [ball(1500, 800)]}]
        self.assertEqual(select_ball_candidates(records, 25, (1920, 1080)), [ball(100, 200), ball(1500, 800)])

    def test_empty_and_invalid_inputs(self):
        """空检测不产生轨迹，非法时间与候选及时拒绝。"""
        self.assertEqual(select_ball_candidates([], 25, (1920, 1080)), [])
        self.assertEqual(select_ball_candidates([{"frame": 0, "candidates": []}], 25, (1920, 1080)), [None])
        for records in ([{"frame": -1, "candidates": []}], [{"frame": 0, "candidates": [[0, 0, 10, 10, float("nan")]]}], [{"frame": 2, "candidates": []}, {"frame": 1, "candidates": []}]):
            with self.assertRaises(ValueError):
                select_ball_candidates(records, 25, (1920, 1080))


if __name__ == "__main__":
    unittest.main()
