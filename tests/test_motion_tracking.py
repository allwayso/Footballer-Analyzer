"""独立验证运动颜色匹配、真实观测缓冲与检测直接映射。"""

from dataclasses import replace
import unittest

import numpy as np
import supervision as sv

from footballer_analyzer.motion_tracking import MotionColorTracker, MotionConfig
from footballer_analyzer.video import FramePacket


def detections(boxes, scores=None):
    """构造不依赖检测网络的真实框输入。"""
    return sv.Detections(xyxy=np.array(boxes, float).reshape(-1, 4),
                         confidence=np.array(scores if scores is not None else [0.8] * len(boxes)),
                         class_id=np.zeros(len(boxes), int))


class MotionTrackingTests(unittest.TestCase):
    """覆盖原先高低分断层、重叠框和延迟确认的边界。"""

    def replay(self, frames, config=None, stride=1, calibration=None):
        """提交合成帧并刷新结尾，返回全部源帧结果。"""
        tracker = MotionColorTracker(25, stride, config, calibration)
        output = []
        for frame, boxes, scores, colors in frames:
            output.extend(tracker.submit(FramePacket(frame, frame / 25, None), detections(boxes, scores), colors))
        output.extend(tracker.flush())
        return tracker, {result.packet.frame_index: result.detections for result in output}

    def test_low_score_zero_iou_and_real_observation_backfill(self):
        """已有速度可延续零IoU的低分检测，确认时补回首帧而不填漏检。"""
        frames = [(i, [[i * 5, 0, i * 5 + 10, 50]], [0.8 if i < 2 else 0.251], [None]) for i in range(5)]
        frames += [(5, [], [], []), (6, [[30, 0, 40, 50]], [0.15], [None])]
        tracker, output = self.replay(frames)
        self.assertEqual(sorted(output), list(range(7)))
        self.assertEqual([len(output[i]) for i in range(7)], [1, 1, 1, 1, 1, 0, 1])
        self.assertEqual({int(row.tracker_id[0]) for row in output.values() if len(row)}, {1})
        self.assertEqual(tracker.stats['birth_attempt'], 1)

    def test_order_reverse_keeps_direct_indices_and_colors(self):
        """检测顺序交换时保持人物身份与原检测下标一一对应。"""
        red, blue = np.eye(44)[0], np.eye(44)[20]
        frames = [(0, [[0, 0, 15, 60], [25, 0, 40, 60]], None, [red, blue]),
                  (1, [[0, 0, 15, 60], [25, 0, 40, 60]], None, [red, blue]),
                  (2, [[24, 0, 39, 60], [1, 0, 16, 60]], None, [blue, red])]
        _, output = self.replay(frames)
        self.assertEqual(output[2].tracker_id.tolist(), output[0].tracker_id[::-1].tolist())
        self.assertEqual(output[2].data['source_detection_index'].tolist(), [0, 1])

    def test_multibody_box_cannot_replace_two_people(self):
        """覆盖两个紧框的大框不产生第三个身份，两个真实人物仍有输出。"""
        boxes = [[0, 0, 15, 35], [0, 40, 15, 75], [0, 0, 15, 75]]
        tracker, output = self.replay([(i, boxes, [0.8, 0.8, 0.9], [None] * 3) for i in range(8)])
        self.assertTrue(all(len(row) == 2 for row in output.values()))
        self.assertEqual(tracker.stats['confirmed'], 2)
        self.assertEqual(output[0].data['source_detection_index'].tolist(), [0, 1])

    def test_image_gap_expiry_and_pending_singleton(self):
        """无可靠参照的画面运动不跨一秒续接，单次出现不伪造确认。"""
        box = [[0, 0, 20, 60]]
        frames = [(0, box, None, [None]), (1, box, None, [None])]
        frames += [(i, [], [], []) for i in range(2, 30)]
        frames += [(30, box, None, [None]), (31, box, None, [None]), (32, [[200, 0, 220, 60]], None, [None])]
        _, output = self.replay(frames)
        self.assertNotEqual(output[0].tracker_id.tolist(), output[30].tracker_id.tolist())
        self.assertEqual(len(output[32]), 0)

    def test_sparse_sampling_and_flush_order(self):
        """0.12秒采样确认并补写首帧，输出顺序稳定且重复flush为空。"""
        tracker, output = self.replay([(i, [[0, 0, 20, 60]], None, [None]) for i in (0, 3, 6)], stride=3)
        self.assertEqual(list(output), [0, 3, 6])
        self.assertTrue(all(len(row) == 1 for row in output.values()))
        self.assertEqual(tracker.flush(), [])

    def test_outside_projection_falls_back_without_losing_person(self):
        """人物越过可靠投影边界时回退短期图像运动，不能直接断轨。"""
        calibration = {'reviewed': True, 'start_frame': 0, 'end_frame': 100, 'height': 300,
                       'homography': [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
                       'roi': [[0, 0], [30, 0], [30, 200], [0, 200]],
                       'pitch_length_m': 105, 'pitch_width_m': 68}
        _, output = self.replay([(i, [[i * 4, 0, i * 4 + 20, 60]], None, [None]) for i in range(10)], calibration=calibration)
        self.assertEqual({int(row.tracker_id[0]) for row in output.values()}, {1})

    def test_invalid_parameters_and_times(self):
        """错误输入不能悄悄进入关联或生成无效代价。"""
        for value in (True, '0', float('nan'), -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MotionConfig(competition_margin=value)
        with self.assertRaises(ValueError):
            replace(MotionConfig(), delay_s=0.3)
        tracker = MotionColorTracker(25, 1)
        tracker.submit(FramePacket(0, 0, None), detections([]), [])
        with self.assertRaises(ValueError):
            tracker.submit(FramePacket(0, 0, None), detections([]), [])

    def test_low_first_observation_and_confirmation_deadline(self):
        """低分首观测可等待后续高分确认；孤立目标最多等待0.2秒。"""
        _, output = self.replay([(0, [[0, 0, 20, 60]], [.15], [None]),
                                 (1, [[1, 0, 21, 60]], [.8], [None])])
        self.assertEqual(output[0].tracker_id.tolist(), output[1].tracker_id.tolist())
        self.assertEqual(len(output[0]), 1)
        tracker = MotionColorTracker(25, 1)
        released = []
        for frame in range(7):
            released += [(frame, item) for item in tracker.submit(
                FramePacket(frame, frame / 25, None),
                detections([[0, 0, 20, 60]] if frame == 0 else []), [None] if frame == 0 else [])]
        self.assertEqual(next(frame for frame, item in released if item.packet.frame_index == 0), 5)
        self.assertTrue(all(len(item.detections) == 0 for _, item in released))

    def test_crossing_players_with_same_and_different_shirts(self):
        """已建立的相反速度在交叉后仍延续人物，不依赖同队颜色区分。"""
        red, blue = np.eye(44)[0], np.eye(44)[20]
        for colors in ([red, red], [red, blue]):
            with self.subTest(same_shirt=bool(np.array_equal(*colors))):
                frames = []
                for frame in range(22):
                    left, right = frame * 3, 63 - frame * 3
                    boxes = [[left, 0, left + 12, 60], [right, 1, right + 12, 61]]
                    reverse = frame % 2 == 1
                    frames.append((frame, boxes[::-1] if reverse else boxes, None,
                                   colors[::-1] if reverse else colors))
                _, output = self.replay(frames)
                identities = output[0].tracker_id.tolist()
                for frame, row in output.items():
                    self.assertEqual(row.tracker_id.tolist(), identities[::-1] if frame % 2 else identities)

    def test_stop_turn_missing_color_and_new_person(self):
        """急停变向和颜色短缺不强制换号，新人须经两次观测独立确认。"""
        red = np.eye(44)[0]
        positions = [0, 2, 4, 6, 8, 8, 8, 8, 6, 4, 2, 0]
        frames = [(frame, [[x, 0, x + 20, 70]] + ([[200, 0, 220, 70]] if frame >= 9 else []),
                   None, [red if frame < 5 else None] + ([None] if frame >= 9 else []))
                  for frame, x in enumerate(positions)]
        _, output = self.replay(frames)
        self.assertEqual({int(row.tracker_id[0]) for row in output.values()}, {1})
        self.assertEqual(output[9].tracker_id.tolist(), [1, 2])

    def test_unreachable_scene_change_does_not_reuse_identity(self):
        """切镜后远处新人物不能仅因球衣一致抢走旧编号。"""
        color = np.eye(44)[0]
        frames = [(i, [[0 if i < 5 else 500, 0, 20 if i < 5 else 520, 60]], None, [color]) for i in range(10)]
        _, output = self.replay(frames)
        self.assertNotEqual(output[0].tracker_id.tolist(), output[5].tracker_id.tolist())


if __name__ == '__main__':
    unittest.main()
