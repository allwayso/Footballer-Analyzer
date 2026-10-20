"""验证缓存重跟踪的隔离、摘要保护及未知身份不能伪装通过验收。"""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from footballer_analyzer.config import Config
from footballer_analyzer.retracking import input_snapshot, prepare_colors, retrack
from footballer_analyzer.storage import DETECTION_SCHEMA, TRACK_SCHEMA, RowWriter, file_hash, read_json, write_json
from footballer_analyzer.tracking_evaluation import freeze_reference, evaluate_tracking


class RetrackingTests(unittest.TestCase):
    """使用本地微型视频，确保回放无需权重且不迁移人工身份。"""

    def make_source(self, root):
        """建立有空帧、低分框和旧轨迹的完整只读输入。"""
        video = root / 'source.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
        for _ in range(8):
            writer.write(np.full((48, 64, 3), 100, np.uint8))
        writer.release()
        run = root / 'old'
        run.mkdir()
        config = Config(video, root / 'absent.pt', root, root / 'players.yaml', 'm', 'c')
        write_json(run / 'manifest.json', {'run_id': 'old', 'config': config.snapshot(), 'start_frame': 0, 'end_frame': 8,
                                          'video': {'fps': 25, 'width': 64, 'height': 48}, 'video_sha256': file_hash(video)})
        with RowWriter(run / 'detections.parquet', DETECTION_SCHEMA) as raw, RowWriter(run / 'tracks.parquet', TRACK_SCHEMA) as tracked:
            for frame in (0, 1, 3, 4, 5, 6, 7):
                row = {'frame_index': frame, 'timestamp_s': frame / 25, 'bbox_x1': 10., 'bbox_y1': 5., 'bbox_x2': 25.,
                       'bbox_y2': 40., 'confidence': .8 if frame < 2 else .15, 'class_id': 0}
                raw.append(row)
                tracked.append({**row, 'track_id': 55, 'team_id': 'unknown'})
        write_json(run / 'field_review.json', {'operations': [{'fake_manual_history': True}]})
        return run

    def test_isolated_cache_replay_and_repeatability(self):
        """两次回放保持映射及帧序一致，未知身份计入未正确，原缓存与历史不变。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            signature = input_snapshot(source)
            history_hash = file_hash(source / 'field_review.json')
            colors = prepare_colors(source, root / 'colors.npz')
            reference = root / 'reference'
            freeze_reference(source, reference)
            first = retrack(source, root / 'new', color_cache=colors, export_video=False)
            second = retrack(source, root / 'new', color_cache=colors, export_video=False)
            self.assertEqual((first / 'tracking_links.jsonl').read_text(), (second / 'tracking_links.jsonl').read_text())
            self.assertEqual(input_snapshot(source), signature)
            self.assertEqual(file_hash(source / 'field_review.json'), history_hash)
            self.assertEqual(read_json(first / 'field_review.json')['operations'], [])
            summary = evaluate_tracking(first, source, reference)
            self.assertEqual(summary['observation_count'], 7)
            self.assertEqual(summary['baseline_observation_proxy_coverage'], 1)
            self.assertEqual(summary['identity_error_rate'], 1)
            self.assertFalse(summary['truth_complete'])
            self.assertEqual(summary['acceptance'], 'pending_identity_truth')
            self.assertFalse(summary['post_association_applied'])

    def test_changed_input_rejects_color_cache(self):
        """输入清单发生变化后不可复用旧颜色缓存。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            cache = prepare_colors(source, root / 'colors.npz')
            manifest = read_json(source / 'manifest.json')
            manifest['note'] = 'changed'
            write_json(source / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, '不匹配'):
                prepare_colors(source, cache)


if __name__ == '__main__':
    unittest.main()
