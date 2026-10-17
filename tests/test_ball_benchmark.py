"""验证切片还原与去重，防止切片检出被错误映射到原视频。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


SPEC = importlib.util.spec_from_file_location("bench_ball", Path(__file__).resolve().parents[1] / "scripts/bench_ball.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def prediction(bounds):
    """构造仅含单个局部球框的模型返回值。"""
    boxes = SimpleNamespace(xyxy=Mock(), conf=Mock())
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([bounds])
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
    return SimpleNamespace(boxes=boxes)


class BallBenchmarkTests(unittest.TestCase):
    """用边界球与相邻球场景验证几何处理。"""

    def test_tile_offsets_return_original_coordinates(self):
        """第二切片的局部球框应加上切片偏移，非零球类别仍可用。"""
        model = Mock(names={4: "ball", 0: "person"})
        model.predict.side_effect = [[prediction([10, 20, 20, 30])], [prediction([10, 20, 20, 30])]]
        candidates = BENCH.detect(model, np.zeros((100, 180, 3), dtype=np.uint8), 100, 100, 0.2, 0.05)
        self.assertEqual([entry[:4] for entry in candidates], [[10, 20, 20, 30], [90, 20, 100, 30]])
        self.assertEqual(model.predict.call_args.kwargs["classes"], [4])

    def test_overlap_deduplicates_only_same_object(self):
        """重复球框只保留高分，邻近但不重叠的候选仍保留给时序筛选。"""
        candidates = [[10, 10, 20, 20, 0.5], [11, 10, 21, 20, 0.9], [25, 10, 35, 20, 0.7]]
        self.assertEqual(BENCH.suppress_duplicates(candidates), [candidates[1], candidates[2]])

    def test_tiles_cover_right_edge_and_small_image(self):
        """非整除图宽与小图均有覆盖，不出现重复末片。"""
        origins = BENCH.tile_origins(1920, 640, 0.2)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1] + 640, 1920)
        self.assertTrue(all(right <= left + 640 for left, right in zip(origins, origins[1:])))
        self.assertEqual(BENCH.tile_origins(320, 640, 0.2), [0])


if __name__ == "__main__":
    unittest.main()
