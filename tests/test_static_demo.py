"""校验静态导出中的隐私边界、过期拒绝和跨页联动。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from footballer_analyzer.static_demo import encoded, export_demo, motion_page, public_history


class StaticDemoTests(unittest.TestCase):
    """保护原始运行，避免向公开站泄露机器配置。"""

    def test_history_whitelist(self):
        """历史保留自动执行依据和撤销状态，排除配置路径。"""
        result = public_history([{"operation_id": "x", "execution": "auto", "status": "undone", "score": .95,
                                  "batch_id": "batch", "input_signature": {"video": "private"}, "config": "private"}])
        self.assertEqual(result, [{"operation_id": "x", "execution": "auto", "status": "undone", "score": .95, "batch_id": "batch"}])

    def test_script_json_is_safe_and_finite(self):
        """脚本数据不能提前结束标签，也不能混入非有限值。"""
        self.assertNotIn("<", encoded({"name": "</script>"}))
        with self.assertRaises(ValueError):
            encoded({"bad": float("nan")})

    def test_stale_report_never_creates_output(self):
        """标注变化后必须拒绝旧报告，且不触碰导出目录。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("footballer_analyzer.static_demo.snapshot_inputs", return_value={"bindings": "new"}), patch("footballer_analyzer.static_demo.read_json", return_value={"bindings": "old"}):
                with self.assertRaisesRegex(ValueError, "过期"):
                    export_demo(root / "run", root / "report", root / "output")
            self.assertFalse((root / "output").exists())

    def test_motion_template_uses_relative_navigation(self):
        """独立报告跨页面时携带身份和源帧，不保留本地 API。"""
        source = Path(__file__).parents[1] / "src/footballer_analyzer/web/player-motion-report.html"
        result = motion_page(source.read_text(encoding="utf-8"), {"players": []})
        for forbidden in ("127.0.0.1", "${origin}", 'href="/field-review"', "__MOTION_DATA__"):
            self.assertNotIn(forbidden, result)
        self.assertIn("player:selected.player_id", result)
        self.assertIn("incoming.get('frame')", result)
        self.assertIn("staticLink('index.html')", result)


if __name__ == "__main__":
    unittest.main()
