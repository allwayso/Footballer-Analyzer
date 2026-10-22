"""独立分析页面与历史足球结果版本的兼容性测试。"""

from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from unittest.mock import patch

from footballer_analyzer.ball_pipeline import load_ball_result
from footballer_analyzer.server import ReviewApp, handler_for
from footballer_analyzer.storage import write_json


class BallPageTests(unittest.TestCase):
    """验证页面路由、足球版本选择和显式任务入口。"""

    def test_versions_routes_and_explicit_job(self):
        """仅暴露完成版本，保留缺失观测，并正确传递启动参数。"""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'run_test'
            write_json(run / 'manifest.json', {'status': 'complete'})
            frames = [{'frame': 0, 'candidates': [], 'selected': None}]
            for name, status in [('ball_old', 'complete'), ('ball_failed', 'failed')]:
                write_json(run / 'balls' / name / 'manifest.json', {
                    'result_id': name, 'model_id': 'football_v11n', 'status': status,
                })
                write_json(run / 'balls' / name / 'frames.json', frames)
            self.assertEqual(load_ball_result(run)['manifest']['result_id'], 'ball_old')
            self.assertIsNone(load_ball_result(run, version='ball_failed'))
            self.assertIsNone(load_ball_result(run, model_id='coco_v8n'))
            app = ReviewApp(temporary)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                for page in ('identity', 'tracking', 'report'):
                    for suffix in ('', '.css', '.js'):
                        with urlopen(base + '/' + page + suffix) as response:
                            self.assertEqual(response.status, 200)
                with urlopen(base + '/api/runs/run_test/ball?version=ball_old') as response:
                    self.assertEqual(json.load(response)['frames'], frames)
                with urlopen(base + '/api/ball-job') as response:
                    self.assertEqual(json.load(response)['status'], 'idle')
                with urlopen(base + '/api/ball-models') as response:
                    self.assertTrue(json.load(response)['models'])
                with patch.object(app, 'start_ball', return_value={'status': 'running'}) as start:
                    request = Request(base + '/api/runs/run_test/ball', data=b'{"model_id":"football_v11n"}', headers={'Content-Type': 'application/json'})
                    with urlopen(request) as response:
                        self.assertEqual(json.load(response)['status'], 'running')
                    start.assert_called_once_with('run_test', {'model_id': 'football_v11n'})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
