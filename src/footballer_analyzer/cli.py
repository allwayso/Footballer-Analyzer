"""解析跟踪、标定、性能测试、报告、历史查询与本地服务命令，统一返回退出码。"""

import argparse
import json
from pathlib import Path
import sys

from .config import ROOT, load_config


def main(argv=None):
    """解析 argv 并执行子命令；默认读取进程参数，成功返回 0，已处理错误返回 1。"""
    parser = argparse.ArgumentParser(description="FootballerAnalyzer · 足球轨迹与球场位置复盘")
    commands = parser.add_subparsers(dest="command", required=True)
    track_command = commands.add_parser("track", help="检测跟踪并保存原始轨迹")
    track_command.add_argument("--config", required=True)
    track_command.add_argument("--device", choices=["cpu", "gpu"], help="覆盖配置中的推理设备；gpu 需要 CUDA 或 ROCm 版 PyTorch")
    retrack_command = commands.add_parser("retrack", help="复用检测缓存进行运动颜色重跟踪，不运行YOLO")
    retrack_command.add_argument("--run", required=True, type=Path)
    retrack_command.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    retrack_command.add_argument("--algorithm", choices=["motion-color-v1", "bytetrack-legacy"], default="motion-color-v1")
    retrack_command.add_argument("--max-cost", type=float, default=0.99)
    retrack_command.add_argument("--position-tolerance", type=float, default=3.0)
    retrack_command.add_argument("--competition-margin", type=float, default=0)
    retrack_command.add_argument("--new-threshold", type=float, default=0.75, help="新轨迹确认阈值，默认0.75")
    retrack_command.add_argument("--color-cache", type=Path)
    retrack_command.add_argument("--no-video", action="store_true")
    retrack_command.add_argument("--without-calibration", action="store_true")
    benchmark_command = commands.add_parser("benchmark", help="重新执行选定流水线，测量并比较阶段速度")
    benchmark_command.add_argument("--config", required=True)
    benchmark_command.add_argument("--device", choices=["cpu", "gpu"], help="覆盖配置中的推理设备")
    benchmark_command.add_argument("--stages", nargs="+", choices=["tracking", "pitch", "ball"], default=["tracking", "pitch", "ball"])
    benchmark_command.add_argument("--repeat", type=int, default=1, help="正式测试轮数")
    benchmark_command.add_argument("--warmup", type=int, default=0, help="不计入结果的完整预热轮数")
    benchmark_command.add_argument("--ball-model", default="football_v11n_tile")
    benchmark_command.add_argument("--start-frame", type=int)
    benchmark_command.add_argument("--end-frame", type=int)
    benchmark_command.add_argument("--output-root", type=Path, default=ROOT / "outputs/bench")
    benchmark_command.add_argument("--baseline", type=Path, help="相同测试协议的旧 result.json")
    calibration_command = commands.add_parser("calibrate", help="从 YAML 对应点保存标定，也可在前端点选")
    calibration_command.add_argument("--run", required=True, type=Path)
    calibration_command.add_argument("--points", required=True, type=Path)
    lines_command = commands.add_parser("calibrate-lines", help="逐帧白线标定；已有运行或独立视频均可验收")
    lines_source = lines_command.add_mutually_exclusive_group(required=True)
    lines_source.add_argument("--run", type=Path)
    lines_source.add_argument("--video", type=Path)
    lines_command.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    lines_command.add_argument("--pitch-length", type=float)
    lines_command.add_argument("--pitch-width", type=float)
    lines_command.add_argument("--start-frame", type=int, default=0)
    lines_command.add_argument("--end-frame", type=int)
    report_command = commands.add_parser("report", help="复用缓存轨迹生成个人跑动报告")
    report_command.add_argument("--run", required=True, type=Path)
    history_command = commands.add_parser("history", help="查询球员在选定报告中的记录")
    history_command.add_argument("--player", required=True)
    history_command.add_argument("--reports", nargs="+", required=True, type=Path)
    serve_command = commands.add_parser("serve", help="启动本地球员追踪与球场位置调试台")
    serve_command.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    serve_command.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            from dataclasses import replace as dataclass_replace

            from .pipeline import track

            config = load_config(args.config)
            if args.device:
                config = dataclass_replace(config, device=args.device)
            track(config)
        elif args.command == "retrack":
            from .motion_tracking import MotionConfig
            from .retracking import retrack

            parameters = MotionConfig(max_cost=args.max_cost, position_tolerance=args.position_tolerance,
                                      competition_margin=args.competition_margin, new_threshold=args.new_threshold)
            retrack(args.run, args.output_root, parameters, args.color_cache, not args.no_video,
                    args.algorithm, not args.without_calibration)
        elif args.command == "benchmark":
            from dataclasses import replace
            from .benchmark import print_benchmark, run_benchmark

            config = load_config(args.config)
            overrides = {key: getattr(args, key) for key in ("start_frame", "end_frame", "device") if getattr(args, key) is not None}
            result = run_benchmark(replace(config, **overrides), args.stages, args.repeat, args.warmup,
                                   args.ball_model, args.output_root, args.baseline)
            print_benchmark(result)
        elif args.command == "calibrate":
            from .calibration import save_calibration
            from .storage import read_yaml

            save_calibration(args.run.resolve(), read_yaml(args.points))
        elif args.command == "calibrate-lines":
            from .white_line_pipeline import calibrate_run, create_line_run

            if args.run:
                if args.start_frame != 0 or args.end_frame is not None:
                    raise ValueError("--run 使用已有分析区间；--start-frame/--end-frame 仅适用于 --video")
                calibrate_run(args.run.resolve(), args.pitch_length, args.pitch_width)
            else:
                create_line_run(args.video, args.output_root, args.pitch_length if args.pitch_length is not None else 105, args.pitch_width if args.pitch_width is not None else 68, args.start_frame, args.end_frame)
        elif args.command == "report":
            from .reporting import create_report

            create_report(args.run.resolve())
        elif args.command == "history":
            from .reporting import history

            print(json.dumps(history(args.player, args.reports), ensure_ascii=False, indent=2))
        elif args.command == "serve":
            from .server import serve

            serve(args.output_root, args.port)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
