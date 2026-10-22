"""用真实流水线重新测量人物、白线和足球阶段，隔离产物并支持优化前后比较。"""

from dataclasses import replace
from pathlib import Path
import hashlib
import math
import platform
import statistics
import subprocess
import sys

import cv2

from .ball_pipeline import DEFAULT_MODEL, model_preset, track_ball_run
from .config import Config, ROOT, config_from_dict
from .detection import gpu_name
from .performance import PerformanceRecorder
from .pipeline import track
from .storage import file_hash, new_id, read_json, versions, write_json
from .video import inspect_video
from .white_line_pipeline import calibrate_run

STAGES = ("tracking", "pitch", "ball")


def summarize(recorder):
    """按阶段汇总重复试验，平均 FPS 使用总帧数除以总耗时。"""
    summaries = {}
    for stage in dict.fromkeys(metric.stage for metric in recorder.metrics):
        metrics = [metric for metric in recorder.metrics if metric.stage == stage]
        elapsed = [metric.elapsed_s for metric in metrics]
        frames = sum(metric.frames for metric in metrics)
        total_s = sum(elapsed)
        summaries[stage] = {
            "samples": len(metrics), "frames": frames, "elapsed_s": total_s,
            "fps": frames / total_s, "ms_per_frame": total_s * 1000 / frames,
            "median_elapsed_s": statistics.median(elapsed),
            "min_elapsed_s": min(elapsed), "max_elapsed_s": max(elapsed),
        }
    return summaries


def code_version():
    """记录 Git 提交与 Python 源码摘要，让未提交的优化也可区分。"""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/footballer_analyzer").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {"git_commit": commit, "python_source_sha256": digest.hexdigest()}


def compare_results(current, baseline):
    """比较相同输入协议的平均阶段速度；不同素材、配置或权重拒绝比较。"""
    if baseline.get("status") != "complete" or baseline.get("protocol") != current["protocol"]:
        raise ValueError("基线未完成或测试协议不同；请使用相同素材、区间、配置、模型及预热次数")
    for stage in current["protocol"]["stages"]:
        fps = baseline.get("summary", {}).get(stage, {}).get("fps")
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"基线缺少有效的 {stage} 阶段速度")
    comparisons = {}
    for stage, summary in current["summary"].items():
        previous = baseline["summary"][stage]
        comparisons[stage] = {
            "baseline_fps": previous["fps"], "current_fps": summary["fps"],
            "speedup": summary["fps"] / previous["fps"],
        }
    return comparisons


def prepare_run(run_dir, config, info, video_hash):
    """创建独立测试清单，让白线或足球单测不必先执行人物检测。"""
    manifest = {
        "schema_version": 1, "run_id": run_dir.name, "status": "complete",
        "kind": "benchmark", "config": config.snapshot(), "video": info.snapshot(),
        "video_sha256": video_hash, "start_frame": config.start_frame, "end_frame": config.end_frame,
    }
    write_json(run_dir / "manifest.json", manifest)


def run_benchmark(config: Config, stages=STAGES, repeat=1, warmup=0, ball_model=DEFAULT_MODEL, output_root=None, baseline=None):
    """重新执行选定流水线并返回结构化结果，每轮均新建运行且不复用旧检测。

    repeat 为正式轮数，warmup 为不计入结果的完整预热轮数。阶段采用现有流水线计时
    边界，包含该阶段的解码与产物写入；本组件另保存独立目录和完整输入协议。
    """
    stages = tuple(stages)
    if not stages or len(set(stages)) != len(stages) or any(stage not in STAGES for stage in stages):
        raise ValueError("阶段必须从 tracking、pitch、ball 中选择且不可重复")
    if type(repeat) is not int or repeat < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("repeat 必须为正整数，warmup 必须为非负整数")
    config = config_from_dict(config.snapshot())
    info = inspect_video(config.video)
    end = info.frame_count if config.end_frame is None else config.end_frame
    if not 0 <= config.start_frame < end <= info.frame_count:
        raise ValueError("性能测试帧区间超出源视频范围")
    if "pitch" in stages and config.pitch_model != "eleven":
        raise ValueError("pitch 测试的是十一人制逐帧白线；八人制手动标定不属于此视频阶段")
    if "tracking" in stages and not config.weights.is_file():
        raise ValueError(f"缺少本地人物权重：{config.weights}")
    preset = model_preset(ball_model) if "ball" in stages else None
    video_hash = file_hash(config.video)
    config = replace(config, end_frame=end)
    settings = config.snapshot()
    settings.pop("output_root")
    protocol = {
        "timing_version": 1, "stages": list(stages), "config": settings,
        "video_sha256": video_hash, "warmup": warmup, "torch_threads": 6,
        "device": config.device,
        "tracking_weights_sha256": file_hash(config.weights) if "tracking" in stages else None,
        "ball_model": ball_model if preset else None,
        "ball_weights_sha256": file_hash(ROOT / preset["weights"]) if preset else None,
    }
    previous = read_json(Path(baseline)) if baseline is not None else None
    if previous is not None:
        compare_results({"protocol": protocol, "summary": {}}, previous)
    import torch

    torch.set_num_threads(6)
    destination = Path(output_root if output_root is not None else ROOT / "outputs/bench") / new_id("benchmark")
    destination.mkdir(parents=True)
    report_path = destination / "result.json"
    recorder = PerformanceRecorder()
    result = {
        "schema_version": 1, "status": "running", "protocol": protocol,
        "repeat": repeat, "code": code_version(), "result_path": str(report_path.resolve()),
        "environment": {"platform": platform.platform(), "python": sys.version, "processor": platform.processor(),
                        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
                        "opencv_threads": cv2.getNumThreads(), "device": config.device,
                        "gpu_available": bool(torch.cuda.is_available()), "gpu_name": gpu_name(),
                        "versions": versions()},
        "records": [], "summary": {},
    }
    write_json(report_path, result)
    print(f"[benchmark] 输出目录：{destination}", flush=True)
    try:
        for iteration in range(warmup + repeat):
            is_warmup = iteration < warmup
            current = PerformanceRecorder()
            label = f"预热 {iteration + 1}/{warmup}" if is_warmup else f"正式 {iteration - warmup + 1}/{repeat}"
            print(f"[benchmark] {label}", flush=True)
            iteration_config = replace(config, output_root=destination / f"iteration_{iteration + 1:02d}")
            if "tracking" in stages:
                run_dir = track(iteration_config, performance=current)
            else:
                run_dir = iteration_config.output_root / new_id("run")
                prepare_run(run_dir, iteration_config, info, video_hash)
            if "pitch" in stages:
                calibrate_run(run_dir, performance=current)
            if "ball" in stages:
                track_ball_run(run_dir, ball_model, performance=current, device=config.device)
            if {metric.stage for metric in current.metrics} != set(stages) or len(current.metrics) != len(stages):
                raise RuntimeError("流水线没有完整上报所选阶段的性能数据")
            if not is_warmup:
                for metric in current.metrics:
                    recorder.record(metric.stage, metric.frames, metric.elapsed_s, metric.source_fps,
                                    {**(metric.metadata or {}), "iteration": iteration - warmup + 1}, source_frames=metric.source_frames)
            result.update(recorder.as_dict(), summary=summarize(recorder))
            write_json(report_path, result)
        if file_hash(config.video) != video_hash:
            raise ValueError("测试期间源视频发生变化，结果不可比较")
        result["status"] = "complete"
        if previous is not None:
            result["comparison"] = compare_results(result, previous)
            result["baseline_environment_matches"] = result["environment"] == previous.get("environment")
        write_json(report_path, result)
        return result
    except Exception as error:
        result.update(status="failed", error=str(error))
        write_json(report_path, result)
        raise


def print_benchmark(result):
    """打印各阶段的重复测量汇总与可选基线倍速。"""
    labels = {"tracking": "人物追踪", "pitch": "逐帧白线", "ball": "足球追踪"}
    print("阶段\t轮数\tFPS\t毫秒/帧\t耗时范围(s)")
    for stage, summary in result["summary"].items():
        print(f"{labels.get(stage, stage)}\t{summary['samples']}\t{summary['fps']:.2f}\t{summary['ms_per_frame']:.1f}\t{summary['min_elapsed_s']:.3f}–{summary['max_elapsed_s']:.3f}")
        if stage in result.get("comparison", {}):
            print(f"  相对基线速度：{result['comparison'][stage]['speedup']:.3f} 倍（大于 1 表示更快）")
    if result.get("baseline_environment_matches") is False:
        print("提示：两次测试的软件或硬件环境不同，请结合环境快照判断差异。")
    print(f"[benchmark] 结果：{result['result_path']}")
