"""为已有运行追加足球检测和时序筛选；独立版本保存，保留人物与标定原件。"""

from dataclasses import asdict
from pathlib import Path
import time

from .ball_detection import detect
from .ball_tracking import BallSelectionConfig, select_ball_candidates
from .config import ROOT
from .performance import PerformanceRecorder
from .storage import file_hash, new_id, read_json, write_json
from .video import inspect_video, read_frames

MODEL_PRESETS = {
    "football_v11n_tile": {"label": "足球 YOLO11n · 小目标切片（推荐）", "weights": "models/yolo11n-football-ball.pt", "imgsz": 640, "tile": 640},
    "football_v11n": {"label": "足球 YOLO11n · 全图（较快）", "weights": "models/yolo11n-football-ball.pt", "imgsz": 1280, "tile": 0},
    "scorevision_ball": {"label": "Scorevision YOLOv8x · 球专用", "weights": "models/scorevision-ball.pt", "imgsz": 1280, "tile": 0},
    "football_v5s": {"label": "足球 YOLOv5s · 对照", "weights": "third_party/Football-Game-Possession-Tracking/models/best.pt", "imgsz": 1280, "tile": 0},
    "football_v8l": {"label": "足球 YOLOv8l · 对照", "weights": "third_party/Football-Analytics-with-DL-and-CV/models/Yolo8L Players/weights/best.pt", "imgsz": 1280, "tile": 0},
    "coco_v8n": {"label": "通用 YOLOv8n · 对照", "weights": "models/yolov8n.pt", "imgsz": 1280, "tile": 0},
}
DEFAULT_MODEL = "football_v11n_tile"


def ball_models():
    """列出可选择的本地模型，缺少权重时只展示而不自动下载。"""
    return [{"id": identifier, "label": preset["label"], "available": (ROOT / preset["weights"]).is_file()} for identifier, preset in MODEL_PRESETS.items()]


def model_preset(identifier):
    """核实模型白名单和本地权重，拒绝任意路径及未知模型。"""
    if not isinstance(identifier, str) or identifier not in MODEL_PRESETS:
        raise ValueError("未知足球模型，请从列表选择")
    preset = MODEL_PRESETS[identifier]
    if not (ROOT / preset["weights"]).is_file():
        raise ValueError(f"缺少本地足球权重：{preset['weights']}")
    return preset


def ball_versions(run_dir):
    """返回已完成的足球结果概要，失败和中途退出的版本不作为可用结果。"""
    versions = []
    paths = sorted(Path(run_dir).glob("balls/ball_*/manifest.json"), key=lambda path: (path.stat().st_mtime_ns, path.parent.name), reverse=True)
    for path in paths:
        payload = read_json(path)
        if payload["status"] == "complete":
            versions.append(payload)
    return versions


def load_ball_result(run_dir, model_id=None, version=None):
    """读取指定模型的最新完成结果；无结果返回 None，旧运行也可直接读取。"""
    if model_id is not None and model_id not in MODEL_PRESETS:
        raise ValueError("未知足球模型")
    metadata = next((entry for entry in ball_versions(run_dir) if (model_id is None or entry["model_id"] == model_id) and (version is None or entry["result_id"] == version)), None)
    if metadata is None:
        return None
    result_id = metadata["result_id"]
    if not isinstance(result_id, str) or not result_id.startswith("ball_") or not result_id.replace("_", "").isalnum():
        raise ValueError("足球结果标识无效")
    return {"manifest": metadata, "frames": read_json(Path(run_dir) / "balls" / result_id / "frames.json")}


def select_in_windows(records, fps, image_size, config, window_size=250, context_size=25):
    """分窗筛选连续源帧，窗口两侧保留上下文以限制内存；缺失不插值。"""
    selected = []
    for start in range(0, len(records), window_size):
        left = max(0, start - context_size)
        stop = min(len(records), start + window_size)
        right = min(len(records), stop + context_size)
        window = select_ball_candidates(records[left:right], fps, image_size, config)
        selected.extend(window[start - left:stop - left])
    return selected


def track_ball_run(run_dir, model_id=DEFAULT_MODEL, progress=None, performance: PerformanceRecorder | None = None):
    """逐源帧补跑足球，保存候选与真实选中观测，成功后发布独立结果版本。

    只读取人物清单和源视频，不读取或修改标定；progress 接收阶段、已处理与总帧数。
    即使人物有抽帧，足球仍覆盖清单内每个源帧；源视频指纹不符或解码不足时失败。
    performance 可选，记录本次足球检测和筛选耗时及选中帧数。
    """
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("请先完成当前运行")
    preset = model_preset(model_id)
    video = Path(manifest["config"]["video"])
    source_hash = file_hash(video)
    if source_hash != manifest["video_sha256"]:
        raise ValueError("源视频已变化，不能将足球结果附加到旧运行")
    info = inspect_video(video)
    start, end = manifest["start_frame"], manifest["end_frame"]
    if not 0 <= start < end <= info.frame_count:
        raise ValueError("足球追踪区间超出源视频范围")
    weights = ROOT / preset["weights"]
    settings = BallSelectionConfig()
    result_dir = run_dir / "balls" / new_id("ball")
    metadata = {
        "schema_version": 1, "pipeline_version": 1, "result_id": result_dir.name,
        "run_id": run_dir.name, "status": "running", "model_id": model_id,
        "model": dict(preset), "weights_sha256": file_hash(weights), "video_sha256": source_hash,
        "start_frame": start, "end_frame": end, "fps": info.fps, "stride": 1,
        "confidence": 0.05, "overlap": 0.2, "filter": asdict(settings),
        "filter_window": 250, "filter_context": 25, "processed_frames": 0,
    }
    write_json(result_dir / "manifest.json", metadata)
    started = time.perf_counter()
    try:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(6)
        model = YOLO(str(weights))
        records = []
        total = end - start
        if progress:
            progress("ball_detecting", 0, total)
        for packet in read_frames(video, start, end, 1):
            candidates = detect(model, packet.image, preset["imgsz"], preset["tile"], 0.2, 0.05)
            records.append({"frame": packet.frame_index, "candidates": candidates})
            if len(records) % 25 == 0 or len(records) == total:
                metadata["processed_frames"] = len(records)
                write_json(result_dir / "manifest.json", metadata)
                if progress:
                    progress("ball_detecting", len(records), total)
        if progress:
            progress("ball_filtering", total, total)
        selected = select_in_windows(records, info.fps, (info.width, info.height), settings)
        frames = [{**record, "selected": candidate} for record, candidate in zip(records, selected)]
        write_json(result_dir / "frames.json", frames)
        elapsed_s = time.perf_counter() - started
        observed_frames = sum(candidate is not None for candidate in selected)
        metadata.update(status="complete", observed_frames=observed_frames, elapsed_s=round(elapsed_s, 3))
        write_json(result_dir / "manifest.json", metadata)
        if performance is not None:
            performance.record("ball", total, elapsed_s, info.fps, {"run_id": run_dir.name, "model_id": model_id, "observed_frames": observed_frames})
        return metadata
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        write_json(result_dir / "manifest.json", metadata)
        raise
