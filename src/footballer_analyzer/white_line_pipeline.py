"""运行逐帧白线标定，原子发布映射、识别掩膜与质量记录，并支持独立验收运行。"""

from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

from .config import Config, ROOT
from .dynamic_calibration import WhiteLineCalibrator
from .pitch_geometry import transform_points
from .pitch_models import pitch_spec
from .performance import PerformanceRecorder
from .storage import TRACK_SCHEMA, RowWriter, file_hash, new_id, read_json, write_json, write_yaml
from .video import inspect_video


def frame_roi(matrix, length, width, image_width, image_height):
    """返回投影球场与图像的交集多边形，像素坐标随当前帧更新。"""
    corners = transform_points([[0, 0], [length, 0], [length, width], [0, width]], np.linalg.inv(matrix))
    boundary = np.array([[0, 0], [image_width - 1, 0], [image_width - 1, image_height - 1], [0, image_height - 1]], np.float32)
    _, intersection = cv2.intersectConvexConvex(cv2.convexHull(corners.astype(np.float32)), boundary)
    return intersection.reshape(-1, 2).tolist() if intersection is not None else []


def draw_line_overlay(image, record, mask, curves):
    """绘制青色白线证据与黄色模型线；无效帧只显示证据和失败状态。"""
    overlay = image.copy()
    evidence = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay[evidence] = (255, 220, 0)
    if record["homography"] is not None:
        inverse = np.linalg.inv(record["homography"])
        for _, points in curves:
            pixels = transform_points(points, inverse)
            for first, second in zip(pixels[:-1], pixels[1:]):
                if np.isfinite([first, second]).all() and np.abs([first, second]).max() < 1e6:
                    cv2.line(overlay, np.rint(first).astype(int), np.rint(second).astype(int), (0, 255, 255), 2, cv2.LINE_AA)
    label = f"Frame {record['frame_index']} | {record['status']} | support {record['support']:.1%}"
    cv2.rectangle(overlay, (0, 0), (min(image.shape[1], 1100), 42), (20, 35, 30), -1)
    cv2.putText(overlay, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 245, 230), 2)
    return overlay


def calibrate_run(run_dir, length=None, width=None, progress=None, performance: PerformanceRecorder | None = None):
    """为运行区间逐源帧计算 H；完成后发布新版本，保留旧标定及人物缓存。

    progress 接收已处理帧数与总帧数；失败帧保留原因，禁止以旧矩阵补齐。
    performance 可选，记录本次白线阶段耗时和有效帧比例。
    """
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    length = float(length if length is not None else manifest["config"]["pitch_length_m"])
    width = float(width if width is not None else manifest["config"]["pitch_width_m"])
    calibrator = WhiteLineCalibrator(length, width)
    source = Path(manifest["config"]["video"])
    if file_hash(source) != manifest["video_sha256"]:
        raise ValueError("源视频摘要发生变化，不能复用该运行")
    calibration_id = new_id("lines")
    destination = run_dir / "calibrations" / calibration_id
    masks = destination / "masks"
    masks.mkdir(parents=True)
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, manifest["start_frame"])
    records = {}
    started = time.perf_counter()
    total = manifest["end_frame"] - manifest["start_frame"]
    try:
        for frame_index in range(manifest["start_frame"], manifest["end_frame"]):
            success, image = capture.read()
            if not success:
                if manifest.get("allow_truncated_tail") and records:
                    break
                raise ValueError(f"源视频无法解码第 {frame_index} 帧，未替换原标定")
            record, mask = calibrator.process(image, frame_index)
            if record["homography"] is not None:
                record["roi"] = frame_roi(record["homography"], length, width, image.shape[1], image.shape[0])
            records[str(frame_index)] = record
            evidence = np.zeros((*mask.shape, 4), np.uint8)
            evidence[:, :, :3] = (255, 220, 0)
            evidence[:, :, 3] = mask
            if not cv2.imwrite(str(masks / f"{frame_index}.png"), evidence):
                raise OSError("白线识别掩膜写入失败")
            if progress is not None:
                progress(len(records), total)
    finally:
        capture.release()
    if not records:
        raise ValueError("分析区间没有可解码视频帧")
    if file_hash(source) != manifest["video_sha256"]:
        raise ValueError("分析过程中源视频发生变化，未替换原标定")
    end_frame = int(next(reversed(records))) + 1
    counts = dict(Counter(record["status"] for record in records.values()))
    elapsed_s = time.perf_counter() - started
    summary = {"frame_count": len(records), "valid_frames": counts.get("valid", 0), "statuses": counts, "elapsed_s": round(elapsed_s, 3)}
    summary["valid_ratio"] = summary["valid_frames"] / len(records)
    calibration = {
        "schema_version": 2, "mode": "white_lines", "calibration_id": calibration_id,
        "pitch_model": "eleven", "pitch_spec": pitch_spec("eleven", length, width),
        "video_sha256": manifest["video_sha256"], "width": manifest["video"]["width"], "height": manifest["video"]["height"],
        "reference_frame": manifest["start_frame"], "start_frame": manifest["start_frame"], "end_frame": end_frame,
        "pitch_length_m": length, "pitch_width_m": width, "reviewed": False,
        "orientation": "image_right_is_pitch_positive_x", "summary": summary,
        "note": "逐帧检测白线并匹配标准十一人制球场；方向自动约定，不代表球队进攻方向。",
    }
    write_json(destination / "frames.json", records)
    write_json(destination / "summary.json", summary)
    write_yaml(destination / "calibration.yaml", calibration)
    write_yaml(run_dir / "calibration.yaml", calibration)
    if performance is not None:
        performance.record("pitch", len(records), elapsed_s, manifest["video"].get("fps"), {"run_id": run_dir.name, "valid_ratio": summary["valid_ratio"]})
    print(f"[calibration] {run_dir} | {summary['valid_frames']}/{len(records)} valid | {summary['elapsed_s']:.1f}s", flush=True)
    return calibration


def create_line_run(video, output_root, length=105.0, width=68.0, start_frame=0, end_frame=None):
    """创建无需人物检测的白线验收运行，返回带逐帧标定的运行目录。"""
    WhiteLineCalibrator(length, width)
    video = Path(video).resolve()
    info = inspect_video(video)
    stop = info.frame_count if end_frame is None else end_frame
    if type(start_frame) is not int or type(stop) is not int or not 0 <= start_frame < stop <= info.frame_count:
        raise ValueError("白线验收帧区间超出源视频")
    output_root = Path(output_root).resolve()
    run_dir = output_root / new_id("run")
    run_dir.mkdir(parents=True)
    config = Config(video, ROOT / "models/yolov8n.pt", output_root, ROOT / "data/players.yaml", "pitch_review", video.stem, start_frame=start_frame, end_frame=stop, pitch_length_m=length, pitch_width_m=width)
    manifest = {"schema_version": 1, "run_id": run_dir.name, "status": "running", "kind": "white_line_review", "config": config.snapshot(), "video": info.snapshot(), "video_sha256": file_hash(video), "start_frame": start_frame, "end_frame": stop, "allow_truncated_tail": end_frame is None}
    write_json(run_dir / "manifest.json", manifest)
    with RowWriter(run_dir / "tracks.parquet", TRACK_SCHEMA):
        pass
    write_json(run_dir / "catalog.json", [])
    write_yaml(run_dir / "bindings.yaml", {"run_id": run_dir.name, "bindings": []})
    try:
        calibration = calibrate_run(run_dir, length, width)
        manifest.update(status="complete", end_frame=calibration["end_frame"], processed_frames=calibration["summary"]["frame_count"])
        if calibration["end_frame"] < stop:
            manifest["video"].update(reported_frame_count=info.frame_count, frame_count=calibration["end_frame"])
        write_json(run_dir / "manifest.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(run_dir / "manifest.json", manifest)
        raise
