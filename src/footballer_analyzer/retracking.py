"""只读检测缓存重跟踪，建立独立运行、稳定检测索引和可审计诊断。"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import supervision as sv

from .calibration import load_calibration, validate_calibration
from .config import config_from_dict
from .field_review import field_segments
from .motion_tracking import MotionColorTracker, MotionConfig
from .storage import TRACK_SCHEMA, RowWriter, file_hash, new_id, read_json, read_rows, write_json, write_yaml
from .teams import shirt_histogram, shirt_feature
from .tracking import create_tracker
from .video import FramePacket, read_frames

BOX_KEYS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


def detection_frames(run):
    """按源帧及Parquet原始顺序建立稳定检测索引。"""
    grouped = defaultdict(list)
    for row in read_rows(Path(run) / "detections.parquet"):
        grouped[row["frame_index"]].append(row)
    return grouped


def detections_from_rows(rows):
    """还原检测输入，不使用旧轨迹编号或框重匹配。"""
    return sv.Detections(xyxy=np.array([[row[key] for key in BOX_KEYS] for row in rows], dtype=float).reshape(-1, 4),
                         confidence=np.array([row["confidence"] for row in rows]), class_id=np.zeros(len(rows), dtype=int))


def input_snapshot(run):
    """冻结输入文件、视频和标定摘要，检测计算期间的输入变化。"""
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    video = Path(manifest["config"]["video"])
    calibration = load_calibration(run)
    if file_hash(video) != manifest["video_sha256"]:
        raise ValueError("源视频内容与原运行不一致")
    if calibration:
        validate_calibration(calibration, manifest, require_reviewed=False)
    return {"source_run": str(run.resolve()), "video_sha256": manifest["video_sha256"],
            "manifest_sha256": file_hash(run / "manifest.json"), "detections_sha256": file_hash(run / "detections.parquet"),
            "tracks_sha256": file_hash(run / "tracks.parquet"), "calibration": calibration,
            "start_frame": manifest["start_frame"], "end_frame": manifest["end_frame"], "stride": manifest["config"]["stride"]}


def prepare_colors(run, destination):
    """顺序解码一次视频提取逐检测颜色，供全部网格复用且不运行YOLO。"""
    destination = Path(destination)
    signature = input_snapshot(run)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as cache:
            if json.loads(str(cache["signature"])) != signature:
                raise ValueError("已有颜色缓存与输入不匹配，请使用新路径")
        return destination
    manifest = read_json(Path(run) / "manifest.json")
    grouped = detection_frames(run)
    values, frames, indices = [], [], []
    for packet in read_frames(Path(manifest["config"]["video"]), manifest["start_frame"], manifest["end_frame"], manifest["config"]["stride"]):
        for index, row in enumerate(grouped[packet.frame_index]):
            value = shirt_histogram(packet.image, [row[key] for key in BOX_KEYS])
            values.append(value if value is not None else np.full(44, np.nan))
            frames.append(packet.frame_index)
            indices.append(index)
        if packet.frame_index % 250 == 0:
            print(f"[colors] frame={packet.frame_index}", flush=True)
    if input_snapshot(run) != signature:
        raise ValueError("提取期间输入发生变化")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, colors=np.array(values, dtype=np.float32).reshape(-1, 44), frames=frames, indices=indices,
                            signature=json.dumps(signature, ensure_ascii=False))
    return destination


def load_colors(path, signature):
    """校验缓存摘要和逐帧索引后恢复颜色；NaN代表缺失证据。"""
    grouped = defaultdict(list)
    with np.load(path, allow_pickle=False) as cache:
        if json.loads(str(cache["signature"])) != signature:
            raise ValueError("颜色缓存输入摘要不匹配")
        for frame, index, value in zip(cache["frames"], cache["indices"], cache["colors"]):
            if int(index) != len(grouped[int(frame)]):
                raise ValueError("颜色缓存检测索引不连续")
            feature = value.astype(float) if np.isfinite(value).all() else None
            if feature is not None:
                feature /= feature.sum()
            grouped[int(frame)].append(feature)
    return grouped


def copy_calibration(source, target, calibration):
    """只复制同视频标定快照，不迁移人工融合或身份绑定。"""
    if calibration:
        shutil.copy2(source / "calibration.yaml", target / "calibration.yaml")
        if calibration.get("mode") == "white_lines":
            identifier = calibration["calibration_id"]
            folder = target / "calibrations" / identifier
            folder.mkdir(parents=True)
            shutil.copy2(source / "calibrations" / identifier / "frames.json", folder / "frames.json")


def retrack(source, output_root, parameters=None, color_cache=None, export_video=True, algorithm="motion-color-v1", use_calibration=True, legacy_fuse_score=True):
    """复用原检测创建新运行；原文件只读，结果提交前再次核验输入。"""
    source, output_root = Path(source), Path(output_root)
    signature = input_snapshot(source)
    original = read_json(source / "manifest.json")
    parameters = parameters or MotionConfig()
    config = config_from_dict({**original["config"], "output_root": str(output_root.resolve()), "tracking_algorithm": algorithm,
                              "tracking_max_cost": parameters.max_cost, "tracking_position_tolerance": parameters.position_tolerance,
                              "tracking_competition_margin": parameters.competition_margin,
                              "tracking_new_threshold": parameters.new_threshold})
    run = output_root / new_id("run")
    run.mkdir(parents=True, exist_ok=False)
    calibration = signature["calibration"]
    started = time.perf_counter()
    manifest = {**original, "run_id": run.name, "status": "running", "config": config.snapshot(),
                "tracking": parameters.snapshot() if algorithm == "motion-color-v1" else {"algorithm": algorithm, "fuse_score": legacy_fuse_score},
                "tracking_implementation_sha256": file_hash(Path(__file__).with_name("motion_tracking.py")),
                "retracking": {"source": signature, "uses_calibration": bool(use_calibration and calibration and calibration.get("reviewed")),
                               "post_association_applied": False}}
    write_json(run / "manifest.json", manifest)
    preview = None
    try:
        shutil.copy2(source / "detections.parquet", run / "detections.parquet")
        copy_calibration(source, run, calibration)
        write_yaml(run / "bindings.yaml", {"run_id": run.name, "bindings": []})
        grouped = detection_frames(source)
        colors = load_colors(color_cache, signature) if color_cache else None
        if not export_video and colors is None:
            raise ValueError("不解码视频时必须提供逐检测颜色缓存")
        fps = original["video"]["fps"]
        width, height = original["video"]["width"], original["video"]["height"]
        if export_video:
            preview = cv2.VideoWriter(str(run / "tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / config.stride, (width, height))
            if not preview.isOpened():
                raise ValueError("无法创建重跟踪预览视频")
        catalog, output_rows, team_votes = {}, [], defaultdict(Counter)
        centers = read_json(source / "teams.json").get("centers", []) if (source / "teams.json").exists() else []
        write_json(run / "teams.json", {"centers": centers, "labels": {"cluster_0": "球队颜色簇 A", "cluster_1": "球队颜色簇 B"}})
        with (run / "tracking_events.jsonl").open("w", encoding="utf-8") as event_file, (run / "tracking_links.jsonl").open("w", encoding="utf-8") as links, RowWriter(run / "tracks.parquet", TRACK_SCHEMA) as writer:
            def event_sink(event):
                """写入匹配证据及拒绝原因，不影响原始运行。"""
                event_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            tracker = MotionColorTracker(fps, config.stride, parameters, calibration if use_calibration else None, event_sink) if algorithm == "motion-color-v1" else create_tracker(config, fps)

            def consume(result):
                """直接保存来源检测及确认身份，不进行第二次IoU重配。"""
                packet, detections = result.packet, result.detections
                annotated = packet.image.copy() if preview is not None else None
                for index, box in enumerate(detections.xyxy):
                    track_id = int(detections.tracker_id[index])
                    source_index = int(detections.data["source_detection_index"][index]) if "source_detection_index" in detections.data else None
                    team = "unknown"
                    if packet.image is not None and centers:
                        feature = shirt_feature(packet.image, box)
                        if feature is not None:
                            cluster = int(np.argmin(np.linalg.norm(np.array(centers) - feature, axis=1)))
                            team_votes[track_id][cluster] += 1
                            team = f"cluster_{team_votes[track_id].most_common(1)[0][0]}"
                    row = {"frame_index": packet.frame_index, "timestamp_s": packet.timestamp_s, "track_id": track_id,
                           **dict(zip(BOX_KEYS, map(float, box))), "confidence": float(detections.confidence[index]), "class_id": 0, "team_id": team}
                    writer.append(row)
                    output_rows.append(row)
                    links.write(json.dumps({"frame_index": packet.frame_index, "detection_index": source_index, "track_id": track_id}) + "\n")
                    if track_id not in catalog:
                        catalog[track_id] = {"track_id": track_id, "start_frame": packet.frame_index, "end_frame": packet.frame_index + 1, "observations": 0}
                        if packet.image is not None:
                            left, top, right, bottom = map(int, box)
                            crop = packet.image[max(0, top):min(height, bottom), max(0, left):min(width, right)]
                            if crop.size:
                                (run / "crops").mkdir(exist_ok=True)
                                cv2.imwrite(str(run / "crops" / f"{track_id}.jpg"), crop)
                    catalog[track_id]["end_frame"] = packet.frame_index + 1
                    catalog[track_id]["observations"] += 1
                    if annotated is not None:
                        left, top, right, bottom = map(int, box)
                        cv2.rectangle(annotated, (left, top), (right, bottom), (80, 220, 255), 2)
                        cv2.putText(annotated, f"#{track_id}", (left, max(15, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
                if annotated is not None:
                    preview.write(annotated)

            packets = read_frames(config.video, original["start_frame"], original["end_frame"], config.stride) if export_video else (
                FramePacket(frame, frame / fps, None) for frame in range(original["start_frame"], original["end_frame"], config.stride))
            from supervision.tracker.byte_tracker import matching
            original_fuse = matching.fuse_score
            try:
                if algorithm == "bytetrack-legacy" and not legacy_fuse_score:
                    matching.fuse_score = lambda cost, detections: cost
                for packet in packets:
                    detections = detections_from_rows(grouped[packet.frame_index])
                    for result in tracker.submit(packet, detections, colors[packet.frame_index] if colors is not None else None):
                        consume(result)
                for result in tracker.flush():
                    consume(result)
            finally:
                matching.fuse_score = original_fuse
        if signature != input_snapshot(source) or file_hash(run / "detections.parquet") != signature["detections_sha256"]:
            raise ValueError("重跟踪期间输入变化，拒绝发布结果")
        segments = field_segments(output_rows, manifest, calibration)
        summary = {"run_id": run.name, "algorithm": algorithm, "track_count": len(catalog), "field_segments": len(segments),
                   "observation_count": len(output_rows), "elapsed_s": round(time.perf_counter() - started, 3),
                   "statistics": dict(tracker.stats), "parameters": manifest["tracking"], "identity_error_rate": None,
                   "true_observation_coverage": None, "acceptance": "pending_identity_truth", "post_association_applied": False}
        summary["tracking_implementation_sha256"] = manifest["tracking_implementation_sha256"]
        write_json(run / "catalog.json", list(catalog.values()))
        write_json(run / "tracking_summary.json", summary)
        write_json(run / "tracking_diagnostics.json", dict(tracker.stats))
        write_json(run / "field_review.json", {"schema_version": 1, "operations": [], "blocked_pairs": []})
        manifest.update(status="complete", track_count=len(catalog), observation_count=len(output_rows),
                        processed_frames=len(range(original["start_frame"], original["end_frame"], config.stride)),
                        elapsed_s=summary["elapsed_s"])
        write_json(run / "manifest.json", manifest)
        print(f"[retrack] {run.name}: tracks={len(catalog)} field={len(segments)} observations={len(output_rows)} elapsed={summary['elapsed_s']}s", flush=True)
        return run
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(run / "manifest.json", manifest)
        raise
    finally:
        if preview is not None:
            preview.release()
