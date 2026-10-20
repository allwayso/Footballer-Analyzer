"""编排跟踪阶段：读取配置、采样分队、持续关联目标，分别缓存检测与原始轨迹。"""

from pathlib import Path
import html
import json
import time

import cv2

from .config import Config
from .detection import YoloDetector
from .performance import PerformanceRecorder
from .storage import DETECTION_SCHEMA, TRACK_SCHEMA, RowWriter, file_hash, new_id, versions, write_json, write_yaml
from .reassociation import EvidenceCollector, compute_result, commit_result, evidence_signature
from .stream_association import StreamingAssociator
from .tracking import create_tracker
from .teams import TeamClassifier, shirt_feature
from .video import inspect_video, read_frames


def track(config: Config, progress=None, performance: PerformanceRecorder | None = None) -> Path:
    """分析配置指定的源帧窗口，创建独立运行目录并返回其路径。

    写出原始缓存并复用跨帧跟踪器；progress 接收阶段、已处理数与总数，失败写入清单后抛错。
    performance 可选，记录当前人物阶段耗时；抽帧吞吐与源区间时长分别计算。
    """
    info = inspect_video(config.video)
    end_frame = config.end_frame if config.end_frame is not None else info.frame_count
    if config.start_frame >= end_frame or end_frame > info.frame_count:
        raise ValueError("分析区间超出视频范围")
    if not config.weights.is_file():
        raise ValueError(f"缺少本地模型权重：{config.weights}")
    run_dir = config.output_root / new_id("run")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    manifest = {
        "schema_version": 1, "run_id": run_dir.name, "status": "running",
        "config": config.snapshot(), "video": info.snapshot(),
        "video_sha256": file_hash(config.video), "weights_sha256": file_hash(config.weights),
        "start_frame": config.start_frame, "end_frame": end_frame,
        "versions": versions(),
    }
    write_json(run_dir / "manifest.json", manifest)
    try:
        detector = YoloDetector(config.weights, config.imgsz, config.confidence, config.nms_iou)
        tracker = create_tracker(config, info.fps)
        manifest["tracking"] = tracker.config.snapshot() if hasattr(tracker, "config") else {"algorithm": "bytetrack-legacy"}
        write_json(run_dir / "manifest.json", manifest)
        teams = TeamClassifier()
        features = []
        sample_end = min(end_frame, config.start_frame + round(info.fps * 8))
        sample_stride = max(1, round(info.fps))
        sample_total = len(range(config.start_frame, sample_end, sample_stride))
        if progress is not None:
            progress("sampling", 0, sample_total)
        for sample_index, packet in enumerate(read_frames(config.video, config.start_frame, sample_end, sample_stride), 1):
            samples = detector.detect(packet.image)
            for box in samples[samples.confidence >= config.track_activation_threshold + 0.1].xyxy:
                feature = shirt_feature(packet.image, box)
                if feature is not None:
                    features.append(feature)
            if progress is not None:
                progress("sampling", sample_index, sample_total)
        teams.fit(features[:256])
        write_json(run_dir / "teams.json", {"centers": teams.model.cluster_centers_.tolist() if teams.model is not None else [], "labels": {"cluster_0": "球队颜色簇 A", "cluster_1": "球队颜色簇 B"}})
        preview = cv2.VideoWriter(str(run_dir / "tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info.fps / config.stride, (info.width, info.height))
        if not preview.isOpened():
            raise ValueError("无法创建轨迹预览视频")
        event_file = (run_dir / "tracking_events.jsonl").open("w", encoding="utf-8")
        links = (run_dir / "tracking_links.jsonl").open("w", encoding="utf-8")
        tracker.event_sink = lambda event: event_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        collector = EvidenceCollector(info.fps)
        streaming = StreamingAssociator(manifest)
        association_error = None
        catalog = {}
        processed = 0
        detection_count = 0
        observation_count = 0
        total = len(range(config.start_frame, end_frame, config.stride))
        if progress is not None:
            progress("tracking", 0, total)
        def consume(result):
            """所有下游只处理按源帧顺序确认的真实观测，包括结束刷新的帧。"""
            nonlocal observation_count, processed, association_error
            packet, detections = result.packet, result.detections
            observation_count += len(detections)
            annotated = packet.image.copy()
            current_rows = []
            for detection_index, box in enumerate(detections.xyxy):
                track_id = int(detections.tracker_id[detection_index])
                left, top, right, bottom = map(float, box)
                team_id = teams.predict(packet.image, box, track_id)
                row = {
                    "frame_index": packet.frame_index, "timestamp_s": packet.timestamp_s,
                    "track_id": track_id, "bbox_x1": left, "bbox_y1": top,
                    "bbox_x2": right, "bbox_y2": bottom,
                    "confidence": float(detections.confidence[detection_index]), "class_id": 0, "team_id": team_id,
                }
                writer.append(row)
                source_indices = detections.data.get("source_detection_index")
                links.write(json.dumps({"frame_index": packet.frame_index, "track_id": track_id,
                                        "detection_index": int(source_indices[detection_index]) if source_indices is not None else None}) + "\n")
                current_rows.append(row)
                if track_id not in catalog:
                    catalog[track_id] = {"track_id": track_id, "start_frame": packet.frame_index, "end_frame": packet.frame_index + 1, "observations": 0}
                    crop = packet.image[max(0, int(top)):int(bottom), max(0, int(left)):int(right)]
                    if crop.size:
                        (run_dir / "crops").mkdir(exist_ok=True)
                        cv2.imwrite(str(run_dir / "crops" / f"{track_id}.jpg"), crop)
                catalog[track_id]["end_frame"] = packet.frame_index + 1
                catalog[track_id]["observations"] += 1
                color = (80, 220, 255)
                cv2.rectangle(annotated, (int(left), int(top)), (int(right), int(bottom)), color, 2)
                cv2.putText(annotated, f"#{track_id}", (int(left), max(15, int(top) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            collector.observe(packet.image, current_rows)
            if association_error is None:
                try:
                    live = streaming.observe(packet.frame_index, current_rows, collector.samples)
                    if live is not None:
                        write_json(run_dir / "association_live.json", live)
                except Exception as error:
                    # 在线关联异常不影响原始追踪，末尾完整分析仍可重试。
                    association_error = str(error)
                    write_json(run_dir / "association_live.json", {"status": "failed", "error": association_error})
            preview.write(annotated)
            processed += 1
        try:
            with RowWriter(run_dir / "tracks.parquet", TRACK_SCHEMA) as writer, RowWriter(run_dir / "detections.parquet", DETECTION_SCHEMA) as detection_writer:
                for packet in read_frames(config.video, config.start_frame, end_frame, config.stride):
                    raw_detections = detector.detect(packet.image)
                    for detection_index, box in enumerate(raw_detections.xyxy):
                        detection_writer.append({
                            "frame_index": packet.frame_index, "timestamp_s": packet.timestamp_s,
                            "bbox_x1": float(box[0]), "bbox_y1": float(box[1]),
                            "bbox_x2": float(box[2]), "bbox_y2": float(box[3]),
                            "confidence": float(raw_detections.confidence[detection_index]), "class_id": 0,
                        })
                    detection_count += len(raw_detections)
                    for result in tracker.submit(packet, raw_detections):
                        consume(result)
                    if progress is not None:
                        progress("tracking", processed, total)
                    if processed and processed % 50 == 0:
                        print(f"[track] {processed} 帧 / {len(catalog)} 条轨迹", flush=True)
                for result in tracker.flush():
                    consume(result)
        finally:
            preview.release()
            event_file.close()
            links.close()
        write_json(run_dir / "tracking_diagnostics.json", dict(tracker.stats))
        write_json(run_dir / "catalog.json", list(catalog.values()))
        cards = "".join(f'<article><img src="crops/{entry["track_id"]}.jpg" alt="轨迹缩略图"><h2>#{entry["track_id"]}</h2><p>帧 {entry["start_frame"]}–{entry["end_frame"]} · {entry["observations"]} 次观测</p></article>' for entry in catalog.values())
        (run_dir / "track_catalog.html").write_text(f'<!doctype html><meta charset="utf-8"><title>轨迹目录</title><style>body{{font-family:system-ui;background:#102026;color:#eee}}main{{display:flex;flex-wrap:wrap;gap:16px}}article{{padding:16px;background:#1e343c}}img{{height:120px}}</style><h1>{html.escape(run_dir.name)}</h1><p>轨迹号仅在本次运行中有效。使用调试台绑定真实身份。</p><main>{cards}</main>', encoding="utf-8")
        write_yaml(run_dir / "bindings.yaml", {"run_id": run_dir.name, "bindings": []})
        elapsed_s = time.perf_counter() - started
        manifest.update(status="complete", processed_frames=processed, track_count=len(catalog), detection_count=detection_count, observation_count=observation_count, elapsed_s=round(elapsed_s, 3))
        write_json(run_dir / "manifest.json", manifest)
        if performance is not None:
            performance.record("tracking", processed, elapsed_s, info.fps, {"run_id": run_dir.name, "input_size": config.imgsz, "track_count": len(catalog), "detection_count": detection_count, "stride": config.stride}, source_frames=end_frame - config.start_frame)
        # 关联失败不会将已完成的原始追踪降级为失败，可在复核页重试。
        try:
            write_json(run_dir / "association_evidence.json", {"schema_version": 1, "signature": evidence_signature(run_dir, manifest), "samples": collector.samples})
            def association_progress(stage, processed, total):
                """在追踪进度通道显示末尾的关联分析。"""
                if progress is not None:
                    progress("association", processed, total)
            result = compute_result(run_dir, progress=association_progress)
            commit_result(run_dir, result, result["input_signature"])
            live = streaming.snapshot(end_frame - 1)
            live.update(status="reconciled", final_batch_id=result["batch_id"], error=association_error)
            write_json(run_dir / "association_live.json", live)
            write_json(run_dir / "association_job.json", {"status": "complete", "stage": "complete", "batch_id": result["batch_id"]})
        except Exception as association_error:
            write_json(run_dir / "association_job.json", {"status": "failed", "error": str(association_error)})
        print(f"[complete] {run_dir}", flush=True)
        return run_dir
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(run_dir / "manifest.json", manifest)
        raise
