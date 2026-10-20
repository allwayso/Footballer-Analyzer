"""只读重放检测缓存，记录 ByteTrack 分阶段匹配代价及原始轨迹复现差异。"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker import matching

from footballer_analyzer.storage import file_hash, read_json, read_rows, write_json
from footballer_analyzer.tracking import ByteTracker


BOX_KEYS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


def track_snapshot(track):
    """复制轨迹状态，帧号转换为从零开始的源缓存帧前另存库内部帧号。"""
    return {
        "id": int(track.external_track_id),
        "internal_id": int(track.internal_track_id),
        "state": track.state.name,
        "activated": bool(track.is_activated),
        "last_internal_frame": int(track.frame_id),
        "start_internal_frame": int(track.start_frame),
        "box": track.tlbr.tolist(),
        "score": float(track.score),
        "mean": track.mean.tolist() if track.mean is not None else None,
    }


def replay(run_dir, end_frame, selected_frames):
    """从运行首帧重放，临时插桩 supervision 并在退出时恢复原函数。"""
    manifest = read_json(run_dir / "manifest.json")
    config = manifest["config"]
    raw_by_frame = defaultdict(list)
    expected_by_frame = defaultdict(list)
    for row in read_rows(run_dir / "detections.parquet"):
        if row["frame_index"] < end_frame:
            raw_by_frame[row["frame_index"]].append(row)
    for row in read_rows(run_dir / "tracks.parquet"):
        if row["frame_index"] < end_frame:
            expected_by_frame[row["frame_index"]].append(row)
    tracker = ByteTracker(
        manifest["video"]["fps"], config["stride"],
        config.get("track_activation_threshold", 0.25),
        config.get("lost_track_buffer_s", 3),
        config.get("minimum_matching_threshold", 0.8),
    )
    original_iou = matching.iou_distance
    original_assignment = matching.linear_assignment
    stages = []
    frame_index = None
    iou_stage = 0
    pending = None

    def capture_iou(first, second):
        """保存每阶段 Kalman 预测框与候选观测；去重阶段只透传。"""
        nonlocal iou_stage, pending
        costs = original_iou(first, second)
        iou_stage += 1
        if frame_index in selected_frames and iou_stage <= 3:
            pending = {
                "stage": ("high", "low", "unconfirmed")[iou_stage - 1],
                "tracks": [track_snapshot(track) for track in first],
                "detections": [track_snapshot(track) for track in second],
                "iou": (1 - costs).tolist(),
            }
        return costs

    def capture_assignment(costs, thresh):
        """在库原地裁剪非法代价前保存矩阵；额外记录输出阶段二次映射。"""
        nonlocal pending
        entry = pending or {"stage": "output_remap"}
        pending = None
        if frame_index in selected_frames:
            entry.update(cost=costs.copy().tolist(), threshold=float(thresh))
        result = original_assignment(costs, thresh)
        if frame_index in selected_frames:
            entry.update(matches=result[0].tolist(), unmatched_tracks=list(result[1]), unmatched_detections=list(result[2]))
            stages.append(entry)
        return result

    frames = []
    different_frames = []
    replay_rows = []
    matching.iou_distance = capture_iou
    matching.linear_assignment = capture_assignment
    try:
        for frame_index in range(manifest["start_frame"], min(end_frame, manifest["end_frame"]), config["stride"]):
            stages = []
            pending = None
            iou_stage = 0
            rows = raw_by_frame[frame_index]
            boxes = np.asarray([[row[key] for key in BOX_KEYS] for row in rows], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray([row["confidence"] for row in rows], dtype=np.float32)
            detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=np.zeros(len(rows), dtype=int))
            result = tracker.update(detections, frame_index / manifest["video"]["fps"])
            actual = [{"frame_index": frame_index, "track_id": int(tid), "box": box.tolist(), "score": float(score)} for tid, box, score in zip(result.tracker_id, result.xyxy, result.confidence)]
            replay_rows.extend(actual)
            expected = [{"track_id": row["track_id"], "box": [row[key] for key in BOX_KEYS]} for row in expected_by_frame[frame_index]]
            actual_sorted = sorted(actual, key=lambda row: (row["track_id"], row["box"]))
            expected_sorted = sorted(expected, key=lambda row: (row["track_id"], row["box"]))
            same = len(actual_sorted) == len(expected_sorted) and all(
                first["track_id"] == second["track_id"] and np.allclose(first["box"], second["box"], rtol=0, atol=1e-4)
                for first, second in zip(actual_sorted, expected_sorted)
            )
            if not same:
                different_frames.append(frame_index)
            if frame_index in selected_frames:
                frames.append({
                    "frame_index": frame_index, "stages": stages, "output": actual,
                    "tracked": [track_snapshot(track) for track in tracker.tracker.tracked_tracks],
                    "lost": [track_snapshot(track) for track in tracker.tracker.lost_tracks],
                    "removed": [track_snapshot(track) for track in tracker.tracker.removed_tracks],
                })
    finally:
        matching.iou_distance = original_iou
        matching.linear_assignment = original_assignment
    return {
        "run_id": run_dir.name, "config": config,
        "recorded_supervision_version": manifest["versions"].get("supervision"),
        "replay_supervision_version": sv.__version__,
        "detections_sha256": file_hash(run_dir / "detections.parquet"),
        "tracks_sha256": file_hash(run_dir / "tracks.parquet"),
        "end_frame_exclusive": end_frame, "different_frames": different_frames,
        "exact_replay": not different_frames,
        "frames": frames, "replay_observations": replay_rows,
    }


def main():
    """解析源帧诊断范围，报告写入新目录，禁止覆盖运行输入。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--end-frame", type=int, default=161)
    parser.add_argument("--frames", default="34-47,58-71,128-141")
    args = parser.parse_args()
    selected = set()
    for text in args.frames.split(","):
        bounds = [int(value) for value in text.split("-")]
        if len(bounds) not in (1, 2) or min(bounds) < 0 or bounds[-1] < bounds[0]:
            parser.error("frames 格式应为 38-46,134-140，边界均包含")
        selected.update(range(bounds[0], bounds[-1] + 1))
    manifest = read_json(args.run / "manifest.json")
    if not manifest["start_frame"] < args.end_frame <= manifest["end_frame"]:
        parser.error("end-frame 必须位于运行窗口内且大于首帧")
    args.output.mkdir(parents=True, exist_ok=False)
    result = replay(args.run.resolve(), args.end_frame, selected)
    write_json(args.output / "trace.json", result)
    print({key: result[key] for key in ("run_id", "exact_replay", "different_frames", "recorded_supervision_version", "replay_supervision_version")})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
