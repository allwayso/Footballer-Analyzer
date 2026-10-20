"""整理场内片段与可撤销身份分组，供追踪和复核共用。"""

from collections import Counter
import math

from .calibration import project_observation


def field_segments(rows, manifest, calibration):
    """把可靠场地内观测整理为轨迹片段；同 ID 的短时漏检不切段。"""
    if calibration is None or calibration.get("reviewed") is not True:
        return []
    fps = manifest["video"]["fps"]
    config = manifest["config"]
    stride = config["stride"]
    max_gap_frames = stride + math.ceil(float(config.get("lost_track_buffer_s", 3.0)) * fps)
    grouped = {}
    for row in rows:
        world_x, world_y, reason = project_observation(row, calibration)
        if reason is None:
            grouped.setdefault(row["track_id"], []).append((row, world_x, world_y))

    segments = []
    for track_id, observations in grouped.items():
        observations.sort(key=lambda item: item[0]["frame_index"])
        current = []
        chunks = []
        for observation in observations:
            if current and observation[0]["frame_index"] - current[-1][0]["frame_index"] > max_gap_frames:
                chunks.append(current)
                current = []
            current.append(observation)
        if current:
            chunks.append(current)
        for chunk in chunks:
            first, last = chunk[0][0], chunk[-1][0]
            teams = Counter(item[0].get("team_id") or "unknown" for item in chunk)
            record_id = f"track-{track_id}-frame-{first['frame_index']}"
            segments.append({
                "record_id": record_id,
                "track_id": track_id,
                "start_frame": first["frame_index"],
                "end_frame": last["frame_index"],
                "start_time_s": first["timestamp_s"],
                "end_time_s": last["timestamp_s"],
                "observation_count": len(chunk),
                "field_frames": [item[0]["frame_index"] for item in chunk],
                "team_id": teams.most_common(1)[0][0],
            })
    return sorted(segments, key=lambda segment: (segment["start_frame"], segment["track_id"], segment["record_id"]))


def merge_groups(segments, operations, input_signature=None):
    """按仍有效的融合记录建立身份组，并标记指向已不存在片段的旧操作。"""
    parent = {segment["record_id"]: segment["record_id"] for segment in segments}

    def find(record_id):
        """查找片段所属融合组，并压缩已遍历的路径。"""
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    known = set(parent)
    history = []
    for operation in operations:
        if operation.get("undone"):
            status = "undone"
        elif operation.get("execution", "manual") == "auto" and (
            operation.get("superseded") or input_signature is None
            or operation.get("input_signature") != input_signature
        ):
            status = "stale"
        elif operation.get("source_record_id") not in known or operation.get("target_record_id") not in known:
            status = "stale"
        else:
            status = "active"
            source = find(operation["source_record_id"])
            target = find(operation["target_record_id"])
            if source != target:
                parent[source] = target
        history.append({**operation, "execution": operation.get("execution", "manual"), "status": status})

    sizes = Counter(find(record_id) for record_id in known)
    for segment in segments:
        root = find(segment["record_id"])
        segment["merge_group_id"] = root
        segment["merge_group_size"] = sizes[root]
    return history
