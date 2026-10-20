"""复用人工绑定和有效融合，生成独立、离线可用的球员跑动探索页。"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math

from .calibration import load_calibration, validate_calibration
from .field_review import field_segments, merge_groups
from .identity import load_players, resolve_identity, validate_bindings
from .metrics import calculate_motion, summarize
from .pitch_geometry import pitch_polylines
from .reassociation import input_signature
from .storage import file_hash, new_id, read_json, read_rows, read_yaml, write_json


def snapshot_inputs(run_dir):
    """记录完整标定及身份输入的摘要，检测导出期间的输入变化。"""
    manifest = read_json(run_dir / "manifest.json")
    payload = {
        "manifest": manifest,
        "tracks_sha256": file_hash(run_dir / "tracks.parquet"),
        "calibration": load_calibration(run_dir),
        "bindings": read_yaml(run_dir / "bindings.yaml"),
        "players": load_players(Path(manifest["config"]["players"])),
        "review": read_json(run_dir / "field_review.json") if (run_dir / "field_review.json").exists() else {"operations": []},
    }
    return payload


def resolved_rows(rows, bindings, segments):
    """在副本内继承无歧义融合身份，并按球员/帧去重；显式绑定优先。"""
    grouped = defaultdict(list)
    for segment in segments:
        grouped[segment["merge_group_id"]].append(segment)
    inherited, conflicts = [], []
    for group_id, members in grouped.items():
        if len(members) < 2:
            continue
        identities = set()
        for member in members:
            for entry in bindings:
                if entry["action"] == "bind" and entry["track_id"] == member["track_id"] and max(entry["start_frame"], member["start_frame"]) < min(entry["end_frame"], member["end_frame"] + 1):
                    identities.add((entry["player_id"], entry["team_id"]))
        if len(identities) != 1:
            conflicts.append({"group_id": group_id, "reason": "identity_conflict" if identities else "no_bound_identity"})
            continue
        player_id, team_id = next(iter(identities))
        for member in members:
            intervals = [(member["start_frame"], member["end_frame"] + 1)]
            # 显式绑定与排除都不得被融合继承覆盖。
            for entry in bindings:
                if entry["track_id"] != member["track_id"]:
                    continue
                remaining = []
                for start, end in intervals:
                    if end <= entry["start_frame"] or start >= entry["end_frame"]:
                        remaining.append((start, end))
                    else:
                        if start < entry["start_frame"]:
                            remaining.append((start, entry["start_frame"]))
                        if end > entry["end_frame"]:
                            remaining.append((entry["end_frame"], end))
                intervals = remaining
            inherited.extend({"track_id": member["track_id"], "start_frame": start, "end_frame": end,
                              "action": "bind", "player_id": player_id, "team_id": team_id,
                              "merge_group_id": group_id} for start, end in intervals)
    effective = bindings + inherited
    winners, unbound, duplicates = {}, [], []
    for row in rows:
        player_id, _, index, _ = resolve_identity(row, effective)
        if player_id is None:
            unbound.append(row)
            continue
        key = (player_id, row["frame_index"])
        rank = (index < len(bindings), row["confidence"], -row["track_id"])
        previous = winners.get(key)
        if previous is None or rank > previous[0]:
            if previous is not None:
                duplicates.append({"player_id": player_id, "frame_index": key[1], "kept_track": row["track_id"], "removed_track": previous[1]["track_id"]})
            winners[key] = (rank, row)
        else:
            duplicates.append({"player_id": player_id, "frame_index": key[1], "kept_track": previous[1]["track_id"], "removed_track": row["track_id"]})
    selected = unbound + [item[1] for item in winners.values()]
    selected.sort(key=lambda row: (row["frame_index"], row["track_id"]))
    return selected, effective, {"inherited_bindings": inherited, "unresolved_groups": conflicts, "duplicate_observations": duplicates}


def player_payload(motions, summary, names, manifest, calibration):
    """构造全帧位置、时间加权热图和逐秒平均速度，不填补观测缺口。"""
    by_player = defaultdict(list)
    for row in motions:
        if row["player_id"] and row["valid"]:
            by_player[row["player_id"]].append(row)
    fps = manifest["video"]["fps"]
    start_s = manifest["start_frame"] / fps
    duration = (manifest["end_frame"] - manifest["start_frame"]) / fps
    length, width = calibration["pitch_length_m"], calibration["pitch_width_m"]
    result = []
    for stats in summary:
        player_id = stats["player_id"]
        observations = by_player[player_id]
        heat = [0.0] * (35 * 23)
        bins = [[0.0, 0.0] for _ in range(math.ceil(duration))]
        points = []
        for row in observations:
            points.append([row["frame_index"], round(row["x_smooth_m"], 4), round(row["y_smooth_m"], 4), row["segment_id"], row["track_id"]])
            delta_t = row["delta_t_s"]
            if delta_t is None:
                continue
            column = min(34, int(row["x_smooth_m"] / length * 35))
            line = min(22, int(row["y_smooth_m"] / width * 23))
            heat[line * 35 + column] += delta_t
            end = row["timestamp_s"] - start_s
            begin = end - delta_t
            for second in range(max(0, math.floor(begin)), min(len(bins), math.ceil(end))):
                overlap = max(0, min(end, second + 1) - max(begin, second))
                bins[second][0] += row["speed_mps"] * overlap
                bins[second][1] += overlap
        # 这里只按用户填写的身份前缀组织显示，不把 home 标签改成客队。
        group = "barca" if player_id.startswith("barca_") else "girona" if player_id.startswith("girona_") else "referee" if player_id.startswith("referee_") else "other"
        result.append({**stats, "name": names[player_id]["display_name"], "group": group,
                       "tracks": sorted({row["track_id"] for row in observations}),
                       "points": points, "heat": [round(value, 6) for value in heat],
                       "speed_bins": [[round(distance / seconds * 3.6, 4) if seconds else None, round(seconds, 6)] for distance, seconds in bins]})
    return result


def create_motion_visualization(run_dir):
    """保存输入快照、CSV 与独立 HTML；不改写原始数据或注册为正式跑动报告。"""
    run_dir = Path(run_dir).resolve()
    before = snapshot_inputs(run_dir)
    manifest, calibration = before["manifest"], before["calibration"]
    if manifest["status"] != "complete":
        raise ValueError("追踪运行尚未完成")
    validate_calibration(calibration, manifest)
    if file_hash(Path(manifest["config"]["video"])) != manifest["video_sha256"]:
        raise ValueError("源视频摘要变化，拒绝导出")
    bindings = validate_bindings(before["bindings"], manifest, before["players"], read_json(run_dir / "catalog.json"))
    rows = list(read_rows(run_dir / "tracks.parquet"))
    segments = field_segments(rows, manifest, calibration)
    operations = before["review"].get("operations", [])
    signature = input_signature(run_dir) if any(item.get("execution") == "auto" for item in operations) else None
    history = merge_groups(segments, operations, signature)
    selected, effective, resolution = resolved_rows(rows, bindings, segments)
    motions = list(calculate_motion(selected, manifest, calibration, effective))
    summary = summarize(motions, manifest, effective)
    players = player_payload(motions, summary, before["players"], manifest, calibration)
    quality = dict(Counter(row["reason"] or "valid" for row in motions))
    payload = {
        "schema_version": 1, "version": "player-motion-explorer-v1", "run_id": manifest["run_id"],
        "clip_id": manifest["config"]["clip_id"], "created_at": datetime.now(timezone.utc).isoformat(),
        "start_frame": manifest["start_frame"], "end_frame": manifest["end_frame"], "fps": manifest["video"]["fps"],
        "stride": manifest["config"]["stride"], "length": calibration["pitch_length_m"], "width": calibration["pitch_width_m"],
        "pitch_lines": [points.tolist() for _, points in pitch_polylines(calibration["pitch_length_m"], calibration["pitch_width_m"], spec=calibration.get("pitch_spec"), model=calibration.get("pitch_model", "eleven"))],
        "players": players, "quality": quality, "resolution": resolution,
        "history": history, "calibration_id": calibration.get("calibration_id", "static"),
        "quality_parameters": {key: manifest["config"][key] for key in ["speed_limit_mps", "min_cutoff_hz", "filter_beta"]},
    }
    # 写入前检查标注、融合与标定在计算期间没有变化。
    if before != snapshot_inputs(run_dir):
        raise ValueError("导出期间输入发生变化，请重试")
    folder = run_dir / "reports" / new_id("report_viz")
    folder.mkdir(parents=True)
    write_json(folder / "input_snapshot.json", before)
    payload["input_sha256"] = file_hash(folder / "input_snapshot.json")
    write_json(folder / "data.json", payload)
    write_json(folder / "diagnostics.json", {"quality": quality, **resolution, "history": history})
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as output:
        keys = ["player_id", "team_id", "distance_m", "valid_seconds", "coverage", "avg_speed_mps"]
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    template = (Path(__file__).parent / "web" / "player-motion-report.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (folder / "report.html").write_text(template.replace("__MOTION_DATA__", encoded), encoding="utf-8")
    return folder
