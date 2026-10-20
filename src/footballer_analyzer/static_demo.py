"""导出不依赖后端的评委演示站，保留原始输入并核对报告快照。"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .calibration import project_observation
from .field_review import field_segments, merge_groups
from .identity import resolve_identity, validate_bindings
from .motion_visualization import snapshot_inputs, resolved_rows
from .reassociation import input_signature
from .storage import file_hash, read_json, read_rows


def encoded(value):
    """生成有限数值且可安全嵌入脚本的紧凑 JSON。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def digest(value):
    """计算可重复的结构化输入摘要。"""
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def public_history(history):
    """只保留展示历史必需的字段，避免传播运行配置和机器路径。"""
    keys = ("operation_id", "source_record_id", "target_record_id", "target_frame", "created_at",
            "undone", "undone_at", "execution", "status", "score", "high_threshold", "low_threshold",
            "algorithm", "algorithm_version", "batch_id")
    return [{key: row[key] for key in keys if key in row} for row in history]


def motion_page(template, payload):
    """将原有跑动报告适配为同一静态站的第三个页面。"""
    template = template.replace("__MOTION_DATA__", encoded(payload))
    template = template.replace("href=\"/\"", 'href="index.html"').replace('href="/field-review"', 'href="review.html"')
    template = template.replace('<a href="#comparison">跑动对比 ↓</a>', '<a id="motion-link" href="motion.html" aria-current="page">跑动分析</a>')
    template = template.replace("FOOTBALLER / MOTION LAB", "FOOTBALLER / DEMO")
    template = template.replace("球员跑动观察</h1>", "球员跑动观察 <small style=\"font-size:14px\">演示数据</small></h1>")
    template = template.replace("const duration=(D.end_frame-D.start_frame)/D.fps, origin=location.protocol==='file:'?'http://127.0.0.1:8778':'';", "const duration=(D.end_frame-D.start_frame)/D.fps;")
    template = template.replace("`${origin}/field-review?run=${encodeURIComponent(D.run_id)}&frame=${frame}`", "staticLink('index.html')")
    template = template.replace("`${origin}/?run=${encodeURIComponent(D.run_id)}`", "staticLink('index.html')")
    template = template.replace("`${origin}/field-review?run=${encodeURIComponent(D.run_id)}`", "staticLink('review.html')")
    template = template.replace("function selectPlayer(p){selected=p;", "function selectPlayer(p){selected=p;syncNav();")
    template = template.replace("frameFloat=frame;drawReplay();}", "frameFloat=frame;syncNav();drawReplay();}")
    # 播放期间也同步导航，避免跨页面后回到旧帧。
    template = template.replace("el('clock').textContent=timeLabel", "syncNav();el('clock').textContent=timeLabel")
    init = """
function staticLink(page){return page+'?'+new URLSearchParams({frame:String(frame),player:selected.player_id});}
function syncNav(){el('main-link').href=staticLink('index.html');el('review-link').href=staticLink('review.html');el('motion-link').href=staticLink('motion.html');history.replaceState(null,'','?'+new URLSearchParams({frame:String(frame),player:selected.player_id}));}
const incoming=new URLSearchParams(location.search),initialFrame=Number(incoming.get('frame'));
if(incoming.has('frame')&&Number.isFinite(initialFrame))frame=Math.max(D.start_frame,Math.min(D.end_frame-1,Math.round(initialFrame)));
frameFloat=frame;selected=D.players.find(p=>p.player_id===incoming.get('player'))||selected;
"""
    template = template.replace("buildOptions();selectPlayer(selected);new ResizeObserver", init + "buildOptions();selectPlayer(selected);new ResizeObserver")
    template = template.replace("这份数据如何读</strong>", "仅统计本片段，米制精度尚未独立验证</strong>")
    template = template.replace("link.href=canvas.toDataURL();link.click();", "canvas.toBlob(blob=>{if(blob)download(blob,selected.player_id+'_heatmap.png','image/png');},'image/png');")
    return template


def export_media(source, output, start, end, fps):
    """逐帧解码导出 JPEG，并用相同区间生成浏览器兼容的视频。"""
    import cv2

    frames = output / "frames"
    frames.mkdir()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("无法读取源视频")
    try:
        # 顺序解码，避免压缩视频随机定位造成源帧偏移。
        for frame_index in range(end):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"源视频在帧 {frame_index} 提前结束")
            if frame_index < start:
                continue
            image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
            ok = cv2.imwrite(str(frames / f"{frame_index:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise ValueError(f"导出源帧 {frame_index} 失败")
            if (frame_index - start) % 250 == 0:
                print(f"已导出源帧 {frame_index}/{end - 1}", flush=True)
    finally:
        capture.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
                    "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,scale=1280:720",
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-r", str(fps), "-frames:v", str(end-start), "-movflags", "+faststart", str(output / "video.mp4")], check=True)
    probe = cv2.VideoCapture(str(output / "video.mp4"))
    try:
        if int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) != end-start or abs(probe.get(cv2.CAP_PROP_FPS)-fps) > .001:
            raise ValueError("导出视频的帧数或帧率与输入不一致")
    finally:
        probe.release()


def export_demo(run_dir, report_dir, output):
    """校验快照、导出三个只读页面，并在媒体导出后再次验证输入不变。"""
    run_dir, report_dir, output = [Path(value).resolve() for value in (run_dir, report_dir, output)]
    if output.exists():
        raise ValueError("输出目录已存在，请指定新的目录，以免覆盖现场文件")
    before = snapshot_inputs(run_dir)
    if before != read_json(report_dir / "input_snapshot.json"):
        raise ValueError("选定报告已过期，请先生成与当前标注一致的跑动可视化报告")
    motion = read_json(report_dir / "data.json")
    if motion["input_sha256"] != file_hash(report_dir / "input_snapshot.json"):
        raise ValueError("报告输入摘要不一致")
    manifest, calibration = before["manifest"], before["calibration"]
    source = Path(manifest["config"]["video"])
    source_hash = file_hash(source)
    detections_hash = file_hash(run_dir / "detections.parquet")
    if source_hash != manifest["video_sha256"]:
        raise ValueError("源视频摘要变化，拒绝导出")
    start, end, fps = manifest["start_frame"], manifest["end_frame"], manifest["video"]["fps"]
    if (motion["run_id"], motion["start_frame"], motion["end_frame"], motion["fps"]) != (manifest["run_id"], start, end, fps):
        raise ValueError("报告与运行的帧区间不一致")
    rows = list(read_rows(run_dir / "tracks.parquet"))
    bindings = validate_bindings(before["bindings"], manifest, before["players"], read_json(run_dir / "catalog.json"))
    segments = field_segments(rows, manifest, calibration)
    operations = before["review"].get("operations", [])
    signature = input_signature(run_dir) if any(op.get("execution") == "auto" for op in operations) else None
    history = public_history(merge_groups(segments, operations, signature))
    selected, effective, resolution = resolved_rows(rows, bindings, segments)
    if resolution != motion["resolution"] or history != public_history(motion["history"]):
        raise ValueError("报告融合口径与当前运行不一致")
    canonical = {(row["frame_index"], row["track_id"]) for row in selected}
    record_map = {(frame, seg["track_id"]): seg["record_id"] for seg in segments for frame in seg["field_frames"]}
    points = {(point[0], point[4]): point[1:3] for player in motion["players"] for point in player["points"]}
    frames = [{"tracks": [], "detections": []} for _ in range(end-start)]
    for row in rows:
        frame = row["frame_index"]
        if not start <= frame < end:
            continue
        player, _, _, reason = resolve_identity(row, effective)
        key = (frame, row["track_id"])
        x, y, projection_reason = project_observation(row, calibration)
        # 已绑定人物优先采用报告的有效点；无效点不作为统计位置显示。
        position = points.get(key) if player else [round(x, 4), round(y, 4)] if projection_reason is None else None
        frames[frame-start]["tracks"].append([
            row["track_id"], *[round(row[key], 3) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")],
            round(row["confidence"], 4), player, position,
            reason or (None if position else projection_reason or "无有效统计位置"),
            record_map.get(key), key in canonical])
    for row in read_rows(run_dir / "detections.parquet"):
        if start <= row["frame_index"] < end:
            frames[row["frame_index"]-start]["detections"].append([
                *[round(row[key], 3) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")], round(row["confidence"], 4)])
    for seg in segments:
        identities = Counter(resolve_identity(row, effective)[0] for row in rows if row["track_id"] == seg["track_id"] and seg["start_frame"] <= row["frame_index"] <= seg["end_frame"])
        seg["players"] = sorted(player for player in identities if player)
    hashes = {"inputs": digest(before), "tracks": before["tracks_sha256"], "detections": detections_hash,
              "source_video": source_hash, "report_data": file_hash(report_dir / "data.json"),
              "calibration": digest(before["calibration"]), "bindings": digest(before["bindings"]), "history": digest(before["review"])}
    payload = {"schema_version": 1, "run_id": manifest["run_id"], "clip_id": motion["clip_id"],
               "start_frame": start, "end_frame": end, "fps": fps, "source_width": manifest["video"]["width"],
               "source_height": manifest["video"]["height"], "length": motion["length"], "width": motion["width"],
               "pitch_lines": motion["pitch_lines"], "calibration_id": motion["calibration_id"],
               "players": [{key: player[key] for key in ("player_id", "name", "group", "tracks", "distance_m", "valid_seconds", "coverage", "avg_speed_mps")} for player in motion["players"]],
               "segments": segments, "history": history, "frames": frames, "hashes": hashes,
               "track_count": len({row["track_id"] for row in rows}), "created_at": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True)
    web = Path(__file__).parent / "web"
    for name in ("index.html", "review.html", "demo.css", "demo.js", "preview.py", "preview.cmd"):
        shutil.copyfile(web / "demo" / name, output / name)
    (output / ".nojekyll").touch()
    (output / "data.js").write_text("window.DEMO_DATA=" + encoded(payload) + ";", encoding="utf-8")
    public_motion = {**motion, "history": history}
    (output / "motion.html").write_text(motion_page((web / "player-motion-report.html").read_text(encoding="utf-8"), public_motion), encoding="utf-8")
    (output / "motion-data.json").write_text(encoded(public_motion), encoding="utf-8")
    shutil.copyfile(report_dir / "summary.csv", output / "summary.csv")
    (output / "input-hashes.json").write_text(encoded(hashes), encoding="utf-8")
    export_media(source, output, start, end, fps)
    if before != snapshot_inputs(run_dir) or detections_hash != file_hash(run_dir / "detections.parquet") or source_hash != file_hash(source):
        (output / "EXPORT_FAILED.txt").write_text("输入在导出期间变化，此包不能发布。", encoding="utf-8")
        raise ValueError("导出期间输入变化，请使用新的输出目录重试")
    check = {"input_unchanged": True, "frames": end-start, "fps": fps, "players": len(motion["players"]),
             "segments": len(segments), "groups": len({seg["merge_group_id"] for seg in segments}), "hashes": hashes}
    (output / "export-validation.json").write_text(encoded(check), encoding="utf-8")
    return check


def main():
    """解析运行、可视化报告和全新的导出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(encoded(export_demo(args.run, args.report, args.output)))


if __name__ == "__main__":
    main()
