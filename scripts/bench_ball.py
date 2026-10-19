"""比较足球模型与切片推理，保存逐帧候选、原图及独立位置标注的验收结果。

位置参考来自原图核对，标注记录区分人工与 Agent；遮挡与出画单独统计。
所有帧号均为源视频帧号，输出放在忽略的 outputs 下，不修改人物分析运行。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from footballer_analyzer.ball_detection import detect, suppress_duplicates, tile_origins

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """将结构化实验记录写入 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    """计算输入文件指纹，便于识别重复素材和复现实验。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frames(clip, step):
    """顺序解码源区间并按步长返回帧，实际解码不足时拒绝静默截断。"""
    capture = cv2.VideoCapture(str(ROOT / clip["video"]))
    if not capture.isOpened():
        raise ValueError(f"无法打开素材：{clip['video']}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, clip["start"])
    try:
        for frame_number in range(clip["start"], clip["end"]):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"素材 {clip['id']} 在源帧 {frame_number} 提前结束")
            if (frame_number - clip["start"]) % step == 0:
                yield frame_number, frame
    finally:
        capture.release()




def prepare(config, step):
    """导出无预测标记的抽样原图和待标注清单，已有标注不会被覆盖。"""
    destination = ROOT / config["output"]
    entries = []
    for clip in config["clips"]:
        capture = cv2.VideoCapture(str(ROOT / clip["video"]))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        clip_entries = []
        for frame_number, frame in frames(clip, step):
            frame_path = destination / "frames" / clip["id"] / f"{frame_number:06d}.jpg"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
            clip_entries.append({"clip": clip["id"], "frame": frame_number, "image": str(frame_path.relative_to(ROOT)), "visibility": "unreviewed", "center": None})
        entries.extend(clip_entries)
        write_json(destination / "sources" / f"{clip['id']}.json", {**clip, "sha256": sha256(ROOT / clip["video"]), "fps": fps, "sample_step": step})
        print(f"准备 {clip['id']}：{len(clip_entries)} 帧", flush=True)
    annotation_path = destination / "annotations.json"
    if not annotation_path.exists():
        write_json(annotation_path, {"schema_version": 1, "tolerance_px": 12, "annotations": entries})
    write_json(destination / "sample_manifest.json", {"clips": config["clips"], "frames": entries})


def infer(config, args):
    """运行指定模型配置并缓存全部球候选，记录耗时和模型指纹。"""
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    weights = ROOT / config["models"][args.model]
    model = YOLO(str(weights))
    identifier = f"{args.model}_size{args.imgsz}_tile{args.tile}_step{args.step}_conf{args.confidence}_overlap{args.overlap}"
    destination = ROOT / config["output"] / "predictions" / identifier
    model_info = {"id": identifier, "weights": str(weights.relative_to(ROOT)), "sha256": sha256(weights), "classes": model.names, "image_size": args.imgsz, "tile_size": args.tile, "overlap": args.overlap, "confidence": args.confidence, "step": args.step, "threads": args.threads}
    metadata_path = destination / "model.json"
    if metadata_path.exists() and read_json(metadata_path) != json.loads(json.dumps(model_info)):
        raise ValueError(f"缓存模型配置已变化，请使用新的输出目录：{destination}")
    write_json(metadata_path, model_info)
    for clip in config["clips"]:
        if args.clips and clip["id"] not in args.clips:
            continue
        output = destination / f"{clip['id']}.json"
        if output.exists() and not args.overwrite:
            cached = read_json(output)
            if cached["clip"] != clip or cached["source_sha256"] != sha256(ROOT / clip["video"]):
                raise ValueError(f"缓存素材已变化，请使用 --overwrite：{output}")
            print(f"已有缓存：{output}", flush=True)
            continue
        records = []
        started = time.perf_counter()
        for frame_number, frame in frames(clip, args.step):
            candidates = detect(model, frame, args.imgsz, args.tile, args.overlap, args.confidence)
            records.append({"frame": frame_number, "candidates": candidates})
            if len(records) % 25 == 0:
                print(f"{identifier} {clip['id']}：{len(records)} 帧，耗时 {time.perf_counter() - started:.1f}s", flush=True)
        result = {"model": model_info, "clip": clip, "source_sha256": sha256(ROOT / clip["video"]), "elapsed_s": time.perf_counter() - started, "frames": records}
        write_json(output, result)
        print(f"完成 {clip['id']}：{len(records)} 帧，{sum(bool(row['candidates']) for row in records)} 帧有候选，耗时 {result['elapsed_s']:.1f}s", flush=True)


def contact_sheet(config, args):
    """导出原图或候选放大图，辅助逐帧人工核对并保留帧号。"""
    destination = ROOT / config["output"]
    for clip in config["clips"]:
        if args.clips and clip["id"] not in args.clips:
            continue
        predictions = {}
        if args.predictions:
            predictions = {row["frame"]: row["candidates"] for row in read_json(destination / "predictions" / args.predictions / f"{clip['id']}.json")["frames"]}
        cells = []
        for frame_number, frame in frames(clip, args.step):
            if args.frames and frame_number not in args.frames:
                continue
            candidates = predictions.get(frame_number, [])
            if args.region:
                left, top, width, height = args.region
                image = cv2.resize(frame[top:top + height, left:left + width], (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)
                cv2.putText(image, f"{frame_number} origin={left},{top} 2x", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cells.append(image)
            elif args.crops:
                for candidate_index, candidate in enumerate(candidates[:1 if args.top1 else 8]):
                    center = [(candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2]
                    left = max(0, min(frame.shape[1] - 120, int(center[0]) - 60))
                    top = max(0, min(frame.shape[0] - 120, int(center[1]) - 60))
                    image = cv2.resize(frame[top:top + 120, left:left + 120], (240, 240), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(image, f"{frame_number} #{candidate_index} {candidate[4]:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cells.append(image)
            else:
                image = cv2.resize(frame, (960, 540))
                cv2.putText(image, f"{clip['id']} frame {frame_number}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cells.append(image)
        columns = min(len(cells), 5 if args.region else 6 if args.crops else 2)
        if not columns:
            continue
        page_size = columns * (5 if args.crops else 3)
        for offset in range(0, len(cells), page_size):
            batch = cells[offset:offset + page_size]
            height, width = batch[0].shape[:2]
            sheet = np.zeros((height * ((len(batch) + columns - 1) // columns), width * columns, 3), dtype=np.uint8)
            for index, cell in enumerate(batch):
                top, left = divmod(index, columns)
                sheet[top * height:(top + 1) * height, left * width:(left + 1) * width] = cell
            folder = destination / "review" / (args.predictions or "unmarked")
            if args.region:
                folder = folder / ("region_" + "_".join(map(str, args.region)))
            elif args.top1:
                folder = folder / "top1"
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{clip['id']}_{offset // page_size:02d}.jpg"), sheet)


def evaluate(config, args):
    """按独立标注报告候选召回、最高分定位正确率和误检；漏评帧不算成功。"""
    destination = ROOT / config["output"]
    annotation_path = args.annotations or destination / "annotations.json"
    labels = read_json(annotation_path)
    if "groups" in labels:
        labels["annotations"] = [{"clip": clip_id, "frame": group["start"] + index * group["step"], "visibility": "visible" if center else "unreviewed", "center": center} for clip_id, group in labels["groups"].items() for index, center in enumerate(group["centers"])]
    tolerance = labels["tolerance_px"]
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("标注容差必须为正有限值")
    seen = set()
    for label in labels["annotations"]:
        key = (label["clip"], label["frame"])
        if key in seen:
            raise ValueError(f"重复标注：{key}")
        seen.add(key)
        if label["visibility"] == "visible" and (label["center"] is None or len(label["center"]) != 2 or not np.isfinite(label["center"]).all()):
            raise ValueError(f"球心标注无效：{key}")
    summaries = {}
    for clip in config["clips"]:
        if args.clips and clip["id"] not in args.clips:
            continue
        predictions = {row["frame"]: row["candidates"] for row in read_json(destination / "predictions" / args.predictions / f"{clip['id']}.json")["frames"]}
        counts = {"visible": 0, "candidate_hits": 0, "top1_hits": 0, "false_candidates": 0, "absent": 0, "absent_with_prediction": 0, "unreviewed": 0, "ambiguous": 0}
        misses = []
        for label in labels["annotations"]:
            if label["clip"] != clip["id"]:
                continue
            visibility = label["visibility"]
            if visibility in {"unreviewed", "ambiguous"}:
                counts[visibility] += 1
                continue
            candidates = [entry for entry in predictions.get(label["frame"], []) if entry[4] >= args.confidence]
            if visibility == "absent":
                counts["absent"] += 1
                counts["absent_with_prediction"] += bool(candidates)
                continue
            if visibility != "visible" or label["center"] is None:
                raise ValueError(f"标注无效：{label}")
            distances = [float(np.linalg.norm(np.asarray([(entry[0] + entry[2]) / 2, (entry[1] + entry[3]) / 2]) - label["center"])) for entry in candidates]
            hit = any(distance <= tolerance for distance in distances)
            counts["visible"] += 1
            counts["candidate_hits"] += hit
            counts["top1_hits"] += bool(distances and distances[0] <= tolerance)
            counts["false_candidates"] += sum(distance > tolerance for distance in distances)
            if not hit:
                misses.append(label["frame"])
        total = counts["visible"] + counts["absent"] + counts["unreviewed"] + counts["ambiguous"]
        summaries[clip["id"]] = {**counts, "labeled_samples": total, "candidate_recall": counts["candidate_hits"] / counts["visible"] if counts["visible"] else None, "top1_accuracy": counts["top1_hits"] / counts["visible"] if counts["visible"] else None, "conservative_hit_fraction": counts["top1_hits"] / total if total else None, "missed_frames": misses}
    output = {"predictions": args.predictions, "confidence": args.confidence, "tolerance_px": tolerance, "annotation_sha256": sha256(annotation_path), "annotation_method": labels.get("annotation_method", "unspecified"), "human_reviewed": labels.get("human_reviewed", False), "clips": summaries}
    write_json(destination / "scores" / f"{args.predictions}_conf{args.confidence}.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def filter_predictions(config, args):
    """离线筛选已有候选缓存，另外保存单球观测与可选核对视频。"""
    from footballer_analyzer.ball_tracking import BallSelectionConfig, select_ball_candidates

    destination = ROOT / config["output"]
    settings = BallSelectionConfig()
    for clip in config["clips"]:
        if args.clips and clip["id"] not in args.clips:
            continue
        source = destination / "predictions" / args.predictions / f"{clip['id']}.json"
        payload = read_json(source)
        capture = cv2.VideoCapture(str(ROOT / clip["video"]))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        selected = select_ball_candidates(payload["frames"], fps, (width, height), settings)
        filtered = {**payload, "raw_predictions_sha256": sha256(source), "filter": asdict(settings), "frames": [{"frame": row["frame"], "candidates": [candidate] if candidate else [], "observed": candidate is not None} for row, candidate in zip(payload["frames"], selected)]}
        output = destination / "predictions" / (args.predictions + "_temporal") / f"{clip['id']}.json"
        write_json(output, filtered)
        if args.video:
            video_path = output.with_suffix(".mp4")
            step = payload["model"]["step"]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (width, height))
            if not writer.isOpened():
                raise ValueError(f"无法生成核对视频：{video_path}")
            selected_by_frame = {row["frame"]: candidate for row, candidate in zip(payload["frames"], selected)}
            try:
                for frame_number, frame in frames(clip, step):
                    candidate = selected_by_frame.get(frame_number)
                    if candidate:
                        left, top, right, bottom = map(round, candidate[:4])
                        cv2.rectangle(frame, (left - 4, top - 4), (right + 4, bottom + 4), (0, 255, 255), 2)
                    label = f"frame {frame_number} | {'observed' if candidate else 'missing'}"
                    cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    writer.write(frame)
            finally:
                writer.release()
        print(f"{clip['id']}：筛选后 {sum(candidate is not None for candidate in selected)}/{len(selected)} 帧有观测", flush=True)


def main():
    """提供准备样本、推理、检查图和计算验收指标的命令入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "infer", "sheet", "evaluate", "filter"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ball_benchmark.json")
    parser.add_argument("--model", default="football_v11n")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--predictions")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--crops", action="store_true")
    parser.add_argument("--top1", action="store_true")
    parser.add_argument("--region", nargs=4, type=int)
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.step <= 0 or args.imgsz <= 0 or args.threads <= 0 or args.tile < 0 or not 0 <= args.overlap < 1 or not 0 < args.confidence < 1:
        parser.error("步长、切片、重叠比例或置信度参数无效")
    if args.region and (min(args.region[:2]) < 0 or min(args.region[2:]) <= 0):
        parser.error("裁剪起点不能为负，尺寸必须为正")
    config = read_json(args.config)
    if args.action == "prepare":
        prepare(config, args.step)
    elif args.action == "infer":
        infer(config, args)
    elif args.action == "sheet":
        contact_sheet(config, args)
    elif args.action == "evaluate":
        if not args.predictions:
            parser.error("验收需要 --predictions")
        evaluate(config, args)
    elif args.action == "filter":
        if not args.predictions:
            parser.error("筛选需要 --predictions")
        filter_predictions(config, args)


if __name__ == "__main__":
    main()
