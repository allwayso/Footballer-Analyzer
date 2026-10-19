"""批量为已完成人物和场地标定的素材补跑足球；逐运行记录成败并保留原件指纹。"""

import argparse
from pathlib import Path

from footballer_analyzer.ball_pipeline import DEFAULT_MODEL, load_ball_result, model_preset, track_ball_run
from footballer_analyzer.config import ROOT
from footballer_analyzer.storage import file_hash, read_json, write_json


def eligible_runs(output_root):
    """只选当前输出根目录中已追踪且已标定的真实素材，跳过仅白线和测试夹具。"""
    for path in sorted(output_root.glob("run_*/manifest.json")):
        manifest = read_json(path)
        video = Path(manifest["config"]["video"]).resolve()
        if manifest["status"] == "complete" and manifest.get("kind") != "white_line_review" and (path.parent / "calibration.yaml").is_file() and (path.parent / "tracks.parquet").is_file() and video.is_relative_to((ROOT / "data/clips").resolve()):
            yield path.parent


def protected_hashes(run_dir):
    """记录已有追踪、标定、身份与运行清单，确认补跑没有改写原始输入。"""
    return {name: file_hash(run_dir / name) for name in ["manifest.json", "tracks.parquet", "detections.parquet", "calibration.yaml", "bindings.yaml"] if (run_dir / name).is_file()}


def main():
    """顺序执行可恢复的补跑，失败继续下一个素材，汇总写入独立 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    preset = model_preset(args.model)
    weights_hash = file_hash(ROOT / preset["weights"])
    runs = list(eligible_runs(args.output_root))
    print(f"准备补跑 {len(runs)} 个已标定人物运行，模型 {args.model}", flush=True)
    report = {"model_id": args.model, "runs": [], "status": "running"}
    destination = args.output_root / "ball-batch" / f"{args.model}.json"
    for run_dir in runs:
        manifest = read_json(run_dir / "manifest.json")
        print(f"{run_dir.name} · {manifest['config']['clip_id']} · [{manifest['start_frame']}, {manifest['end_frame']})", flush=True)
        if args.dry_run:
            continue
        before = protected_hashes(run_dir)
        entry = {"run_id": run_dir.name, "clip_id": manifest["config"]["clip_id"], "protected_before": before}
        try:
            existing = load_ball_result(run_dir, args.model)
            reusable = existing and existing["manifest"]["weights_sha256"] == weights_hash and existing["manifest"]["video_sha256"] == file_hash(Path(manifest["config"]["video"])) and existing["manifest"].get("pipeline_version") == 1 and (existing["manifest"]["start_frame"], existing["manifest"]["end_frame"]) == (manifest["start_frame"], manifest["end_frame"])

            def progress(stage, processed, total):
                """输出可定位到运行和源区间的批处理进度。"""
                print(f"{run_dir.name} · {stage} · {processed}/{total}", flush=True)

            result = existing["manifest"] if reusable else track_ball_run(run_dir, args.model, progress)
            entry.update(status="complete", reused=bool(reusable), result_id=result["result_id"], observed_frames=result["observed_frames"], total_frames=result["end_frame"] - result["start_frame"])
        except Exception as error:
            entry.update(status="failed", error=str(error))
        entry["protected_after"] = protected_hashes(run_dir)
        if entry["protected_after"] != before:
            raise RuntimeError(f"原件指纹发生变化，停止批处理：{run_dir.name}")
        report["runs"].append(entry)
        write_json(destination, report)
    if not args.dry_run:
        report["status"] = "failed" if any(entry["status"] == "failed" for entry in report["runs"]) else "complete"
        write_json(destination, report)
        print(f"批处理 {report['status']}：{destination}", flush=True)
        return 1 if report["status"] == "failed" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
