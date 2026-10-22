"""提供回环复盘服务：素材分析任务、源帧、轨迹、标定、身份编辑与报告生成。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
import json
import mimetypes
import re
import threading

import cv2

from .calibration import build_calibration, calibration_directory, load_calibration, project_observation, save_calibration
from .ball_pipeline import DEFAULT_MODEL, ball_models, ball_versions, load_ball_result, model_preset, track_ball_run
from .config import ROOT, config_from_dict, load_config
from .identity import load_players, validate_bindings
from .pitch_geometry import pitch_polylines
from .pitch_models import PRESETS, pitch_landmarks, pitch_spec
from .storage import file_hash, new_id, read_json, read_rows, read_yaml, write_json, write_yaml
from .video import inspect_video, read_frame

from .field_review import field_segments, merge_groups
from .reassociation import (association_view, commit_result, compute_result,
                            input_signature, load_review, pair_key)

WEB = Path(__file__).parent / "web"


def pitch_definition(spec):
    """将参数快照转换为前端共用的标线和具名参考点。"""
    return {"spec": spec, "landmarks": pitch_landmarks(spec), "lines": [{"name": name, "points": points.tolist()} for name, points in pitch_polylines(spec=spec)]}


class ReviewApp:
    """管理输出根目录、轨迹缓存和编辑锁，供本地 HTTP 处理器共享。"""

    def __init__(self, output_root, clips_root=ROOT / "data/clips"):
        """解析素材与输出根目录，初始化共享锁、轨迹缓存和后台任务状态。"""
        self.output_root = Path(output_root).resolve()
        self.lock = threading.RLock()
        self.track_cache = {}
        self.material_id_cache = {}
        self.calibration_jobs = {}
        self.association_jobs = {}
        self.clips_root = Path(clips_root).resolve()
        self.analysis_job = {"status": "idle"}
        self.ball_job = {"status": "idle"}

    def clip_path(self, clip_id):
        """解析素材库内的视频文件；拒绝越界、子目录及非视频文件。"""
        if not isinstance(clip_id, str) or not clip_id or "/" in clip_id or "\\" in clip_id:
            raise ValueError("素材标识无效")
        path = (self.clips_root / clip_id).resolve()
        if path.parent != self.clips_root or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"} or not path.is_file():
            raise ValueError("找不到该视频素材")
        return path

    def clips(self):
        """列出素材与已完成人物、场地运行数量；仅白线运行不视为人物处理完成。"""
        completed = {}
        for manifest_path in self.output_root.glob("run_*/manifest.json"):
            manifest = read_json(manifest_path)
            if manifest["status"] != "complete" or manifest.get("kind") == "white_line_review":
                continue
            video = str(Path(manifest["config"]["video"]).resolve())
            counts = completed.setdefault(video, {"tracking_runs": 0, "calibrated_runs": 0, "ball_runs": 0, "report_runs": 0})
            counts["tracking_runs"] += 1
            if (manifest_path.parent / "calibration.yaml").is_file():
                counts["calibrated_runs"] += 1
            counts["ball_runs"] += len(ball_versions(manifest_path.parent))
            counts["report_runs"] += sum(read_json(path).get("status") == "complete" for path in (manifest_path.parent / "reports").glob("report_*/manifest.json"))
        entries = []
        for path in sorted(self.clips_root.glob("*")):
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
                try:
                    source = self.clip_path(path.name)
                except ValueError:
                    continue
                stat = source.stat()
                fingerprint = (stat.st_size, stat.st_mtime_ns)
                with self.lock:
                    cached = self.material_id_cache.get(str(source))
                    material_id = cached[1] if cached and cached[0] == fingerprint else file_hash(source)
                    self.material_id_cache[str(source)] = (fingerprint, material_id)
                entries.append({"clip_id": path.name, "material_id": material_id, "size_bytes": stat.st_size, **completed.get(str(source), {"tracking_runs": 0, "calibrated_runs": 0, "ball_runs": 0, "report_runs": 0})})
        return entries

    def start_analysis(self, payload):
        """启动独立人物追踪任务；场地设置只作为后续标定的默认值。"""
        source = self.clip_path(payload.get("clip_id"))
        values = load_config(ROOT / "configs/tracking_quality.yaml").snapshot()
        values.update(video=str(source), output_root=str(self.output_root), clip_id=source.stem, match_id=f"local_{source.stem}")
        preset = pitch_spec(payload.get("pitch_model", "eleven"))
        values.update(pitch_model=preset["model"], pitch_length_m=preset["length_m"], pitch_width_m=preset["width_m"], calibration_method="manual" if preset["model"] == "eight" else "white_lines")
        for key in ["pitch_length_m", "pitch_width_m", "start_frame", "end_frame", "imgsz", "calibration_method"]:
            if key in payload:
                values[key] = payload[key]
        config = config_from_dict(values)
        if config.imgsz not in {1280, 1920}:
            raise ValueError("检测尺寸仅支持 1280 或 1920")
        if not config.weights.is_file():
            raise ValueError(f"缺少本地模型权重：{config.weights}")
        with self.lock:
            if self.analysis_job.get("status") == "running":
                if self.analysis_job["config"] == config.snapshot():
                    return dict(self.analysis_job)
                raise ValueError("已有素材正在分析，请等待完成后再开始")
            job = {"job_id": new_id("analysis"), "status": "running", "stage": "queued", "clip_id": source.name, "processed_frames": 0, "total_frames": 0, "run_id": None, "config": config.snapshot()}
            self.analysis_job = job

        def progress(stage, processed, total, run_dir):
            """原子更新阶段及源帧进度，人物完成后提供可打开的运行标识。"""
            with self.lock:
                job.update(stage=stage, processed_frames=processed, total_frames=total)
                if run_dir is not None:
                    job["run_id"] = run_dir.name

        def worker():
            """顺序分析并发布结果；失败保存原因，保留已经完成的人物缓存。"""
            try:
                from .analysis import analyze_clip

                run_dir = analyze_clip(config, progress)
                with self.lock:
                    job.update(status="complete", stage="awaiting_calibration", run_id=run_dir.name)
            except Exception as error:
                with self.lock:
                    job.update(status="failed", error=str(error))

        threading.Thread(target=worker, daemon=True).start()
        return dict(job)

    def start_ball(self, run_id, payload):
        """启动独立足球任务；相同请求复用运行状态，不同请求等待当前任务结束。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("请先完成人物追踪")
        model_id = payload.get("model_id", DEFAULT_MODEL)
        model_preset(model_id)
        with self.lock:
            if self.ball_job.get("status") == "running":
                if (self.ball_job["run_id"], self.ball_job["model_id"]) == (run_id, model_id):
                    return dict(self.ball_job)
                raise ValueError("已有足球追踪正在运行，请等待完成")
            job = {"job_id": new_id("balljob"), "status": "running", "stage": "queued", "run_id": run_id, "model_id": model_id, "processed_frames": 0, "total_frames": manifest["end_frame"] - manifest["start_frame"]}
            self.ball_job = job

        def progress(stage, processed, total):
            """同步检测与筛选阶段，供页面刷新后恢复进度。"""
            with self.lock:
                job.update(stage=stage, processed_frames=processed, total_frames=total)

        def worker():
            """完成后发布足球版本，失败保留上一份完成结果。"""
            try:
                result = track_ball_run(run_dir, model_id, progress)
                with self.lock:
                    job.update(status="complete", stage="complete", result_id=result["result_id"])
            except Exception as error:
                with self.lock:
                    job.update(status="failed", error=str(error))

        threading.Thread(target=worker, daemon=True).start()
        return dict(job)

    def run_dir(self, run_id):
        """返回根目录内有效运行的绝对路径；非法 ID、越界或缺清单时抛出 ValueError。"""
        if not re.fullmatch(r"run_[A-Za-z0-9_]+", run_id):
            raise ValueError("运行标识无效")
        candidate = (self.output_root / run_id).resolve()
        if candidate.parent != self.output_root or not (candidate / "manifest.json").is_file():
            raise ValueError("找不到该运行")
        return candidate

    def runs(self):
        """按运行名称倒序返回片段标识、运行状态与处理帧数列表。"""
        entries = []
        for path in sorted(self.output_root.glob("run_*/manifest.json"), reverse=True):
            manifest = read_json(path)
            entries.append({"run_id": manifest["run_id"], "status": manifest["status"], "clip_id": manifest["config"]["clip_id"], "processed_frames": manifest.get("processed_frames", 0)})
        return entries

    def data(self, run_id):
        """返回完成运行的检测、轨迹与复盘输入；旧运行的检测缓存为空值。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("该运行未完成，请查看运行清单中的错误")
        tracks_path = run_dir / "tracks.parquet"
        detections_path = run_dir / "detections.parquet"
        cache_key = (run_id, tracks_path.stat().st_mtime_ns, detections_path.stat().st_mtime_ns if detections_path.exists() else None)
        if cache_key not in self.track_cache:
            self.track_cache = {cache_key: {"tracks": list(read_rows(tracks_path)), "detections": list(read_rows(detections_path)) if detections_path.exists() else None}}
        players = load_players(Path(manifest["config"]["players"]))
        bindings = read_yaml(run_dir / "bindings.yaml", {"run_id": run_id, "bindings": []})
        calibration = load_calibration(run_dir)
        segments = field_segments(self.track_cache[cache_key]["tracks"], manifest, calibration)
        review = load_review(run_dir)
        signature = input_signature(run_dir) if review.get("analysis") or any(item.get("execution") == "auto" for item in review["operations"]) else None
        review_history = merge_groups(segments, review.get("operations", []), signature)
        association = association_view(run_dir, segments, review, signature)
        reports = []
        report_paths = (run_dir / "reports").glob("report_*/manifest.json")
        for path in sorted(report_paths, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.parent.name), reverse=True):
            report_manifest = read_json(path)
            if report_manifest["status"] == "complete":
                reports.append(path.parent.name)
        latest = None
        if reports:
            report_dir = run_dir / "reports" / reports[0]
            report_manifest = read_json(report_dir / "manifest.json")
            latest = {
                "report_id": reports[0], "summary": read_json(report_dir / "summary.json"),
                "stale": report_manifest["bindings"] != bindings or report_manifest["calibration"] != calibration or report_manifest["players"] != list(players.values()),
                "motion": list(read_rows(report_dir / "motion.parquet")),
            }
        length = calibration["pitch_length_m"] if calibration else manifest["config"]["pitch_length_m"]
        width = calibration["pitch_width_m"] if calibration else manifest["config"]["pitch_width_m"]
        model = calibration.get("pitch_model", "eleven") if calibration else manifest["config"].get("pitch_model", "eleven")
        spec = calibration.get("pitch_spec") if calibration else None
        definition = pitch_definition(spec or {**pitch_spec(model), "length_m": length, "width_m": width})
        return {"manifest": manifest, **self.track_cache[cache_key], "catalog": read_json(run_dir / "catalog.json"), "players": list(players.values()), "bindings": bindings, "calibration": calibration, "field_segments": segments, "field_review_history": review_history, "association": association, "pitch_lines": definition["lines"], "pitch_definition": definition, "reports": reports, "latest_report": latest, "ball_versions": ball_versions(run_dir)}

    def merge_field_records(self, run_id, payload):
        """把两个当前可见的场内片段合并为同一人工身份并保留审计记录。"""
        data = self.data(run_id)
        source_id = payload.get("source_record_id")
        target_id = payload.get("target_record_id")
        target_frame = payload.get("target_frame")
        known = {segment["record_id"] for segment in data["field_segments"]}
        if not isinstance(source_id, str) or not isinstance(target_id, str) or source_id == target_id:
            raise ValueError("请选择两条不同的场内记录")
        if source_id not in known or target_id not in known:
            raise ValueError("记录已变化或不在有效场地范围内，请刷新列表")
        if isinstance(target_frame, bool) or not isinstance(target_frame, int):
            raise ValueError("请输入有效的融合目标帧")
        by_id = {segment["record_id"]: segment for segment in data["field_segments"]}
        if target_frame not in by_id[target_id].get("field_frames", []):
            raise ValueError("融合目标帧中没有所选目标轨迹，请重新选择")
        if by_id[source_id]["merge_group_id"] == by_id[target_id]["merge_group_id"]:
            raise ValueError("这两条记录已属于同一融合组")
        run_dir = self.run_dir(run_id)
        path = run_dir / "field_review.json"
        review = load_review(run_dir)
        operation = {
            "operation_id": new_id("merge"),
            "source_record_id": source_id,
            "target_record_id": target_id,
            "target_frame": target_frame,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "undone": False,
            "execution": "manual",
        }
        review["operations"].append(operation)
        write_json(path, review)
        return {"saved": True, "operation": operation}

    def undo_field_merge(self, run_id, payload):
        """撤销指定人工融合边；后续操作仍按历史顺序重新解析。"""
        operation_id = payload.get("operation_id")
        if not isinstance(operation_id, str):
            raise ValueError("融合操作编号无效")
        run_dir = self.run_dir(run_id)
        path = run_dir / "field_review.json"
        review = load_review(run_dir)
        operation = next((item for item in review["operations"] if item.get("operation_id") == operation_id), None)
        if operation is None or operation.get("undone"):
            raise ValueError("融合操作不存在或已撤销")
        operation["undone"] = True
        if operation.get("execution") == "auto" and operation.get("node_pair"):
            blocked = pair_key(*operation["node_pair"])
            if blocked not in review["blocked_pairs"]:
                review["blocked_pairs"].append(blocked)
        operation["undone_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(path, review)
        return {"saved": True, "operation_id": operation_id}

    def association_job(self, run_id):
        """刷新恢复任务状态；进程重启后将未完成任务标为可重试失败。"""
        run_dir = self.run_dir(run_id)
        if run_id in self.association_jobs:
            return dict(self.association_jobs[run_id])
        path = run_dir / "association_job.json"
        job = read_json(path) if path.exists() else {"status": "idle"}
        if job.get("status") == "running":
            job.update(status="failed", error="服务已重启，分析中断，请重试")
            write_json(path, job)
        self.association_jobs[run_id] = job
        return dict(job)

    def start_association(self, run_id, payload):
        """复用原始视频与轨迹，在锁外计算，锁内验证并发布自动融合。"""
        run_dir = self.run_dir(run_id)
        if read_json(run_dir / "manifest.json")["status"] != "complete":
            raise ValueError("请等待人物追踪完成")
        with self.lock:
            previous = self.association_job(run_id)
            if previous.get("status") == "running":
                return previous
            if self.calibration_jobs.get(run_id, {}).get("status") == "running":
                raise ValueError("请等待场地标定完成后分析轨迹关联")
            job = {"status": "running", "job_id": new_id("assocjob"), "stage": "queued", "processed_frames": 0, "total_frames": 0}
            self.association_jobs[run_id] = job
            write_json(run_dir / "association_job.json", job)

        def progress(stage, processed, total):
            """按阶段更新可刷新恢复的内存进度。"""
            with self.lock:
                job.update(stage=stage, processed_frames=processed, total_frames=total)

        def worker():
            """提交前重新读取人工历史与输入摘要，保留计算期间的人工编辑。"""
            try:
                result = compute_result(run_dir, progress=progress)
                with self.lock:
                    if self.calibration_jobs.get(run_id, {}).get("status") == "running":
                        raise ValueError("场地标定正在变化，请完成后重新分析")
                    commit_result(run_dir, result, result["input_signature"])
                    job.update(status="complete", stage="complete", batch_id=result["batch_id"])
            except Exception as error:
                with self.lock:
                    job.update(status="failed", error=str(error))
            finally:
                with self.lock:
                    write_json(run_dir / "association_job.json", job)

        threading.Thread(target=worker, daemon=True).start()
        return dict(job)

    def undo_association_batch(self, run_id, payload):
        """逐边撤销指定批次自动操作，同时记录阻止关联；不撤销人工历史。"""
        review = load_review(self.run_dir(run_id))
        batch_id = payload.get("batch_id")
        operations = [item for item in review["operations"] if item.get("execution") == "auto" and item.get("batch_id") == batch_id and not item.get("undone")]
        if not isinstance(batch_id, str) or not operations:
            raise ValueError("找不到可撤销的自动批次")
        for operation in operations:
            self.undo_field_merge(run_id, {"operation_id": operation["operation_id"]})
        return {"saved": True, "undone_count": len(operations)}

    def unblock_association(self, run_id, payload):
        """解除一条撤销关联的自动融合禁用，需重新分析才会执行。"""
        run_dir = self.run_dir(run_id)
        review = load_review(run_dir)
        pair = payload.get("pair")
        if not isinstance(pair, str) or pair not in review["blocked_pairs"]:
            raise ValueError("找不到该阻止关联")
        review["blocked_pairs"].remove(pair)
        write_json(run_dir / "field_review.json", review)
        return {"saved": True}

    def confirm_association(self, run_id, payload):
        """校验当前第一推荐并以手动执行保存，防止确认过期分析。"""
        data = self.data(run_id)
        association = data["association"]
        candidate = association["recommendations"].get(payload.get("record_id"))
        if association["status"] != "complete" or association["batch_id"] != payload.get("batch_id") or not candidate:
            raise ValueError("推荐已变化，请刷新后确认")
        result = self.merge_field_records(run_id, candidate)
        run_dir = self.run_dir(run_id)
        review = load_review(run_dir)
        operation = review["operations"][-1]
        operation.update({key: candidate[key] for key in ["score", "evidence", "node_pair"]})
        operation.update(batch_id=association["batch_id"], algorithm=association["parameters"]["algorithm"])
        write_json(run_dir / "field_review.json", review)
        return result

    def save_bindings(self, run_id, payload):
        """校验整份身份区间后原子保存，返回保存状态；冲突或越界时抛错。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        validate_bindings(payload, manifest, load_players(Path(manifest["config"]["players"])), read_json(run_dir / "catalog.json"))
        write_yaml(run_dir / "bindings.yaml", payload)
        return {"saved": True}

    def save_player(self, run_id, payload):
        """校验球员 ID 和显示名，新增或更新共享档案，返回保存状态。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        path = Path(manifest["config"]["players"])
        players = load_players(path)
        player_id = str(payload.get("player_id", "")).strip()
        display_name = str(payload.get("display_name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", player_id) or not display_name or len(display_name) > 80:
            raise ValueError("球员 ID 使用 1–64 位字母、数字、下划线或短横线，姓名不得为空")
        players[player_id] = {"player_id": player_id, "display_name": display_name}
        write_yaml(path, {"players": list(players.values())})
        return {"saved": True}

    def start_calibration(self, run_id, payload):
        """启动单个后台白线任务，返回进度；重复请求复用进行中的任务。"""
        from .dynamic_calibration import WhiteLineCalibrator
        from .white_line_pipeline import calibrate_run

        run_dir = self.run_dir(run_id)
        if self.analysis_job.get("status") == "running" and self.analysis_job.get("run_id") == run_id:
            raise ValueError("该素材的人物追踪尚未完成，请等待完成")
        manifest = read_json(run_dir / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("运行尚未完成")
        if self.calibration_jobs.get(run_id, {}).get("status") == "running":
            return dict(self.calibration_jobs[run_id])
        if payload.get("pitch_model", "eleven") != "eleven":
            raise ValueError("逐帧白线标定目前仅支持十一人制模型")
        length = float(payload.get("pitch_length_m", manifest["config"]["pitch_length_m"]))
        width = float(payload.get("pitch_width_m", manifest["config"]["pitch_width_m"]))
        WhiteLineCalibrator(length, width)
        job = {"status": "running", "processed_frames": 0, "total_frames": manifest["end_frame"] - manifest["start_frame"]}
        self.calibration_jobs[run_id] = job

        def progress(processed, total):
            """更新共享进度，不持有整个计算阶段的服务锁。"""
            with self.lock:
                job.update(processed_frames=processed, total_frames=total)

        def worker():
            """完成后发布版本并记录状态，异常保留原标定。"""
            try:
                calibration = calibrate_run(run_dir, length, width, progress)
                with self.lock:
                    job.update(status="complete", summary=calibration["summary"], calibration_id=calibration["calibration_id"])
            except Exception as error:
                with self.lock:
                    job.update(status="failed", error=str(error))

        threading.Thread(target=worker, daemon=True).start()
        return dict(job)

    def review_calibration(self, run_id, payload):
        """保存当前动态标定版本的验收状态，拒绝确认已被替换的旧版本。"""
        run_dir = self.run_dir(run_id)
        calibration = read_yaml(run_dir / "calibration.yaml")
        if not calibration or calibration.get("mode") != "white_lines" or payload.get("calibration_id") != calibration.get("calibration_id"):
            raise ValueError("标定版本已变化，请刷新后重新验收")
        if type(payload.get("reviewed")) is not bool:
            raise ValueError("验收状态必须为布尔值")
        calibration["reviewed"] = payload["reviewed"]
        write_yaml(run_dir / "calibration.yaml", calibration)
        return {"saved": True}


def handler_for(app):
    """创建绑定 ReviewApp 的 HTTP 处理器类，限制可访问路径和写入来源。"""
    class ReviewHandler(BaseHTTPRequestHandler):
        """处理静态页面、本地 JSON 接口、源帧与报告文件请求。"""

        def log_message(self, format_string, *args):
            """仅输出非 200/304 状态的访问日志。"""
            if args and str(args[1]) not in ["200", "304"]:
                super().log_message(format_string, *args)

        def send_bytes(self, payload, content_type, status=200):
            """发送字节响应，显式标记类型和长度，并禁用缓存与内容类型嗅探。"""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload, status=200):
            """发送 UTF-8 JSON 响应，拒绝不可序列化数值。"""
            self.send_bytes(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def check_local(self, mutate=False):
            """校验回环 Host；写请求存在 Origin 时要求同源，否则抛出 ValueError。"""
            host = self.headers.get("Host", "")
            if host not in [f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"]:
                raise ValueError("仅允许本地访问")
            origin = self.headers.get("Origin")
            if mutate and origin and origin != f"http://{host}":
                raise ValueError("拒绝跨来源写入")

        def do_GET(self):
            """读取白名单页面、运行数据、源帧或报告；预期错误返回 JSON，断连忽略。"""
            try:
                self.check_local()
                path = unquote(urlsplit(self.path).path)
                if path == "/api/runs":
                    return self.send_json(app.runs())
                if path == "/api/clips":
                    return self.send_json(app.clips())
                if path == "/api/ball-models":
                    return self.send_json({"models": ball_models(), "default_model": DEFAULT_MODEL})
                if path == "/api/ball-job":
                    with app.lock:
                        return self.send_json(dict(app.ball_job))
                metadata_match = re.fullmatch(r"/api/clips/([^/]+)/metadata", path)
                if metadata_match:
                    source = app.clip_path(metadata_match[1])
                    info = inspect_video(source)
                    return self.send_json({"duration_s": info.frame_count / info.fps})
                if path == "/api/pitch-models":
                    query = parse_qs(urlsplit(self.path).query)
                    models = query.get("model", list(PRESETS))
                    return self.send_json({"models": [pitch_definition(pitch_spec(model, query.get("length", [None])[0], query.get("width", [None])[0])) for model in models]})
                if path == "/api/analysis-job":
                    with app.lock:
                        return self.send_json(dict(app.analysis_job))
                clip_match = re.fullmatch(r"/api/clips/([^/]+)/preview", path)
                if clip_match:
                    frame = read_frame(app.clip_path(clip_match[1]), 0)
                    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        raise ValueError("素材预览编码失败")
                    return self.send_bytes(encoded.tobytes(), "image/jpeg")
                match = re.fullmatch(r"/api/runs/([^/]+)(?:/(.*))?", path)
                if match:
                    run_id, action = match.groups()
                    with app.lock:
                        run_dir = app.run_dir(run_id)
                        if action is None:
                            return self.send_json(app.data(run_id))
                        if action == "association-job":
                            return self.send_json(app.association_job(run_id))
                        if action == "association":
                            return self.send_json(app.data(run_id)["association"])
                        if action == "association-result":
                            data = app.data(run_id)
                            return self.send_json({"status": data["association"]["status"], "result": load_review(run_dir).get("analysis")})
                        if action == "calibration-job":
                            return self.send_json(dict(app.calibration_jobs.get(run_id, {"status": "idle"})))
                        if action == "ball":
                            query = parse_qs(urlsplit(self.path).query)
                            model_id = query.get("model", [None])[0]
                            return self.send_json(load_ball_result(run_dir, model_id, query.get("version", [None])[0]))
                        mask_match = re.fullmatch(r"white-lines/(\d+)\.png", action)
                        if mask_match:
                            frame_index = int(mask_match[1])
                            calibration = read_yaml(run_dir / "calibration.yaml")
                            version = parse_qs(urlsplit(self.path).query).get("version", [None])[0]
                            if version is not None:
                                version_dir = calibration_directory(run_dir, {"calibration_id": version})
                                calibration = read_yaml(version_dir / "calibration.yaml")
                            if not calibration or calibration.get("mode") != "white_lines" or not calibration["start_frame"] <= frame_index < calibration["end_frame"]:
                                raise ValueError("该帧没有白线识别缓存")
                            mask_path = calibration_directory(run_dir, calibration) / "masks" / f"{frame_index}.png"
                            return self.send_bytes(mask_path.read_bytes(), "image/png")
                        if action.startswith("frames/"):
                            frame_text = action.removeprefix("frames/")
                            if not frame_text.isdigit():
                                raise ValueError("帧号必须为整数")
                            frame_index = int(frame_text)
                            manifest = read_json(run_dir / "manifest.json")
                            if not manifest["start_frame"] <= frame_index < manifest["end_frame"]:
                                raise ValueError("帧号超出分析区间")
                            frame = read_frame(Path(manifest["config"]["video"]), frame_index)
                            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if not success:
                                raise ValueError("帧编码失败")
                            return self.send_bytes(encoded.tobytes(), "image/jpeg")
                        if action.startswith("crops/"):
                            file_name = action.removeprefix("crops/")
                            if not re.fullmatch(r"\d+\.jpg", file_name):
                                raise ValueError("缩略图名称无效")
                            return self.send_bytes((run_dir / "crops" / file_name).read_bytes(), "image/jpeg")
                report_match = re.fullmatch(r"/reports/([^/]+)/([^/]+)/([^/]+)", path)
                if report_match:
                    run_id, report_id, file_name = report_match.groups()
                    if not re.fullmatch(r"report_[A-Za-z0-9_]+", report_id) or not re.fullmatch(r"(?:report\.html|[a-f0-9]+\.png|summary\.parquet|review\.mp4)", file_name):
                        raise ValueError("报告路径无效")
                    report_path = app.run_dir(run_id) / "reports" / report_id / file_name
                    return self.send_bytes(report_path.read_bytes(), mimetypes.guess_type(file_name)[0] or "application/octet-stream")
                static_paths = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/field-review": "field-review.html", "/field-review.js": "field-review.js", "/field-review.css": "field-review.css"}
                static_paths.update({"/materials": "materials.html", "/materials.css": "materials.css", "/materials.js": "materials.js", "/calibration": "calibration.html", "/calibration.css": "calibration.css", "/calibration.js": "calibration.js"})
                for page in ("identity", "tracking", "report"):
                    static_paths.update({f"/{page}": f"{page}.html", f"/{page}.css": f"{page}.css", f"/{page}.js": f"{page}.js"})
                static_paths.update({"/workspace.js": "workspace.js", "/workspace.css": "workspace.css"})
                if path in static_paths:
                    static_path = WEB / static_paths[path]
                    return self.send_bytes(static_path.read_bytes(), (mimetypes.guess_type(static_path.name)[0] or "text/plain") + "; charset=utf-8")
                self.send_json({"error": "页面不存在"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            """接收最多 1 MiB 的 JSON 对象，串行保存编辑或生成报告，预期错误返回 400。"""
            try:
                self.check_local(mutate=True)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("请求必须使用 application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("请求长度无效")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须为 JSON 对象")
                if urlsplit(self.path).path == "/api/analysis":
                    return self.send_json(app.start_analysis(payload))
                match = re.fullmatch(r"/api/runs/([^/]+)/(ball|calibration|calibration-preview|calibration-auto|calibration-review|bindings|players|field-merges|field-merge-undo|association|association-batch-undo|association-unblock|association-confirm|report)", urlsplit(self.path).path)
                if not match:
                    return self.send_json({"error": "接口不存在"}, 404)
                run_id, action = match.groups()
                with app.lock:
                    analyzing = app.analysis_job.get("status") == "running" and app.analysis_job.get("run_id") == run_id
                    if action in {"calibration", "calibration-review"} and (analyzing or app.calibration_jobs.get(run_id, {}).get("status") == "running"):
                        raise ValueError("逐帧标定仍在计算，请完成后再修改或验收")
                    if action == "association":
                        result = app.start_association(run_id, payload)
                    elif action == "association-batch-undo":
                        result = app.undo_association_batch(run_id, payload)
                    elif action == "association-unblock":
                        result = app.unblock_association(run_id, payload)
                    elif action == "association-confirm":
                        result = app.confirm_association(run_id, payload)
                    elif action == "ball":
                        result = app.start_ball(run_id, payload)
                    elif action == "calibration":
                        result = save_calibration(app.run_dir(run_id), payload)
                    elif action == "calibration-preview":
                        result = build_calibration(read_json(app.run_dir(run_id) / "manifest.json"), {**payload, "reviewed": False})
                    elif action == "calibration-auto":
                        result = app.start_calibration(run_id, payload)
                    elif action == "calibration-review":
                        result = app.review_calibration(run_id, payload)
                    elif action == "bindings":
                        result = app.save_bindings(run_id, payload)
                    elif action == "players":
                        result = app.save_player(run_id, payload)
                    elif action == "field-merges":
                        result = app.merge_field_records(run_id, payload)
                    elif action == "field-merge-undo":
                        result = app.undo_field_merge(run_id, payload)
                    else:
                        from .reporting import create_report

                        result = {"report_id": create_report(app.run_dir(run_id)).name}
                self.send_json(result)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                self.send_json({"error": str(error)}, 400)

    return ReviewHandler


def serve(output_root=ROOT / "outputs", port=8766):
    """在 127.0.0.1 的指定端口阻塞运行复盘服务，收到 Ctrl+C 后关闭。"""
    if not 1 <= port <= 65535:
        raise ValueError("端口无效")
    app = ReviewApp(output_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
    server.daemon_threads = True
    print(f"[review] http://127.0.0.1:{port}  |  Ctrl+C 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
