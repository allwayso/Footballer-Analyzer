"""以球场运动和共同参照球员重关联原始片段；分数不是身份正确概率。"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibration import load_calibration, project_observation
from .field_review import field_segments, merge_groups
from .storage import file_hash, new_id, read_json, read_rows, write_json
from .teams import shirt_feature
from .video import read_frames

ALGORITHM = "pitch-relative-v1"


@dataclass(frozen=True)
class AssociationConfig:
    """版本化关联参数；距离分别使用米或人物高度归一化单位。"""

    high_threshold: float = 0.88
    low_threshold: float = 0.50
    competition_margin: float = 0.08
    auto_gap_s: float = 3.0
    recommendation_gap_s: float = 10.0
    window_s: float = 1.0
    min_observations: int = 5
    min_span_s: float = 0.16
    position_sigma_m: float = 0.65
    velocity_sigma_mps: float = 4.0
    uncertainty_growth: float = 0.75
    max_speed_mps: float = 12.0
    color_distance_limit: float = 0.65
    reference_count: int = 3
    use_motion: bool = True
    use_color: bool = True
    use_relative: bool = True

    def __post_init__(self):
        """拒绝非法阈值和会产生退化预测的参数。"""
        if not 0 <= self.low_threshold < self.high_threshold < 1:
            raise ValueError("关联阈值必须满足 0 ≤ 最低阈值 < 最高阈值 < 1")
        for name in ["competition_margin", "auto_gap_s", "recommendation_gap_s", "window_s", "min_span_s", "position_sigma_m", "velocity_sigma_mps", "uncertainty_growth", "max_speed_mps", "color_distance_limit"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"关联参数 {name} 必须为有限正数")
        if self.auto_gap_s > self.recommendation_gap_s:
            raise ValueError("自动关联时间不能超过推荐时间")
        for name, minimum in [("min_observations", 3), ("reference_count", 3)]:
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} 至少为 {minimum}")
        if any(type(getattr(self, name)) is not bool for name in ["use_motion", "use_color", "use_relative"]):
            raise ValueError("特征开关必须为布尔值")


def config_snapshot(config=None):
    """返回随结果保存的完整算法参数。"""
    return {"algorithm": ALGORITHM, **asdict(config or AssociationConfig())}


def digest(value):
    """为可序列化输入生成确定摘要。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def input_signature(run_dir, config=None):
    """绑定源文件、原始轨迹、标定和参数；不把可变操作历史混入证据摘要。"""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    video = Path(manifest["config"]["video"])
    stat = video.stat()
    return digest({"manifest": {key: manifest.get(key) for key in ["video_sha256", "video", "config", "start_frame", "end_frame"]},
                   "video": [str(video.resolve()), stat.st_size, stat.st_mtime_ns],
                   "tracks": file_hash(run_dir / "tracks.parquet"), "calibration": load_calibration(run_dir),
                   "parameters": config_snapshot(config)})


def raw_nodes(rows, manifest):
    """按原始轨迹与长缺口建立稳定节点，不依赖场内筛选。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    nodes = []
    gap = manifest["config"].get("stride", 1) + math.ceil(manifest["video"]["fps"] * manifest["config"].get("lost_track_buffer_s", 3))
    for track_id, observations in grouped.items():
        observations.sort(key=lambda item: item["frame_index"])
        chunks = [[]]
        for row in observations:
            if chunks[-1] and row["frame_index"] - chunks[-1][-1]["frame_index"] > gap:
                chunks.append([])
            chunks[-1].append(row)
        for chunk in chunks:
            nodes.append({"record_id": f"track-{track_id}-frame-{chunk[0]['frame_index']}", "track_id": track_id,
                          "start_frame": chunk[0]["frame_index"], "end_frame": chunk[-1]["frame_index"],
                          "field_frames": [row["frame_index"] for row in chunk], "rows": chunk})
    return sorted(nodes, key=lambda item: (item["start_frame"], item["track_id"]))


class EvidenceCollector:
    """在追踪时采样多帧颜色；旧运行回放使用相同的采样规则。"""

    def __init__(self, fps):
        """每秒最多采样五帧，保留源帧索引与颜色值。"""
        self.interval = max(1, round(fps / 5))
        self.samples = {}

    def observe(self, image, rows):
        """缓存当前真实观测的上衣特征，不保存预测框。"""
        for row in rows:
            if row["frame_index"] % self.interval:
                continue
            box = [row[key] for key in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
            feature = shirt_feature(image, box)
            if feature is not None:
                self.samples[f"{row['track_id']}:{row['frame_index']}"] = feature.tolist()


def evidence_signature(run_dir, manifest):
    """为在线采样与离线采样生成同一缓存摘要。"""
    stat = Path(manifest["config"]["video"]).stat()
    return digest([manifest.get("video_sha256"), stat.st_size, stat.st_mtime_ns, file_hash(Path(run_dir) / "tracks.parquet")])


def collect_evidence(run_dir, rows, manifest, progress=None):
    """复用兼容缓存，否则顺序读取原视频采样，不运行人物检测。"""
    path = Path(run_dir) / "association_evidence.json"
    signature = evidence_signature(run_dir, manifest)
    if path.exists():
        cached = read_json(path)
        if cached.get("signature") == signature:
            return cached["samples"]
    collector = EvidenceCollector(manifest["video"]["fps"])
    indexed = defaultdict(list)
    for row in rows:
        indexed[row["frame_index"]].append(row)
    total = manifest["end_frame"] - manifest["start_frame"]
    for packet in read_frames(Path(manifest["config"]["video"]), manifest["start_frame"], manifest["end_frame"], 1):
        collector.observe(packet.image, indexed.get(packet.frame_index, []))
        if progress and packet.frame_index % 25 == 0:
            progress("evidence", packet.frame_index - manifest["start_frame"] + 1, total)
    write_json(path, {"schema_version": 1, "signature": signature, "samples": collector.samples})
    return collector.samples


def endpoint(node, last, window):
    """截取首末一秒的真实观测。"""
    rows = node["rows"]
    boundary = rows[-1 if last else 0]["timestamp_s"]
    return [row for row in rows if abs(row["timestamp_s"] - boundary) <= window]


def fit_motion(rows, points, boundary):
    """用成对斜率中位数拟合运动，返回端点、速度与残差。"""
    times = np.asarray([row["timestamp_s"] for row in rows]) - boundary
    points = np.asarray(points, float)
    indices = np.linspace(0, len(times) - 1, min(12, len(times))).astype(int)
    slopes = [(points[j] - points[i]) / (times[j] - times[i]) for i in indices for j in indices if j > i and times[j] > times[i]]
    velocity = np.median(slopes, axis=0) if slopes else np.zeros(2)
    origin = np.median(points - times[:, None] * velocity, axis=0)
    residual = float(np.median(np.linalg.norm(points - origin - times[:, None] * velocity, axis=1)))
    return origin, velocity, residual


def color_feature(rows, evidence):
    """聚合端点窗口中的多帧特征，样本不足时不提供颜色证据。"""
    features = [evidence[f"{row['track_id']}:{row['frame_index']}"] for row in rows if f"{row['track_id']}:{row['frame_index']}" in evidence]
    return np.median(features, axis=0) if len(features) >= 2 else None


def relative_points(first, second, nodes, config, fps, stride):
    """使用共同持续可见的参照轨迹消除整体位移，并按参照人物高度归一化。"""
    needed = first + second
    references = []
    first_time, last_time = needed[0]["frame_index"], needed[-1]["frame_index"]
    excluded = {first[0]["track_id"], second[0]["track_id"]}
    for node in nodes:
        if node["track_id"] in excluded or node["start_frame"] > first_time or node["end_frame"] < last_time:
            continue
        indexed = node.get("_indexed") or {row["frame_index"]: row for row in node["rows"]}
        matched = []
        for row in needed:
            match = next((indexed.get(row["frame_index"] + delta) for delta in [0, -stride, stride] if indexed.get(row["frame_index"] + delta)), None)
            if match is None:
                break
            matched.append(match)
        if len(matched) == len(needed):
            positions = np.asarray([[(row["bbox_x1"] + row["bbox_x2"]) / 2, row["bbox_y2"]] for row in matched])
            heights = np.asarray([max(1, row["bbox_y2"] - row["bbox_y1"]) for row in matched])
            subject = np.asarray([[(row["bbox_x1"] + row["bbox_x2"]) / 2, row["bbox_y2"]] for row in needed])
            distance = float(np.median(np.linalg.norm(positions - subject, axis=1) / heights))
            references.append((positions, heights, distance))
    if len(references) < config.reference_count:
        return None
    references.sort(key=lambda item: item[2])
    references = references[:8]
    # 共同参照之间的相对布局需要稳定，避免一组相反运动的人被误当作镜头运动。
    positions = np.asarray([item[0] for item in references])
    heights = np.median([item[1] for item in references], axis=0)
    center = np.median(positions, axis=0)
    offsets = (positions - center) / heights[None, :, None]
    drift = np.max(np.linalg.norm(offsets - offsets[:, :1], axis=2), axis=1)
    stable = drift < 0.6
    if stable.sum() < config.reference_count:
        return None
    positions = positions[stable]
    center = np.median(positions, axis=0)
    points = (np.asarray([[(row["bbox_x1"] + row["bbox_x2"]) / 2, row["bbox_y2"]] for row in needed]) - center) / heights[:, None]
    return points[:len(first)], points[len(first):], int(stable.sum())


def pitch_continuity(earlier, later, nodes, calibration, manifest, config):
    """动态标定须覆盖缺口，且共同可见球员的球场位置不能跳变。"""
    if not calibration or calibration.get("mode") != "white_lines":
        return True
    start, end = earlier["end_frame"], later["start_frame"]
    stride = manifest["config"].get("stride", 1)
    frames = calibration.get("frames", {})
    for frame in range(start, end + 1, stride):
        record = frames.get(str(frame))
        if not record or record.get("status") != "valid" or record.get("homography") is None:
            return False
    checked = 0
    for node in nodes:
        if node["track_id"] in {earlier["track_id"], later["track_id"]} or node["start_frame"] > start or node["end_frame"] < end:
            continue
        observations = [row for row in node["rows"] if start <= row["frame_index"] <= end]
        if len(observations) < 2 or observations[0]["frame_index"] > start + stride or observations[-1]["frame_index"] < end - stride:
            continue
        projected = [project_observation(row, calibration) for row in observations]
        if any(reason is not None for _, _, reason in projected):
            continue
        dt = np.diff([row["timestamp_s"] for row in observations])
        distances = np.linalg.norm(np.diff([[x, y] for x, y, _ in projected], axis=0), axis=1)
        if np.any(distances > config.max_speed_mps * dt + 0.8):
            return False
        checked += 1
    return checked >= config.reference_count


def score_candidate(earlier, later, nodes, calibration, evidence, manifest, config):
    """双向验证断点运动；无法得到可靠坐标时不构造相似度。"""
    gap = later["rows"][0]["timestamp_s"] - earlier["rows"][-1]["timestamp_s"]
    if gap <= 0 or gap > config.recommendation_gap_s:
        return None
    first = endpoint(earlier, True, config.window_s)
    second = endpoint(later, False, config.window_s)
    mode, references = "pitch", 0
    first_points, second_points = [], []
    if calibration and calibration.get("reviewed") is True:
        for rows, points in [(first, first_points), (second, second_points)]:
            for row in rows:
                x, y, reason = project_observation(row, calibration)
                if reason is not None:
                    break
                points.append([x, y])
    pitch_complete = len(first_points) == len(first) and len(second_points) == len(second)
    if not pitch_complete:
        mode = "relative"
        relative = relative_points(first, second, nodes, config, manifest["video"]["fps"], manifest["config"].get("stride", 1)) if config.use_relative else None
        if relative is None:
            return None
        first_points, second_points, references = relative
    t1, t2 = first[-1]["timestamp_s"], second[0]["timestamp_s"]
    p1, v1, r1 = fit_motion(first, first_points, t1)
    p2, v2, r2 = fit_motion(second, second_points, t2)
    unit = 1.0 if mode == "pitch" else 0.18
    # 投影发生突跳、片段内剧烈串号或参照失效时不允许自动关联。
    span1, span2 = t1 - first[0]["timestamp_s"], second[-1]["timestamp_s"] - t2
    enough = min(len(first), len(second)) >= config.min_observations and min(span1, span2) >= config.min_span_s
    max_speed = config.max_speed_mps * unit
    plausible = max(np.linalg.norm(v1), np.linalg.norm(v2)) <= max_speed and max(r1, r2) < 0.75 * unit
    for rows, points in [(first, first_points), (second, second_points)]:
        dt = np.diff([row["timestamp_s"] for row in rows])
        jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if len(dt) and np.any(jumps > max_speed * dt + 0.8 * unit):
            plausible = False
    speed_delta = float(np.linalg.norm(v1 - v2))
    if config.use_motion:
        error = max(float(np.linalg.norm(p1 + v1 * gap - p2)), float(np.linalg.norm(p2 - v2 * gap - p1)))
    else:
        error = float(np.linalg.norm(p1 - p2))
    sigma = (config.position_sigma_m + config.uncertainty_growth * min(gap, 3)) * unit + r1 + r2
    displacement = float(np.linalg.norm(p2 - p1))
    if displacement > max_speed * gap + 2 * unit or error > 4 * sigma:
        return None
    position_score = math.exp(-0.5 * (error / sigma) ** 2)
    velocity_score = math.exp(-0.5 * (speed_delta / (config.velocity_sigma_mps * unit)) ** 2) if config.use_motion else position_score
    c1, c2 = color_feature(first, evidence), color_feature(second, evidence)
    color_distance = float(np.linalg.norm(c1 - c2)) if c1 is not None and c2 is not None else None
    color_score = math.exp(-2 * color_distance ** 2) if color_distance is not None and config.use_color else None
    color_conflict = config.use_color and color_distance is not None and color_distance > config.color_distance_limit
    score = (0.68 * position_score + 0.27 * velocity_score + 0.05 * math.exp(-gap / 5))
    if color_score is not None:
        score *= 0.8 + 0.2 * color_score
    continuous = pitch_continuity(earlier, later, nodes, calibration, manifest, config) if mode == "pitch" else True
    auto_eligible = continuous and enough and plausible and not color_conflict and gap <= config.auto_gap_s and error <= 2 * sigma
    # 共同参照的自身运动无法完全排除，缺球场标定时采用更严格的短缺口条件。
    if mode == "relative":
        auto_eligible = auto_eligible and gap <= 0.6 and error <= sigma and color_score is not None
    reasons = []
    if not continuous:
        reasons.append("断点区间缺少连续可靠的场地参照")
    if not enough:
        reasons.append("端点观测不足")
    if not plausible:
        reasons.append("运动或投影不稳定")
    if color_conflict:
        reasons.append("颜色明显不符")
    if gap > config.auto_gap_s:
        reasons.append("缺口超过自动关联范围")
    if mode == "relative" and (gap > 0.6 or error > sigma or color_score is None):
        reasons.append("邻近参照模式的短缺口、位置或多帧颜色证据不足")
    return {"source_record_id": later["record_id"], "target_record_id": earlier["record_id"],
            "source_frame": later["start_frame"], "target_frame": earlier["end_frame"], "score": round(score, 6),
            "auto_eligible": bool(auto_eligible), "evidence": {"coordinate_mode": mode, "reference_count": references,
                "gap_s": round(gap, 4), "position_error": round(error, 4), "position_sigma": round(sigma, 4),
                "velocity_delta": round(speed_delta, 4), "color_distance": color_distance,
                "position_score": position_score, "velocity_score": velocity_score, "reasons": reasons}}


def rank_candidates(candidates, config):
    """互为最佳且领先次优的候选才可自动执行，一对一分配允许不匹配。"""
    def source_key(candidate):
        """当前组是候选竞争单位，未分组时使用原始节点。"""
        return candidate.get("source_competition_id", candidate["source_record_id"])

    def target_key(candidate):
        """读取目标组的竞争标识。"""
        return candidate.get("target_competition_id", candidate["target_record_id"])

    by_source, by_target = defaultdict(list), defaultdict(list)
    for candidate in candidates:
        by_source[source_key(candidate)].append(candidate)
        by_target[target_key(candidate)].append(candidate)
    for values in [*by_source.values(), *by_target.values()]:
        values.sort(key=lambda item: (-item["score"], item["source_record_id"], item["target_record_id"]))
    eligible = []
    for candidate in candidates:
        source = by_source[source_key(candidate)]
        target = by_target[target_key(candidate)]
        margin = min(candidate["score"] - (values[1]["score"] if len(values) > 1 else config.low_threshold) for values in [source, target])
        candidate["competition_margin"] = round(margin, 6)
        if candidate["auto_eligible"] and candidate["score"] > config.high_threshold and source[0] is candidate and target[0] is candidate and margin >= config.competition_margin:
            eligible.append(candidate)
    sources, targets = sorted(by_source), sorted(by_target)
    matrix = np.zeros((len(sources), len(targets) + len(sources)))
    source_index = {key: i for i, key in enumerate(sources)}
    target_index = {key: i for i, key in enumerate(targets)}
    for candidate in eligible:
        matrix[source_index[source_key(candidate)], target_index[target_key(candidate)]] = candidate["score"]
    assigned = set()
    if sources:
        ri, ci = linear_sum_assignment(matrix, maximize=True)
        assigned = {(sources[i], targets[j]) for i, j in zip(ri, ci) if j < len(targets) and matrix[i, j] > 0}
    for candidate in candidates:
        candidate["automatic"] = (source_key(candidate), target_key(candidate)) in assigned
    return sorted(candidates, key=lambda item: (-item["score"], item["source_record_id"], item["target_record_id"]))


def analyze_nodes(nodes, manifest, calibration=None, evidence=None, config=None, progress=None):
    """只依赖真实观测生成候选，不使用融合后的长链反向增强分数。"""
    config = config or AssociationConfig()
    candidates = []
    for node in nodes:
        node["_indexed"] = {row["frame_index"]: row for row in node["rows"]}
    for index, later in enumerate(nodes):
        for earlier in nodes:
            if earlier["end_frame"] >= later["start_frame"]:
                continue
            candidate = score_candidate(earlier, later, nodes, calibration, evidence or {}, manifest, config)
            if candidate and candidate["score"] > config.low_threshold:
                candidates.append(candidate)
        if progress:
            progress("matching", index + 1, len(nodes))
    for node in nodes:
        node.pop("_indexed", None)
    return rank_candidates(candidates, config)


def load_review(run_dir):
    """兼容读取旧人工历史并补齐阻止列表。"""
    path = Path(run_dir) / "field_review.json"
    review = read_json(path) if path.exists() else {"schema_version": 2, "operations": []}
    review.setdefault("blocked_pairs", [])
    return review


def pair_key(source, target):
    """用无向边标识阻止重复自动融合。"""
    return "|".join(sorted([source, target]))


def group_conflict(segments, source, target):
    """检查整个融合组的时间重叠，防止链式合组混入同时出现的人物。"""
    by_id = {item["record_id"]: item for item in segments}
    if source not in by_id or target not in by_id:
        return True
    roots = [by_id[key].get("merge_group_id", key) for key in [source, target]]
    if roots[0] == roots[1]:
        return True
    groups = [[item for item in segments if item.get("merge_group_id", item["record_id"]) == root] for root in roots]
    return any(not (a.get("identity_end_frame", a["end_frame"]) < b.get("identity_start_frame", b["start_frame"])
                    or b.get("identity_end_frame", b["end_frame"]) < a.get("identity_start_frame", a["start_frame"])) for a in groups[0] for b in groups[1])


def group_blocked(segments, source, target, blocked_pairs):
    """阻止被撤销的两个原始节点经第三片段间接重新归为同组。"""
    by_id = {item["record_id"]: item for item in segments}
    roots = [by_id[key].get("merge_group_id", key) for key in [source, target]]
    members = [{item.get("identity_node_id", item["record_id"]) for item in segments
                if item.get("merge_group_id", item["record_id"]) == root} for root in roots]
    return any(pair_key(left, right) in blocked_pairs for left in members[0] for right in members[1])


def visible_candidates(result, segments, review):
    """将稳定节点的端点唯一映射到当前场内记录，排除同组和已阻止关联。"""
    mapped = {}
    by_id = {item["record_id"]: item for item in segments}
    by_node = {item["record_id"]: item for item in result["nodes"]}
    node_mapping = {}
    # 先绑定全部组成员的原始时间范围，再进行冲突检查，避免候选遍历顺序影响判断。
    for node_id, node in by_node.items():
        matches = [item for item in segments if item["track_id"] == node["track_id"]
                   and any(node["start_frame"] <= frame <= node["end_frame"] for frame in item["field_frames"])]
        if len(matches) == 1:
            item = matches[0]
            item["identity_start_frame"] = node["start_frame"]
            item["identity_end_frame"] = node["end_frame"]
            item["identity_node_id"] = node_id
            node_mapping[node_id] = item
    for original in result.get("candidates", []):
        candidate = dict(original)
        for side in ["source", "target"]:
            mapped_segment = node_mapping.get(original[f"{side}_record_id"])
            if not mapped_segment or original[f"{side}_frame"] not in mapped_segment["field_frames"]:
                break
            candidate[f"{side}_record_id"] = mapped_segment["record_id"]
        else:
            source, target = candidate["source_record_id"], candidate["target_record_id"]
            if not group_blocked(segments, source, target, review.get("blocked_pairs", [])) and not group_conflict(segments, source, target):
                candidate["node_pair"] = [original["source_record_id"], original["target_record_id"]]
                group_pair = tuple(by_id[key].get("merge_group_id", key) for key in [source, target])
                candidate["source_competition_id"], candidate["target_competition_id"] = group_pair
                if group_pair not in mapped or mapped[group_pair]["score"] < candidate["score"]:
                    mapped[group_pair] = candidate
    return list(mapped.values())


def commit_result(run_dir, result, signature, config=None):
    """在调用方持有编辑锁时校验输入并一次保存自动历史；失败不覆盖上一结果。"""
    run_dir = Path(run_dir)
    if signature != input_signature(run_dir, config):
        raise ValueError("分析期间视频、轨迹、标定或参数已变化，请重新分析")
    rows = list(read_rows(run_dir / "tracks.parquet"))
    manifest = read_json(run_dir / "manifest.json")
    calibration = load_calibration(run_dir)
    segments = field_segments(rows, manifest, calibration) if calibration and calibration.get("reviewed") else [{key: value for key, value in node.items() if key != "rows"} for node in raw_nodes(rows, manifest)]
    review = load_review(run_dir)
    live_path = run_dir / "association_live.json"
    live = read_json(live_path) if live_path.exists() else {}
    online_pairs = {pair_key(*item["node_pair"]): item for item in live.get("operations", []) if item.get("node_pair")}
    # 新版分析替代旧自动结论；撤销记录及手动结论保留。
    for operation in review["operations"]:
        if operation.get("execution", "manual") == "auto" and not operation.get("undone") and operation.get("input_signature") != signature:
            operation["superseded"] = True
    merge_groups(segments, review["operations"], signature)
    added = 0
    while True:
        candidates = visible_candidates(result, segments, review)
        effective = rank_candidates(candidates, config or AssociationConfig())
        round_added = 0
        for candidate in effective:
            if not candidate["automatic"] or group_conflict(segments, candidate["source_record_id"], candidate["target_record_id"]) or group_blocked(segments, candidate["source_record_id"], candidate["target_record_id"], review.get("blocked_pairs", [])):
                continue
            operation = {key: candidate[key] for key in ["source_record_id", "target_record_id", "target_frame", "score", "evidence", "node_pair", "competition_margin"]}
            operation.update(operation_id=new_id("merge"), execution="auto", batch_id=result["batch_id"], created_at=result["created_at"],
                             algorithm=ALGORITHM, thresholds=result["parameters"], input_signature=signature, undone=False)
            online = online_pairs.get(pair_key(*candidate["node_pair"]))
            operation["execution_phase"] = "tracking" if online else "post_tracking"
            if online:
                operation["decided_at_frame"] = online["decided_at_frame"]
            review["operations"].append(operation)
            merge_groups(segments, review["operations"], signature)
            round_added += 1
            # 每次合组后重新计算组级竞争，不能沿用合组前的一批竞争结论。
            break
        added += round_added
        if not round_added:
            break
    result.update(input_signature=signature, automatic_added=added)
    # 分析快照与全部历史作为一份 JSON 原子替换，避免半发布。
    review["analysis"] = result
    review["schema_version"] = 2
    write_json(run_dir / "field_review.json", review)
    return result


def compute_result(run_dir, config=None, progress=None):
    """计算独立关联版本，提交前由调用方再次检查输入与编辑状态。"""
    config = config or AssociationConfig()
    run_dir = Path(run_dir)
    started = time.perf_counter()
    signature = input_signature(run_dir, config)
    manifest = read_json(run_dir / "manifest.json")
    if file_hash(Path(manifest["config"]["video"])) != manifest["video_sha256"]:
        raise ValueError("源视频摘要与追踪运行不符，请恢复原视频")
    rows = list(read_rows(run_dir / "tracks.parquet"))
    evidence = collect_evidence(run_dir, rows, manifest, progress)
    nodes = raw_nodes(rows, manifest)
    candidates = analyze_nodes(nodes, manifest, load_calibration(run_dir), evidence, config, progress)
    from datetime import datetime, timezone

    result = {"schema_version": 1, "batch_id": new_id("association"), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "parameters": config_snapshot(config), "input_signature": signature,
              "nodes": [{key: value for key, value in item.items() if key != "rows"} for item in nodes], "candidates": candidates,
              "elapsed_s": round(time.perf_counter() - started, 3)}
    return result


def association_view(run_dir, segments, review, signature):
    """返回最新分析、按当前组过滤的第一推荐及准确的组数统计。"""
    result = review.get("analysis")
    stale = bool(result and result.get("input_signature") != signature)
    recommendations = {}
    if result and not stale:
        for candidate in visible_candidates(result, segments, review):
            # 组级一条最高推荐；仍保存原始端点，确认时不重新估分。
            key = next(item["merge_group_id"] for item in segments if item["record_id"] == candidate["source_record_id"])
            if key not in recommendations or recommendations[key]["score"] < candidate["score"]:
                recommendations[key] = candidate
    group_recommendations = recommendations
    recommendations = {item["record_id"]: group_recommendations[item["merge_group_id"]] for item in segments if item["merge_group_id"] in group_recommendations}
    manual = [dict(item) for item in segments]
    merge_groups(manual, [operation for operation in review["operations"] if operation.get("execution", "manual") != "auto"], signature)
    groups = len({item["merge_group_id"] for item in segments})
    manual_groups = len({item["merge_group_id"] for item in manual})
    return {"status": "stale" if stale else "complete" if result else "idle", "batch_id": result["batch_id"] if result else None,
            "elapsed_s": result.get("elapsed_s") if result else None, "parameters": result.get("parameters") if result else config_snapshot(),
            "recommendations": recommendations, "blocked_pairs": review.get("blocked_pairs", []),
            "summary": {"original_records": len(segments), "groups": groups, "automatic_reduction": manual_groups - groups}}
