"""追踪期间以有限观测窗口建立临时身份组，完成后由完整输入重新验证。"""

from collections import defaultdict, deque
import math

from .field_review import merge_groups
from .reassociation import AssociationConfig, analyze_nodes, config_snapshot, group_conflict, rank_candidates


class StreamingAssociator:
    """保留近期真实观测；已关闭的前驱和证据充分的后继才可自动连接。"""

    def __init__(self, manifest, config=None):
        """绑定源视频时间轴，按两秒间隔分析，不改写 ByteTrack 轨迹号。"""
        self.manifest = manifest
        self.config = config or AssociationConfig()
        fps = manifest["video"]["fps"]
        self.close_frames = manifest["config"].get("stride", 1) + math.ceil(fps * manifest["config"].get("lost_track_buffer_s", 3))
        self.keep_frames = math.ceil(fps * (self.config.recommendation_gap_s + self.config.window_s + 4))
        self.tick_frames = max(1, round(fps * 2))
        self.last_tick = manifest["start_frame"] - self.tick_frames
        self.recent = deque()
        self.current = {}
        self.nodes = {}
        self.operations = []

    def observe(self, frame, rows, evidence, calibration=None):
        """积累源观测并周期合组；同 ID 超长缺口产生新的稳定节点。"""
        for row in rows:
            track = row["track_id"]
            node = self.current.get(track)
            if node is None or frame - node["end_frame"] > self.close_frames:
                node = {"record_id": f"track-{track}-frame-{frame}", "track_id": track,
                        "start_frame": frame, "end_frame": frame}
                self.nodes[node["record_id"]] = node
                self.current[track] = node
            node["end_frame"] = frame
            self.recent.append((node["record_id"], row))
        while self.recent and self.recent[0][1]["frame_index"] < frame - self.keep_frames:
            self.recent.popleft()
        if frame - self.last_tick < self.tick_frames:
            return None
        self.last_tick = frame
        grouped = defaultdict(list)
        for node_id, row in self.recent:
            grouped[node_id].append(row)
        nodes = [{**self.nodes[node_id], "rows": observations} for node_id, observations in grouped.items()]
        candidates = analyze_nodes(nodes, self.manifest, calibration, evidence, self.config)
        by_id = {node["record_id"]: node for node in nodes}
        for candidate in candidates:
            source, target = (by_id[candidate[f"{side}_record_id"]] for side in ["source", "target"])
            settled = (source["rows"][0]["frame_index"] == source["start_frame"]
                       and frame - source["start_frame"] >= self.manifest["video"]["fps"] * self.config.window_s
                       and frame - target["end_frame"] > self.close_frames)
            candidate["auto_eligible"] = candidate["auto_eligible"] and settled
        segments = list(self.nodes.values())
        while True:
            merge_groups(segments, self.operations, "stream")
            available = []
            for candidate in candidates:
                source, target = (self.nodes[candidate[f"{side}_record_id"]] for side in ["source", "target"])
                if not group_conflict(segments, source["record_id"], target["record_id"]):
                    candidate["source_competition_id"] = source["merge_group_id"]
                    candidate["target_competition_id"] = target["merge_group_id"]
                    available.append(candidate)
            # 同组端点只保留最高分，不能把本组多个原始端点误作竞争者。
            deduplicated = {}
            for candidate in available:
                key = (candidate["source_competition_id"], candidate["target_competition_id"])
                if key not in deduplicated or candidate["score"] > deduplicated[key]["score"]:
                    deduplicated[key] = candidate
            automatic = next((item for item in rank_candidates(list(deduplicated.values()), self.config) if item["automatic"]), None)
            if automatic is None:
                break
            self.operations.append({**automatic, "execution": "auto", "input_signature": "stream",
                                    "node_pair": [automatic["source_record_id"], automatic["target_record_id"]],
                                    "decided_at_frame": frame})
        return self.snapshot(frame)

    def snapshot(self, frame):
        """导出临时组及逐边依据；完整分析会重新验证，不把临时链作为新证据。"""
        return {"schema_version": 1, "status": "provisional", "frame": frame, "parameters": config_snapshot(self.config),
                "groups": {key: item.get("merge_group_id", key) for key, item in self.nodes.items()},
                "operations": self.operations.copy()}
