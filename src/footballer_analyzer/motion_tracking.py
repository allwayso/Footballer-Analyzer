"""在身份分配前融合运动和上衣颜色，延迟确认真实观测并保留直接检测映射。"""

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import math

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter

from .calibration import project_observation, calibration_for_frame
from .teams import shirt_histogram


@dataclass(frozen=True)
class MotionConfig:
    """运动颜色跟踪参数；代价越大越宽松，时间单位为秒。"""

    max_cost: float = 0.99
    position_tolerance: float = 3.0
    competition_margin: float = 0.0
    delay_s: float = 0.2
    history_s: float = 0.6
    color_window_s: float = 1.0
    color_samples: int = 10
    new_threshold: float = 0.75
    detection_floor: float = 0.1
    max_gap_s: float = 3.0
    image_gap_s: float = 1.0
    size_ratio: float = 1.8
    use_motion: bool = True
    use_color: bool = True
    use_relative: bool = True
    protect_boxes: bool = True

    def __post_init__(self):
        """拒绝非有限阈值、错误类型及超出设计范围的等待时间。"""
        for name in ("max_cost", "position_tolerance", "delay_s", "history_s", "color_window_s", "new_threshold", "detection_floor", "max_gap_s", "image_gap_s", "size_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if isinstance(self.competition_margin, bool) or not isinstance(self.competition_margin, (int, float)) or not math.isfinite(self.competition_margin) or not 0 <= self.competition_margin < 1:
            raise ValueError("竞争分差必须在 [0, 1) 内")
        if not self.detection_floor < self.new_threshold < 1 or not self.max_cost < 1:
            raise ValueError("检测、新建阈值或匹配代价无效")
        if self.delay_s > 0.2 or self.image_gap_s > self.max_gap_s or self.max_gap_s > 3:
            raise ValueError("确认延迟最多0.2秒，画面缺口不得超过总缺口，总缺口最多3秒")
        if type(self.color_samples) is not int or self.color_samples < 1:
            raise ValueError("颜色样本数必须为正整数")
        for name in ("use_motion", "use_color", "use_relative", "protect_boxes"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须为布尔值")

    def snapshot(self):
        """返回包含权重和算法版本的可复现配置。"""
        return {"algorithm": "motion-color-v1", **asdict(self),
                "weights": {"position": 0.5, "motion": 0.2, "iou": 0.15, "color": 0.15}}


def feet(boxes):
    """返回一批 xyxy 框的底边中点。"""
    return np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))


def box_iou(first, second):
    """向量化计算两组框的交并比，空矩阵保持二维形状。"""
    size = np.maximum(0, np.minimum(first[:, None, 2:], second[None, :, 2:]) - np.maximum(first[:, None, :2], second[None, :, :2]))
    intersection = size.prod(axis=2)
    area1 = np.maximum(0, first[:, 2:] - first[:, :2]).prod(axis=1)
    area2 = np.maximum(0, second[:, 2:] - second[:, :2]).prod(axis=1)
    return intersection / np.maximum(1e-9, area1[:, None] + area2[None, :] - intersection)


def fit_velocity(times, points):
    """从最多七个时间点的成对斜率中位数估计速度和残差。"""
    indices = np.unique(np.linspace(0, len(times) - 1, min(7, len(times))).astype(int))
    times, points = np.asarray(times)[indices], np.asarray(points)[indices]
    left, right = np.triu_indices(len(times), 1)
    delta = times[right] - times[left]
    valid = delta > 1e-9
    velocity = np.median((points[right[valid]] - points[left[valid]]) / delta[valid, None], axis=0) if valid.any() else np.zeros(2)
    origin = np.median(points - (times - times[-1])[:, None] * velocity, axis=0)
    residual = float(np.median(np.linalg.norm(points - origin - (times - times[-1])[:, None] * velocity, axis=1)))
    return origin, velocity, residual


def measurement(box):
    """把 xyxy 观测转换为 Kalman 的中心、宽高比和高度。"""
    width, height = np.maximum(1, box[2:] - box[:2])
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, width / height, height])


@dataclass
class Observation:
    """单条真实检测及可变身份引用，不把内部预测伪装为观测。"""

    frame: int
    time: float
    index: int
    box: np.ndarray
    score: float
    color: np.ndarray | None
    pitch: np.ndarray | None
    references: dict = field(default_factory=dict)


@dataclass
class Tracklet:
    """维护可信运动、颜色样本、待确认观测与内部Kalman状态。"""

    token: int
    track_id: int | None
    observations: list
    history: deque
    colors: deque
    mean: np.ndarray
    covariance: np.ndarray
    maximum_score: float
    last_time: float
    created_time: float


@dataclass
class TrackedFrame:
    """按源时间提交给下游的已确认帧和直接关联检测。"""

    packet: object
    detections: sv.Detections


class MotionColorTracker:
    """运动颜色联合跟踪，最多缓存0.2秒并为已确认帧按顺序输出。"""

    def __init__(self, source_fps, stride, config=None, calibration=None, event_sink=None):
        """接收采样信息、可选已确认标定和逐事件诊断回调。"""
        if not math.isfinite(source_fps) or source_fps <= 0 or type(stride) is not int or stride <= 0:
            raise ValueError("源帧率和采样间隔必须为正")
        self.config = config or MotionConfig()
        if stride / source_fps > self.config.delay_s + 1e-9:
            raise ValueError("当前采样间隔超过0.2秒，无法在确认期限内获得两次真实观测；请减小stride或选择bytetrack-legacy")
        self.fps, self.stride = source_fps, stride
        self.calibration = calibration if calibration and calibration.get("reviewed") is True else None
        self.event_sink = event_sink
        self.kalman = KalmanFilter()
        self.tracks = []
        self.buffer = deque()
        self.next_token = self.next_id = 1
        self.previous_time = -math.inf
        self.previous_frame = -1
        self.stats = Counter()
        self.closed = False

    def event(self, frame, reason, **details):
        """累计行为计数并按需写入完整可复查事件。"""
        self.stats[reason] += 1
        if self.event_sink:
            self.event_sink({"frame_index": int(frame), "reason": reason, **details})

    def project(self, box, frame):
        """仅使用当前帧已确认标定，缺失映射时不沿用旧矩阵。"""
        if self.calibration is None:
            return None
        row = {"frame_index": frame, **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box))}
        x, y, reason = project_observation(row, self.calibration)
        return np.array([x, y]) if reason is None else None

    def continuous_pitch(self, start, end):
        """动态标定跨缺口必须逐采样帧有效，不只检查两个端点。"""
        return self.calibration is not None and all(calibration_for_frame(self.calibration, frame) is not None for frame in range(start, end + 1, self.stride))

    def references(self, boxes, colors, time):
        """用成熟轨迹的互为最佳紧邻框建立当前邻近球员参照。"""
        mature = [track for track in self.tracks if track.track_id is not None and len(track.history) >= 3 and time - track.last_time <= max(0.12, self.stride / self.fps * 1.5)]
        if not mature or not len(boxes):
            return {}
        overlap = box_iou(np.array([track.history[-1].box for track in mature]), boxes)
        result = {}
        for index, track in enumerate(mature):
            candidate = int(np.argmax(overlap[index]))
            if overlap[index, candidate] < 0.4 or int(np.argmax(overlap[:, candidate])) != index:
                continue
            old_size = np.maximum(1, track.history[-1].box[2:] - track.history[-1].box[:2])
            size = np.maximum(1, boxes[candidate, 2:] - boxes[candidate, :2])
            if np.max(np.maximum(size / old_size, old_size / size)) > 1.5:
                continue
            color_distance = self.color_distance(track, colors[candidate], time)
            if color_distance is not None and color_distance > 0.65:
                continue
            result[track.token] = (feet(boxes[candidate:candidate + 1])[0], float(size[1]))
        return result

    def color_distance(self, track, color, time):
        """将当前直方图与最近一秒可信样本集合比较，使用Hellinger距离。"""
        samples = [sample for stamp, sample in track.colors if track.last_time - stamp <= self.config.color_window_s]
        if color is None or not samples:
            return None
        distances = np.sqrt(np.maximum(0, 1 - np.sqrt(np.array(samples) * color).sum(axis=1)))
        return float(np.median(np.sort(distances)[:max(1, math.ceil(len(distances) / 2))]))

    def color_distances(self, track, colors):
        """一次比较所有候选颜色，避免逐检测重复堆叠模板与中位数运算。"""
        result = [None] * len(colors)
        samples = [sample for stamp, sample in track.colors if track.last_time - stamp <= self.config.color_window_s]
        indices = [i for i, color in enumerate(colors) if color is not None]
        if not self.config.use_color or not samples or not indices:
            return result
        templates = np.sqrt(np.asarray(samples))
        candidates = np.sqrt(np.asarray([colors[i] for i in indices]))
        distances = np.sqrt(np.maximum(0, 1 - templates @ candidates.T))
        values = np.median(np.sort(distances, axis=0)[:max(1, math.ceil(len(samples) / 2))], axis=0)
        for index, value in zip(indices, values):
            result[index] = float(value)
        return result

    def model(self, track, references, time, allow_pitch=True):
        """优先拟合球场或共同邻近球员相对运动，其次使用尺度归一化画面。"""
        history = list(track.history)
        times = np.array([item.time for item in history])
        points = np.array([feet(item.box[None])[0] for item in history])
        height = float(np.median([max(1, item.box[3] - item.box[1]) for item in history]))
        mode, current_center, current_scale = "image", np.zeros(2), height
        if allow_pitch and all(item.pitch is not None for item in history) and self.continuous_pitch(history[0].frame, self.previous_frame):
            points = np.array([item.pitch for item in history])
            mode, current_scale = "pitch", 1.0
        elif self.config.use_relative and references:
            common = set(references) - {track.token}
            for item in history:
                common.intersection_update(item.references)
            nearest = sorted(common, key=lambda key: float(np.linalg.norm(references[key][0] - points[-1])))[:8]
            if len(nearest) >= 3:
                positions = np.array([[item.references[key][0] for key in nearest] for item in history])
                heights = np.array([[item.references[key][1] for key in nearest] for item in history])
                centers, scales = np.median(positions, axis=1), np.median(heights, axis=1)
                layout = (positions - centers[:, None, :]) / scales[:, None, None]
                stable = np.max(np.linalg.norm(layout - layout[0], axis=2), axis=0) < 0.6
                nearest = [key for key, valid in zip(nearest, stable) if valid]
                if len(nearest) >= 3:
                    centers = np.median(positions[:, stable], axis=1)
                    current_center = np.median([references[key][0] for key in nearest], axis=0)
                    current_positions = np.array([references[key][0] for key in nearest])
                    current_layout = (current_positions - current_center) / height
                    previous_layout = (positions[-1, stable] - centers[-1]) / height
                    if np.max(np.linalg.norm(current_layout - previous_layout, axis=1)) < 0.6:
                        # 邻居框高受转身、遮挡抖动影响，会放大远离参照中心的人物位移。
                        # 同一拟合窗统一使用目标可信框高中位数，避免每帧改变坐标尺度。
                        current_scale = height
                        points, mode = (points - centers) / height, "relative"
                    else:
                        current_center = np.zeros(2)
        if mode == "image":
            points = points / height
        origin, velocity, residual = fit_velocity(times, points)
        enough = len(history) >= 3 and times[-1] - times[0] >= 0.08
        if not self.config.use_motion:
            velocity = np.zeros(2)
        return mode, origin, velocity, residual, enough, current_center, current_scale

    def anomalies(self, boxes):
        """标记明显包含两个独立紧框的大框，不统一抑制人物重叠。"""
        bad = np.zeros(len(boxes), dtype=bool)
        if not self.config.protect_boxes or not len(boxes):
            return bad
        size = np.maximum(1, boxes[:, 2:] - boxes[:, :2])
        for index, box in enumerate(boxes):
            intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
            contained = np.flatnonzero((intersection / size.prod(axis=1) > 0.8) & (size.prod(axis=1) < size[index].prod() * 0.7))
            if len(contained) >= 2:
                pairwise = box_iou(boxes[contained], boxes[contained])
                if np.any(pairwise[np.triu_indices(len(contained), 1)] < 0.3):
                    bad[index] = True
        return bad

    def costs(self, tracks, boxes, colors, pitch, references, time, bad):
        """构建归一化联合代价和硬可达性门禁，颜色不替代空间证据。"""
        costs = np.full((len(tracks), len(boxes)), 1e6)
        details = {}
        if not len(boxes):
            return costs, details
        foot = feet(boxes)
        sizes = np.maximum(1, boxes[:, 2:] - boxes[:, :2])
        areas = sizes.prod(axis=1)
        intersections = np.maximum(0, np.minimum(boxes[:, None, 2:], boxes[None, :, 2:]) - np.maximum(boxes[:, None, :2], boxes[None, :, :2])).prod(axis=2)
        has_nested = np.any((intersections / areas[None] > 0.8) & (areas[None] < areas[:, None] * 0.7), axis=1)
        for row, track in enumerate(tracks):
            mode, origin, velocity, residual, enough, center, scale = self.model(track, references, time)
            gap = time - track.history[-1].time
            limit = self.config.image_gap_s if mode == "image" else self.config.max_gap_s
            if gap > limit + 1e-8:
                continue
            candidate_points = (foot - center) / scale
            available = np.ones(len(boxes), dtype=bool)
            if mode == "pitch":
                available = np.array([point is not None for point in pitch])
                candidate_points = np.array([point if point is not None else [1e6, 1e6] for point in pitch])
            prediction = origin + velocity * gap
            tolerance = self.config.position_tolerance * ((0.65 + 3.0 * gap) if mode == "pitch" else (0.28 + 0.75 * gap)) + 2 * residual
            error = np.linalg.norm(candidate_points - prediction, axis=1)
            distance_cost = np.minimum(1, error / max(tolerance, 1e-6))
            predicted_box = track.history[-1].box.copy()
            image_prediction = prediction * scale + center if mode != "pitch" else feet(predicted_box[None])[0]
            # Kalman作为短期辅助；稳健真实观测速度是主要预测。
            if mode == "image" and gap <= 0.12:
                kalman_foot = np.array([track.mean[0], track.mean[1] + track.mean[3] / 2])
                image_prediction = 0.85 * image_prediction + 0.15 * kalman_foot
            predicted_box += np.tile(image_prediction - feet(predicted_box[None])[0], 2)
            overlap = box_iou(predicted_box[None], boxes)[0]
            total, weight = 0.5 * distance_cost + 0.15 * (1 - overlap), np.full(len(boxes), 0.65)
            if self.config.use_motion and enough:
                displacement = candidate_points - origin
                speed = float(np.linalg.norm(velocity))
                if speed > (0.3 if mode == "pitch" else 0.08):
                    cosine = np.sum(displacement * velocity, axis=1) / np.maximum(1e-8, np.linalg.norm(displacement, axis=1) * speed)
                    direction = (1 - np.clip(cosine, -1, 1)) / 2
                    # 急停附近方向证据弱，避免静止或转身强迫换号。
                    direction[np.linalg.norm(displacement, axis=1) < (0.15 if mode == "pitch" else 0.04)] = 0
                    total += 0.2 * (0.5 * direction + 0.5 * distance_cost)
                    weight += 0.2
            color_distances = self.color_distances(track, colors)
            for column, color_distance in enumerate(color_distances):
                if color_distance is not None:
                    total[column] += 0.15 * color_distance
                    weight[column] += 0.15
            old_size = np.median([item.box[2:] - item.box[:2] for item in track.history], axis=0)
            ratio = np.max(np.maximum(sizes / np.maximum(1, old_size), old_size / sizes), axis=1)
            nested = has_nested & (ratio > self.config.size_ratio) if self.config.protect_boxes else np.zeros(len(boxes), bool)
            valid = available & (error <= tolerance) & ~bad & ~nested
            if mode == "pitch":
                valid &= np.linalg.norm(candidate_points - origin, axis=1) <= 12 * gap + 1.0 + residual * 2
            # 颜色仅作辅助代价，光照突变不能单独否决空间和运动都连续的观测。
            values = total / weight
            values += np.where(ratio > self.config.size_ratio, 0.08, 0) if self.config.protect_boxes else 0
            costs[row, valid] = values[valid]
            for column in np.flatnonzero(valid):
                details[row, int(column)] = {"mode": mode, "cost": float(values[column]), "position_error": float(error[column]),
                    "tolerance": float(tolerance), "iou": float(overlap[column]), "color_distance": color_distances[column],
                    "gap_s": float(gap), "mild_box": bool(ratio[column] > self.config.size_ratio)}
        if any(point is None for point in pitch) and self.calibration is not None:
            # 当前检测不能投影时，只允许短期画面/相对运动回退，不跨无效区间外推球场坐标。
            missing = [i for i, point in enumerate(pitch) if point is None]
            fallback, fallback_details = self.costs_without_pitch(tracks, boxes[missing], [colors[i] for i in missing], references, time, bad[missing])
            for fallback_column, column in enumerate(missing):
                costs[:, column] = fallback[:, fallback_column]
                for row in range(len(tracks)):
                    if (row, fallback_column) in fallback_details:
                        details[row, column] = fallback_details[row, fallback_column]
        return costs, details

    def costs_without_pitch(self, tracks, boxes, colors, references, time, bad):
        """暂时禁用投影计算短期回退代价；不改动已有观测。"""
        calibration = self.calibration
        try:
            self.calibration = None
            return self.costs(tracks, boxes, colors, [None] * len(boxes), references, time, bad)
        finally:
            self.calibration = calibration

    def assign(self, costs):
        """带独立不匹配选项的一对一分配；竞争不足的匹配保留待后续确认。"""
        if not costs.shape[0] or not costs.shape[1]:
            return []
        count = len(costs)
        augmented = np.concatenate((costs, np.full((count, count), self.config.max_cost + 1e-7)), axis=1)
        rows, columns = linear_sum_assignment(augmented)
        result = []
        for row, column in zip(rows, columns):
            if column >= costs.shape[1] or costs[row, column] > self.config.max_cost:
                continue
            if self.config.competition_margin:
                alternatives = np.r_[np.delete(costs[row], column), np.delete(costs[:, column], row)]
                if len(alternatives) and float(alternatives.min()) - costs[row, column] < self.config.competition_margin:
                    continue
            result.append((int(row), int(column)))
        return result

    def update_track(self, track, observation, mild=False):
        """更新已分配观测；异常尺寸降低Kalman权重，不污染可信模板。"""
        measured = measurement(observation.box)
        mean, covariance = self.kalman.update(track.mean, track.covariance, measured)
        alpha = 0.2 if mild else max(0.35, min(1.0, observation.score / 0.35))
        track.mean = track.mean + alpha * (mean - track.mean)
        track.covariance = (1 - alpha) * track.covariance + alpha * covariance
        track.last_time = observation.time
        track.maximum_score = max(track.maximum_score, observation.score)
        track.observations.append(observation)
        if not mild:
            track.history.append(observation)
            while len(track.history) > 1 and observation.time - track.history[0].time > self.config.history_s:
                track.history.popleft()
            if observation.color is not None:
                track.colors.append((observation.time, observation.color))
                while track.colors and (observation.time - track.colors[0][0] > self.config.color_window_s or len(track.colors) > self.config.color_samples):
                    track.colors.popleft()

    def submit(self, packet, detections, color_features=None):
        """提交当前真实检测；返回等待结束的帧批次，可预传缓存颜色以复现实验。"""
        time, frame = packet.timestamp_s, packet.frame_index
        if self.closed or not math.isfinite(time) or time <= self.previous_time or frame <= self.previous_frame:
            raise ValueError("跟踪器已结束或源帧时间未递增")
        elapsed = (time - self.previous_time) if math.isfinite(self.previous_time) else self.stride / self.fps
        self.previous_time, self.previous_frame = time, frame
        boxes = np.asarray(detections.xyxy, dtype=float)
        scores = np.asarray(detections.confidence, dtype=float)
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("检测框与置信度必须有效且有限")
        colors = list(color_features) if color_features is not None else [shirt_histogram(packet.image, box) for box in boxes]
        if len(colors) != len(boxes):
            raise ValueError("颜色缓存与检测索引不一致")
        retained = []
        for track in self.tracks:
            gap = time - track.last_time
            if gap > self.config.max_gap_s + 1e-8 or (track.track_id is None and time - track.created_time > self.config.delay_s + 1e-8):
                self.event(frame, "expired", token=track.token, track_id=track.track_id)
                continue
            for _ in range(max(1, round(elapsed * self.fps / self.stride))):
                track.mean, track.covariance = self.kalman.predict(track.mean, track.covariance)
            retained.append(track)
        self.tracks = retained
        references = self.references(boxes, colors, time)
        bad = self.anomalies(boxes)
        pitch = [self.project(box, frame) if not bad[i] else None for i, box in enumerate(boxes)]
        bindings = {}
        self.buffer.append((packet, detections, bindings))
        used = set()
        # 成熟身份先分配，待确认对象不能抢走已有身份的检测。
        for confirmed in (True, False):
            tracks = [track for track in self.tracks if (track.track_id is not None) == confirmed]
            costs, details = self.costs(tracks, boxes, colors, pitch, references, time, bad)
            for index in used:
                costs[:, index] = 1e6
            costs[:, scores < self.config.detection_floor] = 1e6
            for row, index in self.assign(costs):
                track = tracks[row]
                evidence = details[row, index]
                observation = Observation(frame, time, index, boxes[index].copy(), float(scores[index]), colors[index],
                                          None if evidence["mild_box"] else pitch[index], references.copy())
                self.update_track(track, observation, evidence["mild_box"])
                bindings[index] = track
                used.add(index)
                self.event(frame, "matched", token=track.token, track_id=track.track_id, detection_index=index, **evidence)
                if track.track_id is None and len(track.observations) >= 2 and track.maximum_score >= self.config.new_threshold:
                    track.track_id = self.next_id
                    self.next_id += 1
                    self.event(frame, "confirmed", token=track.token, track_id=track.track_id)
        for index, box in enumerate(boxes):
            if index in used:
                continue
            self.event(frame, "unmatched_detection", detection_index=index,
                       reason_detail="multibody" if bad[index] else "no_acceptable_one_to_one_assignment")
            if bad[index]:
                self.event(frame, "multibody_box", detection_index=index)
                continue
            if scores[index] < self.config.detection_floor:
                self.event(frame, "unassigned_low_score", detection_index=index)
                continue
            # 只抑制已有人物紧框的同头部大框，附近独立脚点仍可建立新人。
            duplicate = False
            if self.config.protect_boxes:
                for other in used:
                    small = boxes[other]
                    size = np.maximum(1, small[2:] - small[:2])
                    covered = np.maximum(0, np.minimum(box[2:], small[2:]) - np.maximum(box[:2], small[:2])).prod() / size.prod()
                    if covered > 0.85 and box[3] - box[1] > 1.5 * size[1] and abs(box[1] - small[1]) < 0.2 * size[1]:
                        duplicate = True
                        break
            if duplicate:
                self.event(frame, "duplicate_birth_deferred", detection_index=index)
                continue
            observation = Observation(frame, time, index, box.copy(), float(scores[index]), colors[index], pitch[index], references.copy())
            mean, covariance = self.kalman.initiate(measurement(box))
            track = Tracklet(self.next_token, None, [observation], deque([observation]),
                deque([(time, colors[index])] if colors[index] is not None else []), mean, covariance, float(scores[index]), time, time)
            self.next_token += 1
            self.tracks.append(track)
            bindings[index] = track
            self.event(frame, "birth_attempt", token=track.token, detection_index=index)
        # 实际分配结果成为下一帧可查的邻近参照；不写入异常观测。
        final_references = {track.token: (feet(boxes[index:index + 1])[0], float(boxes[index, 3] - boxes[index, 1]))
            for index, track in bindings.items() if track.track_id is not None and track.history[-1].frame == frame}
        for track in bindings.values():
            if track.history[-1].frame == frame:
                track.history[-1].references = final_references.copy()
        return self.drain(time)

    def drain(self, time, final=False):
        """按时间顺序发布确认观测；未确认检测仍保留在原始缓存。"""
        output = []
        while self.buffer and (final or time - self.buffer[0][0].timestamp_s >= self.config.delay_s - 1e-9
                               or all(track.track_id is not None for track in self.buffer[0][2].values())):
            packet, detections, bindings = self.buffer.popleft()
            indices = sorted(index for index, track in bindings.items() if track.track_id is not None)
            if indices:
                selected = detections[np.array(indices, dtype=int)]
                selected.tracker_id = np.array([bindings[index].track_id for index in indices], dtype=int)
                selected.data["source_detection_index"] = np.array(indices, dtype=int)
            else:
                selected = sv.Detections.empty()
                selected.tracker_id = np.empty(0, dtype=int)
                selected.data["source_detection_index"] = np.empty(0, dtype=int)
            self.stats["output_observations"] += len(selected)
            self.stats["output_frames"] += 1
            output.append(TrackedFrame(packet, selected))
        # 已确认历史只保留拟合窗口；待确认观测最多0.2秒，避免全视频内存增长。
        for track in self.tracks:
            if track.track_id is not None:
                track.observations = track.observations[-2:]
        return output

    def flush(self):
        """结束运行，输出全部剩余帧；未获两次支持的目标不伪造确认。"""
        self.closed = True
        return self.drain(self.previous_time, final=True)
