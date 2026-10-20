"""封装 ByteTrack 跨帧记忆；按实际采样帧率设置缓冲，ID 仅在本次运行内有效。"""

from typing import Protocol
import math

import supervision as sv

from .motion_tracking import MotionColorTracker, MotionConfig, TrackedFrame


class Tracker(Protocol):
    """跟踪接口：按时间顺序关联检测，为匹配观测附加轨迹 ID。"""

    def update(self, detections: sv.Detections, timestamp_s: float) -> sv.Detections:
        """接收当前帧检测和源时间（秒），返回带 tracker_id 的观测。"""
        ...


class ByteTracker:
    """维护同一视频窗口的轨迹状态，通过运动预测和框匹配延续编号。"""

    def __init__(self, source_fps, stride, track_activation_threshold=0.25, lost_track_buffer_s=3.0, minimum_matching_threshold=0.8):
        """初始化跨帧状态；丢失缓冲以秒配置，再换算成实际采样帧数。

        检测下限应为 0.1，使低分框能参与第二次关联；新轨迹需达到激活阈值加 0.1。
        库的缓冲按 30 FPS 归一化，因此传入已换算的帧数并固定归一化帧率。
        """
        if not math.isfinite(source_fps) or source_fps <= 0 or type(stride) is not int or stride <= 0:
            raise ValueError("源帧率和采样间隔必须为正")
        if not math.isfinite(lost_track_buffer_s) or lost_track_buffer_s <= 0:
            raise ValueError("丢失缓冲必须为有限正秒数")
        if not 0.1 < track_activation_threshold < 0.9 or not 0 < minimum_matching_threshold < 1:
            raise ValueError("跟踪激活或匹配阈值无效")
        self.tracker = sv.ByteTrack(
            frame_rate=30,
            lost_track_buffer=max(1, math.ceil(source_fps / stride * lost_track_buffer_s)),
            minimum_consecutive_frames=1,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.previous_timestamp = -math.inf

    def update(self, detections, timestamp_s):
        """关联当前帧并返回带 ID 的检测；时间不递增时抛出 ValueError。"""
        if not math.isfinite(timestamp_s) or timestamp_s <= self.previous_timestamp:
            raise ValueError("跟踪时间戳必须递增")
        self.previous_timestamp = timestamp_s
        return self.tracker.update_with_detections(detections)


class LegacyFrameTracker:
    """将原版ByteTrack适配为统一的已确认帧批次接口。"""

    def __init__(self, config, fps):
        """保留旧算法原参数与即时输出语义。"""
        self.tracker = ByteTracker(fps, config.stride, config.track_activation_threshold,
                                   config.lost_track_buffer_s, config.minimum_matching_threshold)
        self.stats = {}

    def submit(self, packet, detections, color_features=None):
        """即时返回原版跟踪结果，不使用颜色与延迟。"""
        return [TrackedFrame(packet, self.tracker.update(detections, packet.timestamp_s))]

    def flush(self):
        """旧算法没有待确认帧。"""
        return []


def create_tracker(config, fps, calibration=None, event_sink=None):
    """按明确算法版本创建逐帧批次跟踪器，旧阈值只传给旧算法。"""
    if config.tracking_algorithm == "bytetrack-legacy":
        return LegacyFrameTracker(config, fps)
    parameters = MotionConfig(max_cost=config.tracking_max_cost, position_tolerance=config.tracking_position_tolerance,
                              competition_margin=config.tracking_competition_margin,
                              new_threshold=config.tracking_new_threshold,
                              max_gap_s=min(3.0, config.lost_track_buffer_s), image_gap_s=min(1.0, config.lost_track_buffer_s))
    return MotionColorTracker(fps, config.stride, parameters, calibration, event_sink)
