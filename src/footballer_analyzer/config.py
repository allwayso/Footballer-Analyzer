"""读取并校验分析配置；相对路径以项目根目录为基准，帧区间左闭右开。"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import yaml

from .pitch_models import pitch_spec

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """单次分析配置；坐标尺寸用米，时间用秒，帧号来自源视频。"""

    video: Path
    weights: Path
    output_root: Path
    players: Path
    match_id: str
    clip_id: str
    period: int = 1
    match_offset_s: float | None = None
    start_frame: int = 0
    end_frame: int | None = None
    stride: int = 1
    imgsz: int = 1280
    device: str = "cpu"
    confidence: float = 0.1
    nms_iou: float = 0.5
    track_activation_threshold: float = 0.25
    lost_track_buffer_s: float = 3.0
    minimum_matching_threshold: float = 0.8
    tracking_algorithm: str = "motion-color-v1"
    tracking_max_cost: float = 0.99
    tracking_position_tolerance: float = 3.0
    tracking_competition_margin: float = 0.0
    tracking_new_threshold: float = 0.75
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    speed_limit_mps: float = 12.0
    min_cutoff_hz: float = 1.0
    filter_beta: float = 0.0
    pitch_model: str = "eleven"
    calibration_method: str = "white_lines"

    def snapshot(self):
        """返回可序列化配置，将 Path 转为字符串。"""
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def load_config(path: str | Path) -> Config:
    """读取 YAML 并返回有效配置；非对象内容或非法参数抛出 ValueError。"""
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置必须是 YAML 对象")
    return config_from_dict(payload)


def config_from_dict(payload: dict) -> Config:
    """补齐默认路径并构造配置；校验帧区间、数值范围和比赛标识。"""
    values = dict(payload)
    if values.get("pitch_model") == "eight":
        values.setdefault("pitch_length_m", 68.0)
        values.setdefault("pitch_width_m", 50.0)
        values.setdefault("calibration_method", "manual")
    for key, default in [("output_root", "outputs"), ("players", "data/players.yaml")]:
        values.setdefault(key, default)
    for key in ["video", "weights", "output_root", "players"]:
        if not values.get(key):
            raise ValueError(f"缺少配置：{key}")
        candidate = Path(values[key])
        values[key] = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    config = Config(**values)
    if config.tracking_algorithm not in {"motion-color-v1", "bytetrack-legacy"}:
        raise ValueError("跟踪算法仅支持 motion-color-v1 或 bytetrack-legacy")
    from .motion_tracking import MotionConfig
    MotionConfig(max_cost=config.tracking_max_cost, position_tolerance=config.tracking_position_tolerance,
                 competition_margin=config.tracking_competition_margin, new_threshold=config.tracking_new_threshold)
    pitch_spec(config.pitch_model, config.pitch_length_m, config.pitch_width_m)
    if config.calibration_method not in {"manual", "white_lines"}:
        raise ValueError("标定方式仅支持 manual 或 white_lines")
    if config.pitch_model == "eight" and config.calibration_method != "manual":
        raise ValueError("八人制目前仅支持手动固定标定")
    for key in ["start_frame", "stride", "imgsz", "period"]:
        value = getattr(config, key)
        if type(value) is not int or value < (0 if key == "start_frame" else 1):
            raise ValueError(f"{key} 必须是有效整数")
    if config.device not in {"cpu", "gpu"}:
        raise ValueError("device 仅支持 cpu 或 gpu")
    if config.end_frame is not None and (type(config.end_frame) is not int or config.end_frame <= config.start_frame):
        raise ValueError("end_frame 必须大于 start_frame")
    for key in ["confidence", "pitch_length_m", "pitch_width_m", "speed_limit_mps", "min_cutoff_hz"]:
        value = getattr(config, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    if config.confidence >= 1:
        raise ValueError("confidence 必须小于 1")
    for key in ["nms_iou", "minimum_matching_threshold"]:
        value = getattr(config, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{key} 必须在 (0, 1) 内")
    if not isinstance(config.track_activation_threshold, (int, float)) or not math.isfinite(config.track_activation_threshold) or not 0.1 < config.track_activation_threshold < 0.9:
        raise ValueError("track_activation_threshold 必须在 (0.1, 0.9) 内")
    if not isinstance(config.lost_track_buffer_s, (int, float)) or not math.isfinite(config.lost_track_buffer_s) or config.lost_track_buffer_s <= 0:
        raise ValueError("lost_track_buffer_s 必须为有限正数")
    if not math.isfinite(config.filter_beta) or config.filter_beta < 0:
        raise ValueError("filter_beta 必须为有限非负数")
    if config.match_offset_s is not None and (not math.isfinite(config.match_offset_s) or config.match_offset_s < 0):
        raise ValueError("match_offset_s 必须为空或有限非负数")
    if not config.match_id or not config.clip_id:
        raise ValueError("match_id 和 clip_id 不能为空")
    return config
