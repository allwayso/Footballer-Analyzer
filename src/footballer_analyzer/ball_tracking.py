"""离线筛选逐帧球候选，利用时间与位置连续性抑制跳点；漏检保持为空。"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BallSelectionConfig:
    """筛选参数；速度按图像对角线每秒计，间隔按源视频秒计。"""

    max_speed: float = 0.65
    max_gap_s: float = 0.4
    confidence_floor: float = 0.15
    continuation_bonus: float = 0.6
    beam_width: int = 24


def select_ball_candidates(records, fps, image_size, config=None):
    """返回每帧一个原始候选或 None，不插值、不修改检测框。

    records 含递增源帧 frame 和原图 xyxy+confidence 候选；image_size 为宽高。
    离线束搜索保留多条路径，按实际时间限制位移；超时允许重新捕获。
    该方法无法单凭连续性区分长期静止的比赛用球与相似杂物。
    """
    config = config or BallSelectionConfig()
    if not math.isfinite(fps) or fps <= 0 or len(image_size) != 2 or any(not math.isfinite(value) or value <= 0 for value in image_size):
        raise ValueError("帧率与图像尺寸必须为正有限值")
    if config.beam_width < 1 or config.max_gap_s <= 0 or config.max_speed <= 0 or not 0 < config.confidence_floor < 1:
        raise ValueError("球筛选参数无效")
    diagonal = math.hypot(*image_size)
    beams = [(0.0, None, None, ())]
    previous_frame = -1
    for record in records:
        frame = record["frame"]
        if not isinstance(frame, int) or frame <= previous_frame:
            raise ValueError("源帧号必须为非负且严格递增的整数")
        previous_frame = frame
        candidates = record["candidates"]
        for candidate in candidates:
            if len(candidate) != 5 or any(not math.isfinite(value) for value in candidate) or not 0 < candidate[4] <= 1 or candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
                raise ValueError("球候选必须含有效的 xyxy 像素框与置信度")
        expanded = []
        for score, last_frame, center, path in beams:
            elapsed = (frame - last_frame) / fps if last_frame is not None else math.inf
            active = elapsed <= config.max_gap_s
            expanded.append((score - (0.15 if active else 0), last_frame if active else None, center if active else None, path + (None,)))
            for candidate_index, candidate in enumerate(candidates):
                candidate_center = ((candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2)
                distance = math.dist(center, candidate_center) if active else 0
                gate = diagonal * (0.008 + config.max_speed * elapsed) if active else math.inf
                if distance > gate:
                    continue
                reward = math.log(candidate[4] / config.confidence_floor)
                reward += config.continuation_bonus - 0.4 * (distance / gate) ** 2 if active else -0.4
                expanded.append((score + reward, frame, candidate_center, path + (candidate_index,)))
        unique = {}
        for state in sorted(expanded, key=lambda entry: entry[0], reverse=True):
            key = state[1:3]
            if key not in unique:
                unique[key] = state
            if len(unique) >= config.beam_width:
                break
        beams = list(unique.values())
    if not records:
        return []
    return [None if index is None else record["candidates"][index].copy() for record, index in zip(records, beams[0][3])]
