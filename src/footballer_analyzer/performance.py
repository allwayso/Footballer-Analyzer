"""提供可复用的阶段耗时记录器，统一输出帧数、FPS 和每帧耗时。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator


@dataclass(frozen=True)
class StageMetric:
    """记录一个处理阶段的墙上耗时和帧级吞吐。"""

    stage: str
    frames: int
    elapsed_s: float
    fps: float
    ms_per_frame: float
    source_fps: float | None = None
    source_frames: int | None = None
    realtime_factor: float | None = None
    metadata: dict[str, Any] | None = None


class PerformanceRecorder:
    """收集可复用的阶段计时结果，并支持表格或 JSON 输出。"""

    def __init__(self) -> None:
        """创建空的性能记录器。"""
        self._metrics: list[StageMetric] = []

    @contextmanager
    def stage(self, name: str, frames: int, source_fps: float | None = None, metadata: dict[str, Any] | None = None, *, source_frames: int | None = None) -> Iterator[None]:
        """计时一个已知帧数的阶段，异常退出时不写入不完整结果。"""
        started = time.perf_counter()
        yield
        self.record(name, frames, time.perf_counter() - started, source_fps, metadata, source_frames=source_frames)

    def record(self, name: str, frames: int, elapsed_s: float, source_fps: float | None = None, metadata: dict[str, Any] | None = None, *, source_frames: int | None = None) -> StageMetric:
        """追加一条阶段记录并计算 FPS 与毫秒/帧。"""
        if not name.strip():
            raise ValueError("阶段名称不能为空")
        if type(frames) is not int or frames <= 0:
            raise ValueError("帧数必须是正整数")
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError("耗时必须为有限正数")
        if source_fps is not None and (not math.isfinite(source_fps) or source_fps <= 0):
            raise ValueError("源视频帧率必须为有限正数")
        if source_frames is None:
            source_frames = frames
        if type(source_frames) is not int or source_frames < frames:
            raise ValueError("源区间帧数必须为不小于处理帧数的整数")
        realtime = None if source_fps is None else source_frames / source_fps / elapsed_s
        metric = StageMetric(name, frames, elapsed_s, frames / elapsed_s, elapsed_s * 1000 / frames, source_fps, source_frames, realtime, metadata)
        self._metrics.append(metric)
        return metric

    @property
    def metrics(self) -> tuple[StageMetric, ...]:
        """返回只读的阶段记录。"""
        return tuple(self._metrics)

    def as_dict(self) -> dict[str, Any]:
        """返回适合保存和比较的 JSON 结构。"""
        return {"records": [asdict(metric) for metric in self._metrics]}

    def write_json(self, path: str | Path) -> None:
        """将当前记录写入 UTF-8 JSON 文件。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def print_metrics(recorder: PerformanceRecorder) -> None:
    """以终端表格打印记录器中的阶段结果。"""
    print("阶段\t帧数\t耗时(s)\tFPS\t毫秒/帧\t相对实时")
    for metric in recorder.metrics:
        realtime = "-" if metric.realtime_factor is None else f"{metric.realtime_factor:.2f}x"
        print(f"{metric.stage}\t{metric.frames}\t{metric.elapsed_s:.3f}\t{metric.fps:.2f}\t{metric.ms_per_frame:.1f}\t{realtime}")
