"""编排独立人物追踪：核实视频末尾并生成缓存，场地标定由用户另外启动。"""

from dataclasses import replace

import cv2

from .pipeline import track
from .video import inspect_video


def analyze_clip(config, progress):
    """仅生成人物缓存；progress 接收阶段、数量及运行路径，不触发场地标定。

    未指定结束帧时扫描至可解码末尾，避免素材容器虚报帧数；指定区间仍严格校验。
    """
    info = inspect_video(config.video)
    progress("inspecting", 0, info.frame_count, None)
    if config.end_frame is None:
        capture = cv2.VideoCapture(str(config.video))
        decoded = 0
        try:
            while capture.grab():
                decoded += 1
                if decoded % 25 == 0:
                    progress("inspecting", decoded, info.frame_count, None)
        finally:
            capture.release()
        if decoded <= config.start_frame:
            raise ValueError("分析起点之后没有可解码视频帧")
        config = replace(config, end_frame=decoded)
    elif config.end_frame > info.frame_count:
        raise ValueError("分析区间超出视频范围")

    def tracking_progress(stage, processed, total):
        """转发人物预采样与逐帧追踪进度。"""
        progress(stage, processed, total, None)

    run_dir = track(config, tracking_progress)
    total = config.end_frame - config.start_frame
    progress("awaiting_calibration", total, total, run_dir)
    return run_dir
