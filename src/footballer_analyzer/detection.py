"""用 YOLO 检测单帧人物，输出原图像素框；跨帧身份由 tracking 模块维护。"""

from typing import Protocol

import numpy as np
import supervision as sv
from ultralytics import YOLO

DEVICE_CHOICES = ("cpu", "gpu")


def resolve_device(device):
    """校验设备选项；gpu 在当前 PyTorch 构建不可用时抛出可执行的修复提示。

    CUDA 与 ROCm 构建都复用 torch.cuda 接口，因此 gpu 映射为 0；不可用说明装的是
    CPU 版 torch，需另行安装 CUDA 或 ROCm 版本。
    """
    if device not in DEVICE_CHOICES:
        raise ValueError(f"推理设备仅支持 {' 或 '.join(DEVICE_CHOICES)}")
    if device == "cpu":
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            f"请求 GPU 推理但当前 PyTorch（{torch.__version__}）没有可用 GPU；"
            "请安装 CUDA 或 ROCm 版本 torch，或改用 device=cpu"
        )
    return 0


def gpu_name():
    """返回当前 GPU 名称；无 GPU 时返回空字符串，供运行清单如实记录。"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - 设备信息缺失不应中断分析
        return ""
    return ""


class Detector(Protocol):
    """检测接口：接收 BGR 图像，返回 supervision 检测结果。"""

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """返回当前帧的像素框、类别和置信度，不分配轨迹 ID。"""
        ...


class YoloDetector:
    """YOLO person 检测器，仅保留 COCO 类别 0；设备由构造参数决定。"""

    def __init__(self, weights, imgsz=1280, confidence=0.1, nms_iou=0.5, device="cpu"):
        """加载本地权重；confidence 为检测下限，nms_iou 控制重复框去重。

        device 为 cpu 或 gpu；gpu 需要 CUDA 或 ROCm 版 PyTorch，不可用时明确报错。
        """
        if not weights.is_file():
            raise ValueError(f"缺少本地模型权重：{weights}")
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.device = resolve_device(device)
        self.requested_device = device

    def detect(self, frame):
        """检测 BGR 帧中的人物，返回原图坐标下的 sv.Detections。"""
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.confidence, iou=self.nms_iou, classes=[0], device=self.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
