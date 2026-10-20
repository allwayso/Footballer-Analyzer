"""根据上衣 HSV 特征聚为两组并按轨迹投票；颜色簇需人工映射到真实队伍。"""

from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans


def shirt_histogram(frame, box):
    """提取上衣中央的彩色HS及低饱和度亮度直方图，返回归一化分布。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    image_height, image_width = frame.shape[:2]
    x0, x1 = np.clip([int(left + width * 0.25), int(left + width * 0.75)], 0, image_width)
    y0, y1 = np.clip([int(top + height * 0.18), int(top + height * 0.50)], 0, image_height)
    crop = frame[y0:y1, x0:x1]
    if crop.size < 30:
        return None
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    chromatic = pixels[:, 1] >= 50
    indices = np.where(chromatic, (pixels[:, 0].astype(int) * 12 // 180) * 3 + pixels[:, 1].astype(int) * 3 // 256,
                       36 + pixels[:, 2].astype(int) * 8 // 256)
    histogram = np.bincount(indices, minlength=44).astype(float)
    return histogram / histogram.sum()


def shirt_feature(frame, box):
    """从 BGR 图像的 xyxy 框内提取上衣中央颜色特征；裁剪过小时返回 None。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    crop = frame[max(0, int(top + height * 0.18)):int(top + height * 0.50), max(0, int(left + width * 0.25)):int(left + width * 0.75)]
    if crop.size < 30:
        return None
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue, saturation, value = np.median(pixels, axis=0)
    angle = hue / 180 * 2 * np.pi
    saturation /= 255
    return np.array([np.cos(angle) * saturation, np.sin(angle) * saturation, saturation, value / 255 * 0.25])


class TeamClassifier:
    """保存两类球衣聚类模型与各轨迹的累计投票。"""

    def __init__(self):
        """创建未拟合模型和空投票表。"""
        self.model = None
        self.votes = {}

    def fit(self, features):
        """至少有四个有效特征时拟合两类 KMeans；样本不足则保留原模型。"""
        if len(features) >= 4:
            self.model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.asarray(features))

    def predict(self, frame, box, track_id):
        """返回该轨迹累计票数最多的 cluster_0/1；缺模型或特征时返回 unknown。"""
        feature = shirt_feature(frame, box)
        if self.model is None or feature is None:
            return "unknown"
        cluster = int(self.model.predict(feature.reshape(1, -1))[0])
        self.votes.setdefault(track_id, Counter())[cluster] += 1
        winner = self.votes[track_id].most_common(1)[0][0]
        return f"cluster_{winner}"
