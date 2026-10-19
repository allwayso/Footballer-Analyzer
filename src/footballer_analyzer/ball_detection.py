"""检测足球候选：共用全图、重叠切片、原图坐标还原与跨切片去重。"""


def tile_origins(extent, tile_size, overlap):
    """生成覆盖完整图像轴的切片起点，末片贴边且不重复。"""
    if tile_size <= 0 or extent <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    return sorted(set([*range(0, extent - tile_size + 1, stride), extent - tile_size]))


def suppress_duplicates(candidates, threshold=0.3):
    """按置信度进行跨切片 IoU 去重，坐标均为原图像素。"""
    remaining = sorted(candidates, key=lambda entry: entry[4], reverse=True)
    kept = []
    while remaining:
        chosen = remaining.pop(0)
        kept.append(chosen)
        survivors = []
        for candidate in remaining:
            left = max(chosen[0], candidate[0])
            top = max(chosen[1], candidate[1])
            right = min(chosen[2], candidate[2])
            bottom = min(chosen[3], candidate[3])
            intersection = max(0, right - left) * max(0, bottom - top)
            area_chosen = (chosen[2] - chosen[0]) * (chosen[3] - chosen[1])
            area_candidate = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
            if intersection / max(area_chosen + area_candidate - intersection, 1e-9) <= threshold:
                survivors.append(candidate)
        remaining = survivors
    return kept


def detect(model, frame, image_size, tile_size, overlap, confidence):
    """全图或重叠切片检测球；自动核实模型类别，并还原原图框。"""
    ball_ids = [identifier for identifier, name in model.names.items() if name.lower() in {"ball", "sports ball", "football", "soccer ball"}]
    if not ball_ids:
        raise ValueError(f"模型没有可识别的球类别：{model.names}")
    height, width = frame.shape[:2]
    candidates = []
    for top in tile_origins(height, tile_size, overlap):
        for left in tile_origins(width, tile_size, overlap):
            crop = frame[top:min(top + tile_size, height), left:min(left + tile_size, width)] if tile_size else frame
            result = model.predict(crop, imgsz=image_size, conf=confidence, classes=ball_ids, device="cpu", verbose=False, iou=0.5)[0]
            for bounds, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                candidates.append([float(bounds[0] + left), float(bounds[1] + top), float(bounds[2] + left), float(bounds[3] + top), float(score)])
    return suppress_duplicates(candidates)
