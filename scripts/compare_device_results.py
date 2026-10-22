# -*- coding: utf-8 -*-
"""汇总 CPU 与 GPU 的实测速度对比，生成图表与 Markdown 表格。

用法：python scripts/compare_device_results.py
输出：docs/bench/device_compare.json、docs/bench/device_compare.md
      docs/bench/device_compare_*.png（图��不入库，见 .gitignore）
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/bench"
OUT.mkdir(parents=True, exist_ok=True)

# 采集到的实测数据；每条都来自本机真实素材运行，不是外推值。
STAGE_ROWS = [
    # 阶段, CPU FPS, GPU FPS, 说明
    ("人物追踪", 4.36, 9.20, "检测 + 跟踪 + 产物写盘"),
    ("逐帧白线标定", 8.13, 8.08, "无神经网络，GPU 不介入"),
    ("足球追踪", 3.79, 9.80, "每帧 6 块 640 切片检测"),
]

CLIP_ROWS = [
    # 素材, 每帧检测框, CPU FPS, GPU FPS, 端到端加速, 纯推理加速, teams.predict ms
    ("barca_girona_05m00s", 80, 3.65, 6.27, 1.72, 10.56, 18.4),
    ("barca_girona_85m00s", 43, 4.26, 8.59, 2.02, 10.74, 8.8),
    ("clip_105m00s", 46, 4.34, 8.93, 2.05, 9.42, 10.8),
]

IMGSZ_ROWS = [
    # imgsz, CPU ms, GPU ms, CPU FPS, GPU FPS
    (640, 23.8, 10.2, 42.0, 97.7),
    (1280, 65.4, 12.8, 15.3, 78.3),
    (1920, 139.3, 13.6, 7.2, 73.4),
]

INFER_ROWS = [
    # 素材, CPU ms, GPU ms
    ("barca_girona_05m00s", 147.4, 14.0),
    ("barca_girona_85m00s", 144.4, 13.4),
    ("clip_105m00s", 151.5, 16.1),
]

FP16_ROW = {"fp32_ms": 14.0, "fp16_ms": 12.0, "fp32_boxes": 46.2, "fp16_boxes": 46.4,
            "same_count_frames": 13, "total_frames": 20}

ENVIRONMENT = {
    "cpu": platform.processor(),
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)",
    "torch_collect": "2.11.0+cu128",
    "torch_deployed": "2.14.1+cu130",
    "note": "CPU 与 GPU 数据在同一虚拟环境内顺序采集，仅切换 device 参数，避免跨环境差异",
}

# 主环境（2.14.1+cu130）按同一协议复核的结果，用于确认加速比不随 torch 构建变化。
RECHECK = {
    "torch": "2.14.1+cu130",
    "tracking_cpu_fps": 3.76, "tracking_gpu_fps": 8.96, "tracking_speedup": 2.38,
    "ball_cpu_fps": 3.58, "ball_gpu_fps": 9.59, "ball_speedup": 2.68,
}


def setup_font():
    """挑选系统里可用的中文字体，避免图表出现方块。"""
    for candidate in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Microsoft JhengHei"]:
        if any(candidate == font.name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save(fig, name):
    """保存图表到 docs/bench 并关闭。"""
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {path.relative_to(ROOT)}")


def chart_stage():
    """三阶段 CPU/GPU 吞吐对比。"""
    labels = [row[0] for row in STAGE_ROWS]
    cpu = [row[1] for row in STAGE_ROWS]
    gpu = [row[2] for row in STAGE_ROWS]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.36
    ax.bar([i - width / 2 for i in x], cpu, width, label="CPU", color="#8a9ba8")
    ax.bar([i + width / 2 for i in x], gpu, width, label="GPU (RTX 4060)", color="#2f6f4e")
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        ax.text(i - width / 2, c + 0.15, f"{c:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, g + 0.15, f"{g:.2f}", ha="center", fontsize=9)
        ax.text(i, max(c, g) + 0.55, f"{g / c:.2f}×", ha="center", fontsize=10, color="#2f6f4e", weight="bold")
    ax.set_ylabel("吞吐（处理源帧 / 秒）")
    ax.set_title("三阶段吞吐对比 · 同一素材 125 帧 · imgsz 1920")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(cpu), max(gpu)) * 1.22)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "device_compare_stage.png")


def chart_clips():
    """多素材端到端与纯推理加速对比。"""
    labels = [row[0].replace("_", " ") for row in CLIP_ROWS]
    e2e = [row[4] for row in CLIP_ROWS]
    infer = [row[5] for row in CLIP_ROWS]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.36
    ax.bar([i - width / 2 for i in x], infer, width, label="纯推理加速", color="#2f6f4e")
    ax.bar([i + width / 2 for i in x], e2e, width, label="端到端加速", color="#c47f3a")
    for i, (a, b) in enumerate(zip(infer, e2e)):
        ax.text(i - width / 2, a + 0.15, f"{a:.2f}×", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 0.15, f"{b:.2f}×", ha="center", fontsize=9)
    ax.set_ylabel("相对 CPU 的加速倍数")
    ax.set_title("多素材加速对比 · 300 帧 · imgsz 1920")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(infer) * 1.2)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "device_compare_clips.png")


def chart_imgsz():
    """输入尺寸对 CPU/GPU 耗时的影响。"""
    sizes = [str(row[0]) for row in IMGSZ_ROWS]
    cpu = [row[1] for row in IMGSZ_ROWS]
    gpu = [row[2] for row in IMGSZ_ROWS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(sizes, cpu, "o-", label="CPU", color="#8a9ba8", linewidth=2)
    ax1.plot(sizes, gpu, "o-", label="GPU (RTX 4060)", color="#2f6f4e", linewidth=2)
    for sx, c, g in zip(sizes, cpu, gpu):
        ax1.annotate(f"{c:.0f}ms", (sx, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#5a6b78")
        ax1.annotate(f"{g:.0f}ms", (sx, g), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color="#2f6f4e")
    ax1.set_xlabel("输入尺寸 imgsz")
    ax1.set_ylabel("单帧推理耗时（毫秒）")
    ax1.set_title("推理耗时 vs 输入尺寸")
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.25)

    speedups = [c / g for c, g in zip(cpu, gpu)]
    bars = ax2.bar(sizes, speedups, 0.5, color="#2f6f4e")
    for bar, value in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f}×", ha="center", fontsize=10, weight="bold")
    ax2.set_xlabel("输入尺寸 imgsz")
    ax2.set_ylabel("加速倍数")
    ax2.set_title("加速倍数 vs 输入尺寸")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25)
    save(fig, "device_compare_imgsz.png")


def chart_overhead():
    """端到端耗时构成：推理与 CPU 侧环节的占比。"""
    labels = [row[0].replace("_", " ") for row in CLIP_ROWS]
    cpu_infer = [147.4, 144.4, 151.5]
    gpu_infer = [14.0, 13.4, 16.1]
    teams = [18.4, 8.8, 10.8]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.36
    ax.bar([i - width / 2 for i in x], cpu_infer, width, label="推理（CPU）", color="#8a9ba8")
    ax.bar([i - width / 2 for i in x], teams, width, bottom=cpu_infer, label="球衣特征（CPU，两种设备同样耗时）", color="#c47f3a")
    ax.bar([i + width / 2 for i in x], gpu_infer, width, label="推理（GPU）", color="#2f6f4e")
    ax.bar([i + width / 2 for i in x], teams, width, bottom=gpu_infer, color="#e0b080")
    for i, (ci, gi, tm) in enumerate(zip(cpu_infer, gpu_infer, teams)):
        ax.text(i - width / 2, ci + tm + 4, f"共 {ci + tm:.0f}ms", ha="center", fontsize=9)
        ax.text(i + width / 2, gi + tm + 4, f"共 {gi + tm:.0f}ms", ha="center", fontsize=9, color="#2f6f4e", weight="bold")
    ax.set_ylabel("每帧耗时（毫秒）")
    ax.set_title("端到端耗时构成 · 推理加速后瓶颈转移到 CPU 球衣特征提取")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(c + t for c, t in zip(cpu_infer, teams)) * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.72))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "device_compare_overhead.png")


def build_markdown():
    """生成 Markdown 表格。"""
    lines = ["# CPU 与 GPU 实测对比", ""]
    lines += [f"- 采集时间：2026-10-22", f"- 环境：{ENVIRONMENT['platform']}", f"- GPU：{ENVIRONMENT['gpu']}",
              f"- 采集时 torch：{ENVIRONMENT['torch_collect']}", f"- 说明：{ENVIRONMENT['note']}", "",
              f"> **主环境复核（torch {RECHECK['torch']}）**：同一协议重测，人物追踪 "
              f"{RECHECK['tracking_cpu_fps']:.2f} → {RECHECK['tracking_gpu_fps']:.2f} FPS"
              f"（**{RECHECK['tracking_speedup']:.2f}×**），足球追踪 "
              f"{RECHECK['ball_cpu_fps']:.2f} → {RECHECK['ball_gpu_fps']:.2f} FPS"
              f"（**{RECHECK['ball_speedup']:.2f}×**），与下表一致，差异在测量波动范围内。", ""]

    lines += ["## 一、三阶段端到端吞吐", "",
              "同一素材 125 帧，imgsz 1920，每阶段 2 轮正式测量 + 1 轮预热。", "",
              "| 阶段 | CPU (FPS) | GPU (FPS) | 加速 | 阶段内容 |", "|---|---:|---:|---:|---|"]
    for name, cpu, gpu, note in STAGE_ROWS:
        lines.append(f"| {name} | {cpu:.2f} | {gpu:.2f} | **{gpu / cpu:.2f}×** | {note} |")
    lines += ["", "逐帧白线标定不含神经网络，GPU 加速为 0.99×，属于测量波动。", ""]

    lines += ["## 二、多素材端到端与纯推理加速", "",
              "每段 300 帧，imgsz 1920。纯推理为单帧 detect 计时（含 letterbox 与 NMS）。", "",
              "| 素材 | 每帧检测框 | CPU (FPS) | GPU (FPS) | 端到端加速 | 纯推理加速 | 球衣特征 (ms/帧) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, boxes, cpu, gpu, e2e, infer, team_ms in CLIP_ROWS:
        lines.append(f"| {name} | {boxes} | {cpu:.2f} | {gpu:.2f} | {e2e:.2f}× | {infer:.2f}× | {team_ms:.1f} |")
    lines += ["", "纯推理加速在三个素材上稳定在 9–11×，端到端加速却随检测框数量下降——"
              "检测框越多，CPU 侧球衣特征提取占比越高，GPU 加速被稀释。", ""]

    lines += ["## 三、输入尺寸的影响", "",
              "素材 barca_girona_85m00s，单帧 detect 计时。", "",
              "| imgsz | CPU (ms) | CPU (FPS) | GPU (ms) | GPU (FPS) | 加速 |", "|---:|---:|---:|---:|---:|---:|"]
    for size, cms, gms, cfps, gfps in IMGSZ_ROWS:
        lines.append(f"| {size} | {cms:.1f} | {cfps:.1f} | {gms:.1f} | {gfps:.1f} | **{cms / gms:.2f}×** |")
    lines += ["", "GPU 耗时几乎不随输入尺寸变化（10.2→13.6 毫秒），CPU 则近似线性增长，"
              "因此分辨率越高 GPU 优势越大。", ""]

    lines += ["## 四、FP16 探测（不建议启用）", "",
              f"imgsz 1920，RTX 4060 有 Tensor Core。FP32 {FP16_ROW['fp32_ms']:.1f} 毫秒，"
              f"FP16 {FP16_ROW['fp16_ms']:.1f} 毫秒，仅快 {FP16_ROW['fp32_ms'] / FP16_ROW['fp16_ms']:.2f}×。", "",
              f"代价是检测结果改变：20 帧中只有 {FP16_ROW['same_count_frames']} 帧的检出框数量与 FP32 一致"
              f"（每帧平均 {FP16_ROW['fp32_boxes']:.1f} → {FP16_ROW['fp16_boxes']:.1f} 个框）。", "",
              "**结论：17% 的收益不抵精度风险，本轮不启用 FP16。**", ""]

    lines += ["## 五、图表", "",
              "- `device_compare_stage.png` — 三阶段吞吐对比",
              "- `device_compare_clips.png` — 多素材加速对比",
              "- `device_compare_imgsz.png` — 输入尺寸对耗时与加速的影响",
              "- `device_compare_overhead.png` — 端到端耗时构成", "",
              "图片位于 `docs/bench/`，按仓库约定不随公共仓库发布。", ""]
    return "\n".join(lines)


def main():
    """生成全部图表、JSON 数据与 Markdown 表格。"""
    setup_font()
    chart_stage()
    chart_clips()
    chart_imgsz()
    chart_overhead()
    payload = {
        "schema_version": 1,
        "collected_at": "2026-10-22",
        "environment": ENVIRONMENT,
        "stage_throughput_fps": [{"stage": n, "cpu": c, "gpu": g, "speedup": g / c, "note": note} for n, c, g, note in STAGE_ROWS],
        "clips": [{"clip": n, "boxes_per_frame": b, "cpu_fps": c, "gpu_fps": g, "end_to_end_speedup": e,
                   "inference_speedup": i, "teams_ms_per_frame": t} for n, b, c, g, e, i, t in CLIP_ROWS],
        "imgsz": [{"imgsz": s, "cpu_ms": c, "gpu_ms": g, "cpu_fps": cf, "gpu_fps": gf, "speedup": c / g}
                  for s, c, g, cf, gf in IMGSZ_ROWS],
        "fp16_probe": FP16_ROW,
        "recheck_main_env": RECHECK,
    }
    (OUT / "device_compare.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "device_compare.md").write_text(build_markdown(), encoding="utf-8")
    print(f"[out] {(OUT / 'device_compare.json').relative_to(ROOT)}")
    print(f"[out] {(OUT / 'device_compare.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
