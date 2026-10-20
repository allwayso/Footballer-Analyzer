"""生成独立身份标注页和两个指定换号案例的关键帧证据，不修改现有校验页。"""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import cv2
import numpy as np

from footballer_analyzer.retracking import BOX_KEYS
from footballer_analyzer.storage import read_json, read_rows, write_json
from footballer_analyzer.tracking_evaluation import observation_keys
from footballer_analyzer.video import read_frames


def reference_page(source, root):
    """为全部固定片段保存首中末真实观测，并提供可导出的空身份标注表。"""
    reference = read_json(root / 'reference.json')
    truth = read_json(root / 'identity_truth.json')
    manifest = read_json(source / 'manifest.json')
    rows = list(read_rows(source / 'tracks.parquet'))
    indexed = {(row['frame_index'], row['track_id']): row for row in rows}
    wanted = defaultdict(list)
    for segment in reference['segments']:
        frames = segment['field_frames']
        for number, frame in enumerate((frames[0], frames[len(frames) // 2], frames[-1])):
            wanted[frame].append((segment['record_id'], number, indexed[frame, segment['track_id']]))
    images = root / 'reference_images'
    images.mkdir(exist_ok=True)
    for packet in read_frames(Path(manifest['config']['video']), manifest['start_frame'], manifest['end_frame'], manifest['config']['stride']):
        for record, number, row in wanted.get(packet.frame_index, []):
            x1, y1, x2, y2 = [int(row[key]) for key in BOX_KEYS]
            crop = packet.image[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8]
            if crop.size:
                cv2.imwrite(str(images / f'{record}-{number}.jpg'), cv2.resize(crop, (100, 150)))
    cards = []
    for index, segment in enumerate(reference['segments']):
        record = segment['record_id']
        cards.append(f'<article><b>{html.escape(record)}</b><p>帧 {segment["start_frame"]}–{segment["end_frame"]}，{segment["observation_count"]}次观测</p>'
                     + ''.join(f'<img src="reference_images/{record}-{number}.jpg" alt="首中末观测">' for number in range(3))
                     + f'<p>真实人物 <input data-id="{index}" placeholder="未知留空；非人物填ignore"></p><label><input type="checkbox" data-consistent="{index}"> 已回看整段，内部身份一致</label></article>')
    payload = json.dumps(truth, ensure_ascii=False).replace('<', '\\u003c')
    script = """
const truth = TRUTH;
document.querySelectorAll('[data-id]').forEach(el=>{el.value=truth.segments[+el.dataset.id].person_id||'';el.oninput=()=>truth.segments[+el.dataset.id].person_id=el.value.trim()||null;});
document.querySelectorAll('[data-consistent]').forEach(el=>{el.checked=truth.segments[+el.dataset.consistent].internally_consistent===true;el.onchange=()=>truth.segments[+el.dataset.consistent].internally_consistent=el.checked?true:null;});
document.querySelector('#save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(truth,null,2)],{type:'application/json'}));a.download='identity_truth.json';a.click();URL.revokeObjectURL(a.href);};
""".replace('TRUTH', payload)
    (root / 'identity_review.html').write_text(f'<!doctype html><meta charset="utf-8"><title>固定{len(reference["segments"])}条片段身份核验</title><style>body{{font-family:system-ui;background:#17232c;color:white;padding:24px}}main{{display:flex;flex-wrap:wrap;gap:16px}}article{{width:340px;padding:16px;background:#263844}}img{{width:100px;height:150px;object-fit:contain}}input{{padding:6px}}button{{padding:12px;position:sticky;top:12px}}</style><h1>固定片段身份核验</h1><p>三张缩略图仅供导航。需回看完整片段才能确认身份一致；串号需在JSON的observation_overrides中按frame:index逐观测覆盖。未知保持空值，不算正确关联。</p><button id="save">导出身份真值</button><main>' + ''.join(cards) + '</main><script>' + script + '</script>', encoding='utf-8')


def case_evidence(source, run, root):
    """按原检测精确索引对照指定案例，并保存包含邻近人物的放大图。"""
    baseline = observation_keys(source, source)
    current = observation_keys(run, source)
    mapping = {key: row for row, key in current if key}
    by_frame = defaultdict(list)
    for row, key in current:
        by_frame[row['frame_index']].append((row, key))
    cases = [(38, 15), (46, 66), (65, 71), (69, 71), (134, 2), (140, 98)]
    selected, report = {}, []
    for frame, old_id in cases:
        old, key = next((row, key) for row, key in baseline if row['frame_index'] == frame and row['track_id'] == old_id)
        selected[frame] = (old, key)
        row = mapping.get(key)
        report.append({'frame': frame, 'old_id': old_id, 'detection_key': key,
                       'new_id': row['track_id'] if row else None, 'human_identity_confirmed': False})
    manifest = read_json(source / 'manifest.json')
    tiles = []
    for packet in read_frames(Path(manifest['config']['video']), 0, max(selected) + 1, 1):
        if packet.frame_index not in selected:
            continue
        old, key = selected[packet.frame_index]
        center = [(old['bbox_x1'] + old['bbox_x2']) / 2, (old['bbox_y1'] + old['bbox_y2']) / 2]
        left, top = max(0, int(center[0]) - 125), max(0, int(center[1]) - 85)
        right, bottom = min(packet.image.shape[1], left + 250), min(packet.image.shape[0], top + 170)
        crop = packet.image[top:bottom, left:right].copy()
        for row, row_key in by_frame[packet.frame_index]:
            x1, y1, x2, y2 = [int(row[name]) for name in BOX_KEYS]
            if x2 < left or x1 > right or y2 < top or y1 > bottom:
                continue
            color = (30, 30, 255) if row_key == key else (20, 240, 100)
            cv2.rectangle(crop, (x1-left, y1-top), (x2-left, y2-top), color, 1)
            cv2.putText(crop, str(row['track_id']), (x1-left, max(10, y1-top-2)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
        tile = cv2.resize(crop, (750, 510))
        cv2.putText(tile, f"frame {packet.frame_index} / old #{old['track_id']}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        tiles.append(tile)
    root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / 'cases.jpg'), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    write_json(root / 'cases.json', {'observations': report, 'note': '同检测索引同编号只验证连续性；邻近人物归属仍需看图或回放，不等价于全片准确率。'})


def main():
    """生成标注参考或候选运行的关键帧对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--skip-reference-images', action='store_true')
    args = parser.parse_args()
    if not args.skip_reference_images:
        reference_page(args.source, args.reference_dir)
    if args.candidate:
        case_evidence(args.source, args.candidate, args.reference_dir / 'case_evidence' / args.candidate.name)


if __name__ == '__main__':
    main()
