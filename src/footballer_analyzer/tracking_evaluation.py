"""冻结原片段、建立人工真值入口并独立统计跟踪覆盖与关联错误。"""

from collections import defaultdict
from pathlib import Path
import json

import numpy as np

from .calibration import load_calibration
from .field_review import field_segments
from .retracking import BOX_KEYS, detection_frames, input_snapshot
from .storage import read_json, read_rows, write_json


def observation_keys(run, source):
    """将输出框严格映射到原检测；只用于评估旧格式，不进行IoU身份重分配。"""
    raw = detection_frames(source)
    indexed = {}
    for frame, rows in raw.items():
        indexed[frame] = np.array([[row[key] for key in BOX_KEYS] for row in rows]).reshape(-1, 4)
    output = []
    for row in read_rows(Path(run) / 'tracks.parquet'):
        boxes = indexed.get(row['frame_index'], np.empty((0, 4)))
        equal = np.flatnonzero(np.all(np.isclose(boxes, [row[key] for key in BOX_KEYS], atol=1e-4, rtol=0), axis=1))
        output.append((row, f"{row['frame_index']}:{int(equal[0])}" if len(equal) == 1 else None))
    return output


def freeze_reference(source, destination):
    """保存固定分母和空人工标注模板；不把旧编号冒充真实身份。"""
    source, destination = Path(source), Path(destination)
    manifest = read_json(source / 'manifest.json')
    rows = list(read_rows(source / 'tracks.parquet'))
    segments = field_segments(rows, manifest, load_calibration(source))
    snapshot = {'input': input_snapshot(source), 'field_segment_count': len(segments),
                'full_frame_track_count': len({row['track_id'] for row in rows}),
                'observation_count': len(rows), 'segments': segments}
    path = destination / 'reference.json'
    if path.exists() and read_json(path) != snapshot:
        raise ValueError('固定基准发生变化，禁止覆盖原参考')
    write_json(path, snapshot)
    truth = destination / 'identity_truth.json'
    if not truth.exists():
        write_json(truth, {'schema_version': 1, 'input': snapshot['input'],
                          'instructions': '逐片段人工确认真实人物；内部串号用observation_overrides的frame:index覆盖，未知保留null。非人物用ignore。',
                          'segments': [{**item, 'person_id': None, 'internally_consistent': None} for item in segments],
                          'observation_overrides': {},
                          'coverage_review': {'complete': False, 'true_detection_keys': []}})
    return snapshot


def evaluate_tracking(run, source, reference_dir):
    """统计代理覆盖和新增续接审核清单；人工未知关联按未正确统计。"""
    run, source, reference_dir = Path(run), Path(source), Path(reference_dir)
    reference = read_json(reference_dir / 'reference.json')
    criteria = read_json(reference_dir / 'acceptance.json') if (reference_dir / 'acceptance.json').exists() else {}
    target = criteria.get('field_segment_target', reference['field_segment_count'] // 2)
    truth = read_json(reference_dir / 'identity_truth.json')
    if reference['input'] != input_snapshot(source) or truth['input'] != reference['input']:
        raise ValueError('人工标注与当前输入不匹配')
    baseline = observation_keys(source, source)
    candidate = observation_keys(run, source)
    baseline_ids = {key: row['track_id'] for row, key in baseline if key}
    baseline_keys, candidate_keys = set(baseline_ids), {key for _, key in candidate if key}
    identities = dict(truth.get('observation_overrides', {}))
    for segment in truth['segments']:
        if segment.get('person_id') and segment.get('internally_consistent') is True:
            for row, key in baseline:
                if key and row['track_id'] == segment['track_id'] and row['frame_index'] in segment['field_frames']:
                    identities.setdefault(key, segment['person_id'])
    grouped = defaultdict(list)
    for row, key in candidate:
        grouped[row['track_id']].append((row, key))
    correct = wrong = unknown = total = 0
    errors, review, polluted = [], [], set()
    for track_id, items in grouped.items():
        items.sort(key=lambda item: item[0]['frame_index'])
        for position, ((first, left), (last, right)) in enumerate(zip(items, items[1:])):
            a, b = identities.get(left), identities.get(right)
            if a == 'ignore' or b == 'ignore':
                continue
            total += 1
            entry = {'track_id': track_id, 'from': left, 'to': right,
                     'start_frame': first['frame_index'], 'end_frame': last['frame_index'],
                     'duration_s': last['timestamp_s'] - first['timestamp_s']}
            if a is None or b is None:
                unknown += 1
            elif a != b:
                wrong += 1
                polluted.add(track_id)
                continuation_end = last['timestamp_s']
                for following, following_key in items[position + 2:]:
                    if identities.get(following_key) != b:
                        break
                    continuation_end = following['timestamp_s']
                errors.append({**entry, 'from_person': a, 'to_person': b,
                               'wrong_continuation_s': continuation_end - first['timestamp_s']})
            else:
                correct += 1
            if baseline_ids.get(left) != baseline_ids.get(right) or left not in baseline_keys or right not in baseline_keys:
                review.append({**entry, 'truth': 'unknown' if a is None or b is None else ('correct' if a == b else 'wrong')})
    added = sorted(candidate_keys - baseline_keys)
    missing = sorted(baseline_keys - candidate_keys)
    summary = read_json(run / 'tracking_summary.json')
    error_rate = (wrong + unknown) / total if total else None
    complete = total > 0 and unknown == 0
    coverage = truth.get('coverage_review', {})
    coverage_ratio = None
    if coverage.get('complete') is True:
        true_keys = set(coverage.get('true_detection_keys', []))
        raw = detection_frames(source)
        valid_keys = {f'{frame}:{index}' for frame, rows in raw.items() for index in range(len(rows))}
        if not true_keys or not true_keys <= valid_keys:
            raise ValueError('真实人物覆盖核验键为空或不属于当前检测缓存')
        baseline_true = len(true_keys & baseline_keys)
        coverage_ratio = len(true_keys & candidate_keys) / max(1, baseline_true)
    summary.update(baseline_observation_proxy_coverage=len(candidate_keys & baseline_keys) / max(1, len(baseline_keys)),
                   missing_baseline_observations=len(missing), added_detection_observations=len(added),
                   identity_error_rate=error_rate, confirmed_wrong_edges=wrong, unknown_edges=unknown,
                   correct_edges=correct, evaluated_edges=total, id_switches=wrong,
                   polluted_tracks=sorted(polluted), longest_wrong_edge_s=max((item['duration_s'] for item in errors), default=0),
                   longest_wrong_continuation_s=max((item['wrong_continuation_s'] for item in errors), default=0),
                   truth_complete=complete, requires_tightening=bool(complete and error_rate > 0.1),
                   acceptance='pending_identity_truth' if not complete else 'failed_pending_coverage_review')
    summary['true_observation_coverage'] = coverage_ratio
    summary['field_segment_target'] = target
    summary['quantity_target_met'] = summary['field_segments'] <= target
    if complete and coverage_ratio is not None:
        summary['acceptance'] = 'passed' if error_rate <= .1 and coverage_ratio >= 1 and summary['quantity_target_met'] else 'failed'
    write_json(run / 'tracking_summary.json', summary)
    write_json(run / 'tracking_review.json', {'new_edges': review, 'confirmed_errors': errors,
                                             'missing_baseline_keys': missing, 'added_keys': added,
                                             'note': '原版覆盖仅是代理；原版误检、重复框及新增检测必须人工核验，不能当作真实人物覆盖。'})
    return summary
