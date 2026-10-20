"""复用同一检测缓存执行固定18组参数、消融实验及人工参考评估。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from itertools import product
from pathlib import Path

from footballer_analyzer.motion_tracking import MotionConfig
from footballer_analyzer.retracking import prepare_colors, retrack
from footballer_analyzer.storage import write_json, read_json, file_hash
import footballer_analyzer.motion_tracking as motion_module
from footballer_analyzer.tracking_evaluation import freeze_reference, evaluate_tracking


def experiment(task):
    """在独立进程运行一个配置，异常写入实验表而不省略失败项。"""
    source, root, label, config, algorithm, fuse = task
    try:
        run = retrack(source, root / label, config, root / 'colors.npz', False, algorithm, True, fuse)
        summary = evaluate_tracking(run, source, root)
        summary.update(label=label, run_path=str(run.resolve()), status='complete')
        return summary
    except Exception as error:
        return {'label': label, 'status': 'failed', 'error': str(error), 'parameters': config.snapshot()}


def main():
    """冻结分母后运行全部配置，完整保留结果，未标注时仅给出待复核候选。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--field-target', type=int, help='固定场内片段目标；不指定时为原版的一半')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('并行进程数必须在1～4之间')
    root = args.output_root.resolve()
    reference = freeze_reference(args.run, root)
    target = args.field_target if args.field_target is not None else reference['field_segment_count'] // 2
    if target < 1:
        parser.error('场内片段目标必须为正整数')
    criteria = {'field_segment_target': target, 'identity_error_limit': .1, 'minimum_true_coverage': 1.0}
    criteria_path = root / 'acceptance.json'
    if criteria_path.exists() and read_json(criteria_path) != criteria:
        parser.error('已冻结的验收目标不同，请使用新实验目录')
    write_json(criteria_path, criteria)
    prepare_colors(args.run, root / 'colors.npz')
    tasks = []
    reference_config = MotionConfig(max_cost=.75, position_tolerance=1.5, new_threshold=.35)
    for cost, tolerance, margin in product((.65, .75, .85), (1., 1.5, 2.), (0., .03)):
        label = f'grid_c{cost}_p{tolerance}_m{margin}'
        tasks.append((args.run, root, label, replace(reference_config, max_cost=cost, position_tolerance=tolerance, competition_margin=margin), 'motion-color-v1', True))
    if not args.skip_ablation:
        tasks += [(args.run, root, 'legacy', reference_config, 'bytetrack-legacy', True),
                  (args.run, root, 'legacy_no_score', reference_config, 'bytetrack-legacy', False),
                  (args.run, root, 'motion_only', replace(reference_config, use_color=False, protect_boxes=False), 'motion-color-v1', True),
                  (args.run, root, 'motion_color_no_box_protection', replace(reference_config, protect_boxes=False), 'motion-color-v1', True)]
    results = []
    # 恢复已有完成项时仍按冻结的输入、参数和真值重新评估，不重复计算。
    pending = []
    for task in tasks:
        completed = list((root / task[2]).glob('run_*/tracking_summary.json'))
        match = next((path for path in completed if read_json(path).get('parameters') == (task[3].snapshot() if task[4] == 'motion-color-v1' else {'algorithm': task[4], 'fuse_score': task[5]}) and read_json(path).get('tracking_implementation_sha256') == file_hash(Path(motion_module.__file__))), None)
        if match:
            result = evaluate_tracking(match.parent, args.run, root)
            result.update(label=task[2], run_path=str(match.parent), status='complete')
            results.append(result)
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(experiment, task) for task in pending]):
            result = future.result()
            results.append(result)
            write_json(root / 'experiments.json', {'reference': reference, 'experiments': sorted(results, key=lambda row: row['label'])})
            print({key: result.get(key) for key in ('label', 'status', 'field_segments', 'track_count', 'observation_count', 'elapsed_s', 'error')}, flush=True)
    for row in list(results):
        if row.get('requires_tightening'):
            original_task = next(task for task in tasks if task[2] == row['label'])
            parameters = replace(original_task[3], max_cost=max(.1, original_task[3].max_cost - .1),
                                 position_tolerance=original_task[3].position_tolerance * .8,
                                 competition_margin=max(.03, original_task[3].competition_margin))
            retry = experiment((args.run, root, row['label'] + '_tightened', parameters, original_task[4], original_task[5]))
            retry['tightened_from'] = row['label']
            results.append(retry)
    results.sort(key=lambda row: row['label'])
    fields = ['label', 'status', 'field_segments', 'track_count', 'observation_count', 'baseline_observation_proxy_coverage',
              'missing_baseline_observations', 'added_detection_observations', 'identity_error_rate', 'unknown_edges',
              'confirmed_wrong_edges', 'truth_complete', 'acceptance', 'elapsed_s', 'run_path', 'error']
    with (root / 'experiments.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    grid = [row for row in results if row['label'].startswith('grid') and row['status'] == 'complete']
    verified = [row for row in grid if row.get('truth_complete') and row['identity_error_rate'] <= .1 and row.get('true_observation_coverage') is not None and row['true_observation_coverage'] >= 1]
    selected = min(verified or grid, key=lambda row: (row['field_segments'], row.get('identity_error_rate') or 0, -row['baseline_observation_proxy_coverage'])) if grid else None
    write_json(root / 'selection.json', {'status': 'verified' if verified else 'pending_manual_review',
                                        'candidate': selected, 'note': '缺少完整人工真值及真实人物覆盖核验时，不能据数量宣称通过。'})
    write_json(root / 'experiments.json', {'reference': reference, 'experiments': results})


if __name__ == '__main__':
    main()
