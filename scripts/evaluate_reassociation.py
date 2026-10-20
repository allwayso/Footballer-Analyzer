"""固定关联基准、建立待人工真值、独立评估与消融比较的命令行工具。"""

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import time

from footballer_analyzer.association_evaluation import evaluate
from footballer_analyzer.calibration import load_calibration
from footballer_analyzer.field_review import field_segments, merge_groups
from footballer_analyzer.reassociation import (AssociationConfig, analyze_nodes, association_view, collect_evidence, commit_result,
    compute_result, config_snapshot, digest, input_signature, load_review, raw_nodes, visible_candidates, rank_candidates, group_conflict)
from footballer_analyzer.storage import file_hash, read_json, read_rows, write_json


def freeze(run_dir, destination):
    """复制原始输入到新目录并固定分母，已有目录绝不覆盖。"""
    destination.mkdir(parents=True,exist_ok=False)
    for name in ['manifest.json','tracks.parquet','calibration.yaml','catalog.json','bindings.yaml']:
        if (run_dir/name).exists():shutil.copy2(run_dir/name,destination/name)
    calibration=load_calibration(run_dir)
    if calibration and calibration.get('mode')=='white_lines':
        version=calibration['calibration_id'];folder=destination/'calibrations'/version;folder.mkdir(parents=True)
        shutil.copy2(run_dir/'calibrations'/version/'frames.json',folder/'frames.json')
    snapshot_input(destination)


def snapshot_input(run_dir):
    """写入原始清单与未知真值，不把任何预测填入身份标签。"""
    manifest=read_json(run_dir/'manifest.json')
    segments=field_segments(list(read_rows(run_dir/'tracks.parquet')),manifest,load_calibration(run_dir))
    if not segments:
        raise ValueError('没有经确认的有效场内片段，请先完成标定验收，再建立基准')
    snapshot={'run_id':manifest['run_id'],'start_frame':manifest['start_frame'],'end_frame':manifest['end_frame'],
              'video_sha256':manifest['video_sha256'],'tracks_sha256':file_hash(run_dir/'tracks.parquet'),
              'input_signature':input_signature(run_dir),'parameters':config_snapshot(),'segments':segments}
    path=run_dir/'benchmark_input.json'
    if path.exists() and read_json(path)!=snapshot:raise ValueError('已有基准与当前输入不一致，不能覆盖')
    write_json(path,snapshot)
    truth=run_dir/'identity_truth.json'
    if not truth.exists():
        write_json(truth,{'schema_version':1,'input_signature':snapshot['input_signature'],'annotator':None,'independent':False,
                         'records':{item['record_id']:{'player_id':None,'internally_consistent':None,'note':''} for item in segments}})
    return snapshot


def ablate(run_dir):
    """固定同一原始输入，比较位置、运动、相对参照与颜色，不把数量当精度。"""
    manifest=read_json(run_dir/'manifest.json');rows=list(read_rows(run_dir/'tracks.parquet'));calibration=load_calibration(run_dir)
    if not field_segments(rows, manifest, calibration):
        raise ValueError('没有经确认的有效场内片段，不能进行有效消融比较')
    evidence=collect_evidence(run_dir,rows,manifest);results=[]
    variants=[('ByteTrack',None),('位置',replace(AssociationConfig(),use_motion=False,use_color=False,use_relative=False)),
              ('位置+运动',replace(AssociationConfig(),use_color=False,use_relative=False)),
              ('位置+运动+邻近参照',replace(AssociationConfig(),use_color=False)),('位置+运动+邻近参照+颜色',AssociationConfig())]
    for label,config in variants:
        started=time.perf_counter();nodes=raw_nodes(rows,manifest)
        segments=field_segments(rows,manifest,calibration);operations=[]
        if config:
            candidates=analyze_nodes(nodes,manifest,calibration,evidence,config)
            result={'nodes':[{key:value for key,value in node.items() if key!='rows'} for node in nodes],'candidates':candidates}
            while True:
                merge_groups(segments,operations)
                mapped=rank_candidates(visible_candidates(result,segments,{'blocked_pairs':[]}),config)
                added=0
                for candidate in mapped:
                    if candidate['automatic'] and not group_conflict(segments,candidate['source_record_id'],candidate['target_record_id']):
                        operations.append(candidate);merge_groups(segments,operations);added+=1
                        break  # 与实际提交一致：每合一条即重新计算组级竞争。
                if not added:break
        else:candidates=[];merge_groups(segments,[])
        results.append({'variant':label,'raw_records':len(segments),'groups':len({item['merge_group_id'] for item in segments}),
                        'edges':len(operations),'candidates':len(candidates),'elapsed_s':round(time.perf_counter()-started,3),'precision':None,
                        'precision_note':'没有独立完整人工真值，未证明准确率'})
    write_json(run_dir/'ablation.json',results)
    return results


def review_gallery(run_dir, review_base_url='http://127.0.0.1:8766'):
    """生成含首中末真实裁剪及视频跳转的独立身份复核索引。"""
    import base64
    import html
    import cv2
    snapshot=read_json(run_dir/'benchmark_input.json');run_id=snapshot['run_id']
    manifest=read_json(run_dir/'manifest.json')
    rows={(row['track_id'],row['frame_index']):row for row in read_rows(run_dir/'tracks.parquet')}
    capture=cv2.VideoCapture(manifest['config']['video'])
    cards=[]
    try:
        for segment in snapshot['segments']:
            frames=segment['field_frames'];chosen=sorted({frames[0],frames[len(frames)//2],frames[-1]})
            figures=[]
            for frame in chosen:
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read()
                if not ok:raise ValueError(f'无法读取源帧 {frame}')
                row=rows[(segment['track_id'],frame)]
                left,top,right,bottom=[int(row[key]) for key in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
                height=max(1,bottom-top);width=max(1,right-left)
                x0=max(0,left-width);y0=max(0,top-height//3)
                x1=min(image.shape[1],right+width);y1=min(image.shape[0],bottom+height//3)
                crop=image[y0:y1,x0:x1].copy()
                cv2.rectangle(crop,(left-x0,top-y0),(right-x0,bottom-y0),(0,0,255),1)
                _,encoded=cv2.imencode('.jpg',crop)
                source='data:image/jpeg;base64,'+base64.b64encode(encoded).decode()
                url=html.escape(f'{review_base_url.rstrip("/")}/field-review?run={run_id}&frame={frame}', quote=True)
                figures.append(f'<figure><a target="_blank" href="{url}"><img src="{source}" alt="源帧 {frame}"></a><figcaption>源帧 {frame}</figcaption></figure>')
            cards.append(f'<article><h3>{html.escape(segment["record_id"])}</h3><p>{len(frames)} 观测</p><div class="frames">'+''.join(figures)+'</div><p>player_id：待人工填写；内部串号：待检查</p></article>')
    finally:capture.release()
    (run_dir/'identity_review_index.html').write_text('<!doctype html><meta charset="utf-8"><title>身份真值复核索引</title><style>body{font:14px system-ui;max-width:1000px;margin:auto}article{border-bottom:1px solid #ddd;padding:10px}.frames{display:flex;gap:16px}figure{margin:0}img{height:170px;max-width:300px;object-fit:contain}</style><h1>独立人工身份标注</h1><p>首中末裁剪仅作索引，必须播放检查片段内部串号，再填写 identity_truth.json。不要依据算法分数赋予身份；不能确认时保留 null。点击图片跳转所配置的校对服务。</p>'+''.join(cards),encoding='utf-8')


def main():
    """执行独立快照、分析、真值评估或特征消融。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','analyze','evaluate','ablate','gallery'])
    parser.add_argument('run_dir',type=Path);parser.add_argument('--destination',type=Path)
    parser.add_argument('--review-base-url',default='http://127.0.0.1:8766',help='gallery 跳转的校对服务地址')
    args=parser.parse_args();run=args.run_dir.resolve()
    if args.action=='freeze':
        if not args.destination:parser.error('freeze 需要 --destination')
        freeze(run,args.destination.resolve());return
    snapshot=read_json(run/'benchmark_input.json')
    if snapshot['input_signature']!=input_signature(run):raise ValueError('输入摘要已变化，不能继续使用固定基准')
    if args.action=='analyze':
        if load_review(run)['operations']:raise ValueError('独立验收需要没有已有融合的副本')
        write_json(run/'parameters_frozen.json',{'parameters':config_snapshot(),'sha256':digest(config_snapshot()),'status':'provisional-uncalibrated'})
        result=compute_result(run);commit_result(run,result,result['input_signature'])
        print({'automatic_added':result['automatic_added'],'elapsed_s':result['elapsed_s']})
    elif args.action=='evaluate':
        report=evaluate(snapshot,read_json(run/'identity_truth.json'),load_review(run),run)
        write_json(run/'acceptance_report.json',report)
        print({key:value for key,value in report.items() if key not in {'edges','recommendations','contaminated_or_unconfirmed_groups'}})
    elif args.action=='ablate':print(ablate(run))
    else:review_gallery(run,args.review_base_url)


if __name__=='__main__':main()
