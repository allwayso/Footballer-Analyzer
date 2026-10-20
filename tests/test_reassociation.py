"""验证双向球场运动、邻近参照、竞争及可撤销分析发布。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from footballer_analyzer.calibration import load_calibration
from footballer_analyzer.config import config_from_dict
from footballer_analyzer.field_review import field_segments, merge_groups
from footballer_analyzer.reassociation import (AssociationConfig, analyze_nodes, association_view, commit_result, compute_result,
    input_signature, load_review, pair_key, raw_nodes, rank_candidates, group_conflict, config_snapshot, visible_candidates, group_blocked)
from footballer_analyzer.association_evaluation import evaluate
from footballer_analyzer.server import ReviewApp
from footballer_analyzer.stream_association import StreamingAssociator
from footballer_analyzer.storage import TRACK_SCHEMA, RowWriter, file_hash, read_json, read_rows, write_json, write_yaml


def observation(track, frame, x=50, y=70):
    """生成带真实时间及人物框的源观测。"""
    return {"track_id":track, "frame_index":frame, "timestamp_s":frame/25, "bbox_x1":x-8, "bbox_y1":y-30,
            "bbox_x2":x+8, "bbox_y2":y, "confidence":.9, "class_id":0, "team_id":"cluster_0"}


def prepare_run(root):
    """创建可解码三片段素材；前两段可自动关联，第三段仅推荐。"""
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    video=root/'source.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(192,128))
    rows=[observation(1,f) for f in range(10)]+[observation(2,f) for f in range(13,30)]+[observation(3,f,60) for f in range(40,55)]
    for f in range(55):
        image=np.full((128,192,3),(40,100,40),np.uint8)
        for row in rows:
            if row['frame_index']==f:
                cv2.rectangle(image,(int(row['bbox_x1']),40),(int(row['bbox_x2']),70),(20,20,230),-1)
        writer.write(image)
    writer.release()
    players=root/'players.yaml';write_yaml(players,{'players':[]})
    config=config_from_dict({'video':str(video.resolve()),'players':str(players.resolve()),'weights':str(root/'unused.pt'),'output_root':str(root), 'match_id':'synthetic','clip_id':'synthetic'})
    run=root/'run_test';run.mkdir()
    manifest={'run_id':'run_test','status':'complete','config':config.snapshot(),'video':{'fps':25,'width':192,'height':128,'frame_count':55},'video_sha256':file_hash(video),'start_frame':0,'end_frame':55}
    write_json(run/'manifest.json',manifest);write_json(run/'catalog.json',[{'track_id':t} for t in [1,2,3]])
    with RowWriter(run/'tracks.parquet',TRACK_SCHEMA) as output:
        for row in rows: output.append(row)
    write_yaml(run/'calibration.yaml',{'reviewed':True,'mode':'static','video_sha256':manifest['video_sha256'],'width':192,'height':128,'start_frame':0,'end_frame':55,
        'pitch_length_m':105,'pitch_width_m':68,'homography':[[.1,0,0],[0,.1,0],[0,0,1]],'roi':[[0,0],[191,0],[191,127],[0,127]]})
    return run,manifest


class ReassociationTests(unittest.TestCase):
    """验证算法拒绝条件与复核历史，不将评分当真值。"""

    def test_direction_beats_nearest_static_position(self):
        """同队交叉中，双向运动应选择沿原方向移动的对象。"""
        manifest={'video':{'fps':25},'config':{'stride':1}}
        rows=[observation(1,f,50+f) for f in range(20)]
        rows += [observation(2,f,50+f) for f in range(25,45)]
        rows += [observation(3,f,75-(f-25)) for f in range(25,45)]
        with patch('footballer_analyzer.reassociation.project_observation',side_effect=lambda row,cal:((row['bbox_x1']+row['bbox_x2'])/20,row['bbox_y2']/10,None)):
            candidates=analyze_nodes(raw_nodes(rows,manifest),manifest,{'reviewed':True})
        best=max(candidates,key=lambda item:item['score'])
        self.assertEqual(best['source_record_id'],'track-2-frame-25')
        self.assertTrue(best['automatic'])
        self.assertFalse(any(item['automatic'] for item in candidates if item['source_record_id']=='track-3-frame-25'))

    def test_short_turn_and_competition_remain_manual(self):
        """短片段与两个同分候选均不能自动合组，最高阈值等号也只推荐。"""
        config=AssociationConfig()
        def candidate(source,score):
            """构造共享前驱的竞争项。"""
            return {'source_record_id':source,'target_record_id':'a','score':score,'auto_eligible':True}
        candidates=rank_candidates([candidate('b',.99),candidate('c',.99)],config)
        self.assertFalse(any(item['automatic'] for item in candidates))
        self.assertFalse(rank_candidates([candidate('b',config.high_threshold)],config)[0]['automatic'])
        manifest={'video':{'fps':25},'config':{'stride':1}}
        rows=[observation(1,f) for f in range(20)]+[observation(2,25)]
        with patch('footballer_analyzer.reassociation.project_observation',return_value=(5,7,None)):
            result=analyze_nodes(raw_nodes(rows,manifest),manifest,{'reviewed':True})
        self.assertTrue(result)
        self.assertFalse(result[0]['automatic'])

    def test_relative_references_remove_pan_and_zoom_and_reject_cut(self):
        """共同参照可处理平移缩放，参照布局突变时不自动融合。"""
        manifest={'video':{'fps':25},'config':{'stride':1}}
        rows=[observation(1,f,50+f) for f in range(20)]+[observation(2,f,50+f) for f in range(25,45)]
        for track,x in [(10,20),(11,90),(12,130),(13,170)]:
            rows += [observation(track,f,x+f) for f in range(45)]
        evidence={f"{r['track_id']}:{r['frame_index']}":[1,0,1,0] for r in rows}
        candidates=analyze_nodes(raw_nodes(rows,manifest),manifest,None,evidence)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]['evidence']['coordinate_mode'],'relative')
        self.assertTrue(candidates[0]['automatic'])
        for row in rows:
            scale=1+row['frame_index']*.002
            for key in ['bbox_x1','bbox_x2','bbox_y1','bbox_y2']: row[key]*=scale
        zoom=analyze_nodes(raw_nodes(rows,manifest),manifest,None,evidence)
        self.assertTrue(zoom[0]['automatic'])
        for row in rows:
            if row['track_id']==12 and row['frame_index']>=25:
                row['bbox_x1']+=200;row['bbox_x2']+=200
            if row['track_id']==13 and row['frame_index']>=25:
                row['bbox_x1']-=200;row['bbox_x2']-=200
        self.assertFalse(any(item['automatic'] for item in analyze_nodes(raw_nodes(rows,manifest),manifest,None,evidence)))

    def test_dynamic_projection_gap_and_internal_jump_disable_auto(self):
        """动态映射中间失效或端点跳变都不能自动跨越。"""
        manifest={'video':{'fps':25},'config':{'stride':1}}
        rows=[observation(1,f) for f in range(20)]+[observation(2,f) for f in range(25,45)]
        for track,x in [(10,20),(11,90),(12,130)]: rows += [observation(track,f,x) for f in range(45)]
        frames={str(f):{'status':'valid','homography':np.eye(3).tolist()} for f in range(45)}
        calibration={'reviewed':True,'mode':'white_lines','frames':frames}
        with patch('footballer_analyzer.reassociation.project_observation',side_effect=lambda row,cal:((row['bbox_x1']+row['bbox_x2'])/20,7,None)):
            self.assertTrue(analyze_nodes(raw_nodes(rows,manifest),manifest,calibration)[0]['automatic'])
            frames['22']['status']='failed'
            self.assertFalse(any(item['automatic'] for item in analyze_nodes(raw_nodes(rows,manifest),manifest,calibration)))
        with patch('footballer_analyzer.reassociation.project_observation',side_effect=lambda row,cal:(5+(10 if row['frame_index']==5 else 0),7,None)):
            result=analyze_nodes(raw_nodes(rows,manifest),manifest,{'reviewed':True})
            self.assertFalse(any(item['automatic'] for item in result))

    def test_streaming_waits_for_closed_predecessor_and_matches_replay(self):
        """在线等待前驱真正关闭；相同真实观测的融合必须能被离线复算确认。"""
        manifest={'start_frame':0,'video':{'fps':25},'config':{'stride':1,'lost_track_buffer_s':3}}
        rows=[observation(1,f,50+f*.2) for f in range(20)]+[observation(2,f,50+f*.2) for f in range(25,45)]
        for track,x in [(10,20),(11,90),(12,130)]:
            rows += [observation(track,f,x+f*.2) for f in range(151)]
        evidence={f"{row['track_id']}:{row['frame_index']}":[1,0,1,0] for row in rows}
        stream=StreamingAssociator(manifest)
        for frame in range(151):
            stream.observe(frame,[row for row in rows if row['frame_index']==frame],evidence)
            if frame<=95:self.assertFalse(stream.operations)
        self.assertEqual(len(stream.operations),1)
        replay=analyze_nodes(raw_nodes(rows,manifest),manifest,None,evidence)
        automatic={pair_key(item['source_record_id'],item['target_record_id']) for item in replay if item['automatic']}
        self.assertIn(pair_key(*stream.operations[0]['node_pair']),automatic)
        self.assertLessEqual(len(stream.recent),25*15*5)

    def test_whole_group_conflict(self):
        """两个端点不重叠也必须检查整组里其他片段。"""
        segments=[{'record_id':'a','merge_group_id':'a','start_frame':0,'end_frame':9},
                  {'record_id':'b','merge_group_id':'a','start_frame':20,'end_frame':29},
                  {'record_id':'c','merge_group_id':'c','start_frame':25,'end_frame':35}]
        self.assertTrue(group_conflict(segments,'a','c'))

    def test_mapping_checks_all_original_intervals_and_indirect_blocks(self):
        """候选尚未访问的组成员也应以原始时间检查，阻止边不能绕行。"""
        segments=[{'record_id':'a','track_id':1,'merge_group_id':'a','start_frame':0,'end_frame':9,'field_frames':[0,9]},
                  {'record_id':'b','track_id':2,'merge_group_id':'a','start_frame':20,'end_frame':29,'field_frames':[20,29]},
                  {'record_id':'c','track_id':3,'merge_group_id':'c','start_frame':40,'end_frame':49,'field_frames':[40,49]}]
        result={'nodes':[{'record_id':key,'track_id':track,'start_frame':start,'end_frame':end}
                         for key,track,start,end in [('a',1,0,9),('b',2,20,45),('c',3,40,49)]],
                'candidates':[{'source_record_id':'c','target_record_id':'a','source_frame':40,'target_frame':9,'score':.99}]}
        self.assertFalse(visible_candidates(result,segments,{'blocked_pairs':[]}))
        self.assertTrue(group_blocked(segments,'a','c',[pair_key('b','c')]))
        result['nodes'][1]['end_frame']=29
        self.assertFalse(visible_candidates(result,segments,{'blocked_pairs':[pair_key('b','c')]}))
        self.assertEqual(len(visible_candidates(result,segments,{'blocked_pairs':[]})),1)

    def test_publish_undo_block_unblock_and_stale(self):
        """反复分析幂等；撤销阻止重合；输入过期只暂停自动项。"""
        with tempfile.TemporaryDirectory() as root:
            run,manifest=prepare_run(root);original=file_hash(run/'tracks.parquet')
            result=compute_result(run);commit_result(run,result,result['input_signature'])
            review=load_review(run)
            self.assertEqual(len(review['operations']),1)
            app=ReviewApp(root)
            data=app.data('run_test');self.assertEqual(data['association']['summary']['groups'],2)
            self.assertEqual(data['field_review_history'][0]['execution'],'auto')
            self.assertIn('track-3-frame-40',data['association']['recommendations'])
            commit_result(run,compute_result(run),result['input_signature'])
            self.assertEqual(len(load_review(run)['operations']),1)
            app.undo_field_merge('run_test',{'operation_id':review['operations'][0]['operation_id']})
            commit_result(run,compute_result(run),result['input_signature'])
            self.assertEqual(len(load_review(run)['operations']),1)
            app.unblock_association('run_test',{'pair':pair_key(*review['operations'][0]['node_pair'])})
            commit_result(run,compute_result(run),result['input_signature'])
            self.assertEqual(len(load_review(run)['operations']),2)
            app.confirm_association('run_test',{'record_id':'track-3-frame-40','batch_id':load_review(run)['analysis']['batch_id']})
            cal=load_calibration(run);cal['note']='changed';write_yaml(run/'calibration.yaml',cal)
            data=app.data('run_test')
            self.assertEqual(data['association']['status'],'stale')
            self.assertEqual(data['field_review_history'][-1]['execution'],'manual')
            self.assertEqual(data['field_review_history'][-1]['status'],'active')
            self.assertEqual(data['field_review_history'][1]['status'],'stale')
            self.assertEqual(file_hash(run/'tracks.parquet'),original)
            with self.assertRaisesRegex(ValueError,'分析期间'):
                commit_result(run,result,result['input_signature'])

    def test_failed_job_retry_restart_and_reentry(self):
        """重入复用任务，重启恢复为可重试，失败保留已有结果。"""
        with tempfile.TemporaryDirectory() as root:
            run,_=prepare_run(root);app=ReviewApp(root)
            write_json(run/'association_job.json',{'status':'running'})
            self.assertEqual(app.association_job('run_test')['status'],'failed')
            entered,release=threading.Event(),threading.Event()
            def failing(*args,**kwargs):
                """模拟耗时计算后失败。"""
                entered.set();release.wait(5);raise ValueError('test failure')
            with patch('footballer_analyzer.server.compute_result',side_effect=failing):
                job=app.start_association('run_test',{})
                self.assertTrue(entered.wait(5))
                self.assertEqual(app.start_association('run_test',{})['job_id'],job['job_id'])
                release.set()
                for worker in threading.enumerate():
                    if worker.name.endswith('(worker)'):worker.join(5)
            self.assertEqual(app.association_job('run_test')['status'],'failed')
            self.assertFalse(load_review(run)['operations'])

    def test_evaluation_counts_unknown_and_prevents_denominator_changes(self):
        """未知不能充当正确边，真值不能删片段缩小分母。"""
        with tempfile.TemporaryDirectory() as root:
            run,manifest=prepare_run(root);result=compute_result(run);commit_result(run,result,result['input_signature'])
            segments=field_segments(list(read_rows(run/'tracks.parquet')),manifest,load_calibration(run))
            snapshot={'input_signature':result['input_signature'],'segments':segments}
            labels={item['record_id']:{'player_id':None,'internally_consistent':None} for item in segments}
            truth={'input_signature':result['input_signature'],'records':labels}
            report=evaluate(snapshot,truth,load_review(run),run)
            self.assertFalse(report['hard_metrics_passed']);self.assertEqual(report['edge_precision'],0)
            for label in labels.values():label.update(player_id='p1',internally_consistent=True)
            report=evaluate(snapshot,truth,load_review(run),run)
            self.assertEqual(report['edge_precision'],1);self.assertEqual(report['recommendation_precision'],1)
            self.assertFalse(report['independent_truth'])
            self.assertFalse(report['recommendations_passed'])
            truth.update(annotator='独立复核员',independent=True)
            self.assertTrue(evaluate(snapshot,truth,load_review(run),run)['recommendations_passed'])
            labels.pop(next(iter(labels)))
            with self.assertRaises(ValueError):evaluate(snapshot,truth,load_review(run),run)


if __name__=='__main__':unittest.main()
