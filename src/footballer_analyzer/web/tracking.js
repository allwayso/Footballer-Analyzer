import {$,state,option,notify,request,api,drawVideo,pitch,svg,showFrame,initialize} from './workspace.js';
let models=[],result=null,timer=null,ballRequest=0;
function modelState(){
  const model=models.find((m)=>m.id===$('model').value);
  $('run-ball').disabled=!model?.available;
  $('model-info').textContent=model?.available?'本地模型可用。逐源帧检测并保存独立结果版本。':'缺少本地模型权重，暂不能运行；已有结果仍可查看。';
}
function project(candidate,frame){
  const c=state.data.calibration;
  if(!c?.reviewed||frame<c.start_frame||frame>=c.end_frame)return null;
  const record=c.mode==='white_lines'?c.frames?.[String(frame)]:c;
  if(!record?.homography||(c.mode==='white_lines'&&record.status!=='valid'))return null;
  const h=record.homography,x=(candidate[0]+candidate[2])/2,y=(candidate[1]+candidate[3])/2;
  const d=h[2][0]*x+h[2][1]*y+h[2][2];if(Math.abs(d)<1e-8)return null;
  const p=[(h[0][0]*x+h[0][1]*y+h[0][2])/d,(h[1][0]*x+h[1][1]*y+h[1][2])/d];
  return p.every(Number.isFinite)&&p[0]>=0&&p[0]<=c.pitch_length_m&&p[1]>=0&&p[1]<=c.pitch_width_m?p:null;
}
function draw(){
  drawVideo();const canvas=$('video-canvas'),ctx=canvas.getContext('2d'),record=result?.frames.find((r)=>r.frame===state.frame);
  ctx.lineWidth=2;ctx.strokeStyle='#84AEBF';ctx.setLineDash([5,4]);
  for(const c of record?.candidates||[])ctx.strokeRect(c[0],c[1],c[2]-c[0],c[3]-c[1]);ctx.setLineDash([]);
  const selected=record?.selected;
  if(selected){
    const x=(selected[0]+selected[2])/2,size=Math.max(12,canvas.width/100),tip=selected[1]-Math.max(4,size/3);
    ctx.fillStyle='#36D9AB';ctx.strokeStyle='#0F2940';ctx.lineWidth=Math.max(1.5,size/8);
    ctx.beginPath();ctx.moveTo(x-size/2,tip-size);ctx.lineTo(x+size/2,tip-size);ctx.lineTo(x,tip);ctx.closePath();ctx.fill();ctx.stroke();
  }
  $('frame-info').textContent=!result?'尚无足球结果。选择可用模型后运行追踪。':`${record?.candidates.length||0} 个候选 · ${selected?`选中置信度 ${(selected[4]*100).toFixed(1)}%`:'缺失观测，不插值'}`;
  const node=pitch('ball-pitch');let previous=null,count=0;
  for(const row of result?.frames||[]){
    if(row.frame>state.frame)break;
    const point=row.selected?project(row.selected,row.frame):null;
    if(point){count++;if(previous&&row.frame===previous.frame+1)node.append(svg('line',{x1:previous.point[0],y1:previous.point[1],x2:point[0],y2:point[1],stroke:'#36D9AB','stroke-width':0.45}));if(row.frame===state.frame)node.append(svg('circle',{cx:point[0],cy:point[1],r:1,fill:'#fff'}));}
    previous=point?{frame:row.frame,point}:null;
  }
  $('projection-status').textContent=state.data.calibration?.reviewed?`近似地面投影 · 当前已显示 ${count} 个有效点`:'仅图像坐标 · 请先确认场地标定';
}
function renderResult(){
  const versions=state.data.ball_versions||[];
  $('versions').replaceChildren(option('','最新结果'),...versions.map((v)=>option(v.result_id,`${v.model_id} / ${v.result_id}`)));
  if(result){
    $('versions').value=result.manifest.result_id;
    const m=result.manifest;
    $('ball-summary').textContent=`${m.result_id}\n模型 ${m.model_id} · 阈值 ${m.confidence}\n覆盖 ${m.observed_frames} / ${m.processed_frames} 帧`;
  }else $('ball-summary').textContent='该模型尚无已完成结果，选择可用模型并点击运行。';
  const missing=(result?.frames||[]).filter((r)=>!r.selected);
  $('missing').replaceChildren();
  if(!missing.length)$('missing').textContent=result?'当前版本没有缺失帧。':'完成追踪后显示缺失帧。';
  // A select exposes every gap without creating thousands of DOM buttons.
  if(missing.length){const select=document.createElement('select');select.setAttribute('aria-label','选择缺失帧');select.append(option('',`${missing.length} 个缺失帧`),...missing.map((r)=>option(r.frame,`源帧 ${r.frame}`)));select.onchange=()=>{if(select.value!=='')showFrame(Number(select.value)).catch((e)=>notify(e.message,true));};$('missing').append(select);}
}
async function loadResult(model){
  const token=++ballRequest,id=state.runId;
  const loaded=await request(api('ball')+(model?`?model=${encodeURIComponent(model)}`:''));
  if(token!==ballRequest||id!==state.runId)return;
  result=loaded;renderResult();draw();
}
async function watch(){
  clearTimeout(timer);const id=state.runId;
  try{
    const job=await request('/api/ball-job');if(id!==state.runId)return;
    if(job.run_id===id){
      $('progress').value=job.total_frames?job.processed_frames/job.total_frames:0;
      $('job').textContent=job.status==='failed'?`任务失败：${job.error}`:job.status==='complete'?'追踪完成':`运行中 · ${job.processed_frames||0} / ${job.total_frames||0} 帧`;
      if(job.status==='running'){$('run-ball').disabled=true;timer=setTimeout(watch,1800);return;}
      if(job.status==='complete'){const data=await request(api());if(id!==state.runId)return;state.data=data;await loadResult();}
    }else $('job').textContent=job.status==='running'?'另一素材正在进行足球追踪。':'当前没有运行中的足球任务。';
    modelState();
  }catch(error){notify(error.message,true);modelState();}
}
$('model').onchange=()=>{modelState();loadResult($('model').value).catch((e)=>notify(e.message,true));};
$('versions').onchange=()=>{
  const version=state.data.ball_versions.find((v)=>v.result_id===$('versions').value);
  // The API also accepts a version so historical results remain selectable.
  const id=state.runId,token=++ballRequest;
  request(api('ball')+(version?`?version=${encodeURIComponent(version.result_id)}`:'')).then((data)=>{if(id!==state.runId||token!==ballRequest)return;result=data;renderResult();draw();}).catch((e)=>notify(e.message,true));
};
$('run-ball').onclick=async()=>{try{$('run-ball').disabled=true;await request(api('ball'),{model_id:$('model').value});await watch();}catch(e){notify(e.message,true);modelState();}};
initialize(async()=>{clearTimeout(timer);result=null;++ballRequest;const data=await request('/api/ball-models');models=data.models;$('model').replaceChildren(...models.map((m)=>option(m.id,`${m.label}${m.available?'':'（缺权重）'}`)));$('model').value=state.data.ball_versions?.[0]?.model_id||data.default_model;modelState();await loadResult();await watch();},draw).catch((e)=>notify(e.message,true));
