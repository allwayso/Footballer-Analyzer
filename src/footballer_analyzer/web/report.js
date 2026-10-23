import {$,state,escape,option,notify,request,api,playerName,drawVideo,pitch,svg,showFrame,initialize} from './workspace.js';
let selected='',speedRange=null;
const number=(v,factor=1)=>v===null||v===undefined?'—':(v*factor).toFixed(2);
const report=()=>state.data.latest_report;
const motion=()=>report()?.motion.filter((r)=>r.player_id===selected)||[];
function draw(){
  const r=report();if(!r)return;
  const ids=new Set(r.motion.filter((row)=>row.frame_index===state.frame&&row.player_id===selected).map((row)=>row.track_id));
  drawVideo(state.data.tracks.filter((row)=>row.frame_index===state.frame&&ids.has(row.track_id)));
  const node=pitch('replay-pitch');let previous=null;
  for(const row of motion()){
    if(row.frame_index>state.frame)continue;
    if(!row.valid){previous=null;continue;}
    if(previous&&previous.segment_id===row.segment_id)node.append(svg('line',{x1:previous.x_smooth_m,y1:previous.y_smooth_m,x2:row.x_smooth_m,y2:row.y_smooth_m,stroke:'#2da683','stroke-width':0.4}));
    if(row.frame_index===state.frame)node.append(svg('circle',{cx:row.x_smooth_m,cy:row.y_smooth_m,r:1.2,fill:'#36d9ab',stroke:'#fff','stroke-width':0.3}));previous=row;
  }
  $('frame-info').textContent=`${playerName(selected)} · 源帧 ${state.frame} · ${ids.size?'当前帧有该球员观测':'当前帧没有该球员观测，保留缺口'}`;
  drawSpeed();
}
function drawHeat(){
  const node=pitch('heatmap'),{length_m:l,width_m:w}=state.data.pitch_definition.spec;
  const bins=new Map();
  for(const row of motion()){
    if(!row.valid||row.delta_t_s===null)continue;
    const x=Math.min(34,Math.max(0,Math.floor(row.x_smooth_m/l*35))),y=Math.min(22,Math.max(0,Math.floor(row.y_smooth_m/w*23))),key=`${x},${y}`;
    bins.set(key,(bins.get(key)||0)+row.delta_t_s);
  }
  const max=Math.max(...bins.values(),0.0001);
  for(const [key,time] of bins){const [x,y]=key.split(',').map(Number);node.append(svg('rect',{x:x*l/35,y:y*w/23,width:l/35,height:w/23,fill:'#36d9ab',opacity:0.12+0.75*time/max}));}
  const row=report().summary.players.find((p)=>p.player_id===selected);
  $('metrics').textContent=row?`${playerName(selected)}　距离 ${number(row.distance_m)} m　平均速度 ${number(row.avg_speed_mps,3.6)} km/h　有效 ${number(row.valid_seconds)} s　覆盖 ${number(row.coverage,100)}%`:'没有有效统计';
}
function drawSpeed(){
  const node=$('speed');node.replaceChildren();
  const m=state.data.manifest,rows=motion(),start=m.start_frame/m.video.fps,end=m.end_frame/m.video.fps;
  const max=Math.max(10,...rows.filter((r)=>r.valid&&r.speed_mps!==null).map((r)=>r.speed_mps*3.6));
  const x=(time)=>45+(time-start)/(end-start)*570,y=(speed)=>220-speed/max*185;
  speedRange={start,end};
  for(let i=0;i<=4;i++){const speed=i*max/4;node.append(svg('line',{x1:45,y1:y(speed),x2:615,y2:y(speed),stroke:'#dbe5e8','stroke-width':1}),svg('text',{x:38,y:y(speed)+4,'text-anchor':'end',fill:'#627e89','font-size':11},speed.toFixed(0)));}
  for(let i=0;i<=4;i++){const time=start+(end-start)*i/4;node.append(svg('text',{x:x(time),y:245,'text-anchor':'middle',fill:'#627e89','font-size':11},time.toFixed(1)));}
  let previous=null;
  for(const row of rows){if(!row.valid||row.speed_mps===null){previous=null;continue;}if(previous&&previous.segment_id===row.segment_id)node.append(svg('line',{x1:x(previous.timestamp_s),y1:y(previous.speed_mps*3.6),x2:x(row.timestamp_s),y2:y(row.speed_mps*3.6),stroke:'#2da683','stroke-width':2}));previous=row;}
  const current=x(state.frame/m.video.fps);node.append(svg('line',{x1:current,y1:25,x2:current,y2:220,stroke:'#0f2940','stroke-width':1.5,'stroke-dasharray':'4 4'}));
}
function table(){
  const field=$('sort').value,rows=[...report().summary.players].sort((a,b)=>(b[field]??-1)-(a[field]??-1));
  $('comparison').innerHTML=rows.map((r)=>`<tr class="${r.player_id===selected?'comparison-selected':''}"><td><button data-player="${escape(r.player_id)}">${escape(playerName(r.player_id))}</button></td><td>${({home:'主队',away:'客队',unknown:'未知'}[r.team_id])||'未知'}</td><td>${number(r.distance_m)}</td><td>${number(r.avg_speed_mps,3.6)}</td><td>${number(r.valid_seconds)}</td><td>${number(r.coverage,100)}%</td></tr>`).join('');
  $('comparison').querySelectorAll('[data-player]').forEach((b)=>b.onclick=()=>select(b.dataset.player));
}
function select(id){
  selected=id;$('report-player').value=id;drawHeat();draw();table();
  const hex=Array.from(new TextEncoder().encode(id),(b)=>b.toString(16).padStart(2,'0')).join('');
  $('png').href=`/reports/${encodeURIComponent(state.runId)}/${encodeURIComponent(report().report_id)}/${hex}.png`;$('png').download=`${id}.png`;
}
function render(){
  const r=report(),ready=state.data.calibration?.reviewed&&state.data.bindings.bindings.some((b)=>b.action==='bind');
  $('generate').disabled=!ready;$('generate').textContent=r?'重算报告':'生成报告';
  $('report-version').textContent=r?`报告版本 ${r.report_id}`:'这条素材尚未生成报告';
  const messages=[];
  if(!ready)messages.push('先确认场地标定并绑定至少一名球员，再生成报告。');
  if(r?.stale)messages.push('报告已过期：球员档案、身份绑定或场地标定发生变化，请重新生成。');
  $('report-status').textContent=messages.join(' ')||(r?'报告数据与当前绑定及标定一致。':'已满足生成条件，点击生成报告开始。');
  $('report-content').hidden=!r;
  for(const id of ['csv','json'])$(id).disabled=!r;
  for(const id of ['png','html','review'])$(id).hidden=!r;
  if(!r)return;
  const base=`/reports/${encodeURIComponent(state.runId)}/${encodeURIComponent(r.report_id)}`;
  $('html').href=`${base}/report.html`;$('review').href=`${base}/review.mp4`;
  $('report-player').replaceChildren(...r.summary.players.map((p)=>option(p.player_id,playerName(p.player_id))));
  $('quality').textContent='观测质量：'+Object.entries(r.summary.quality).map(([k,v])=>`${k} ${v}`).join('，');
  if(r.summary.players.length)select(r.summary.players.some((p)=>p.player_id===selected)?selected:r.summary.players[0].player_id);
}
function download(data,type,suffix){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=`${report().report_id}.${suffix}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('csv').onclick=()=>{const rows=report().summary.players;if(!rows.length)return;const keys=Object.keys(rows[0]);const cell=(v)=>`"${String(v??'').replace(/"/g,'""')}"`;download('\ufeff'+[keys,...rows.map((r)=>keys.map((k)=>r[k]))].map((r)=>r.map(cell).join(',')).join('\r\n'),'text/csv;charset=utf-8','csv');};
$('json').onclick=()=>download(JSON.stringify(report(),null,2),'application/json','json');
$('generate').onclick=async()=>{
  const id=state.runId;$('generate').disabled=true;notify('正在生成报告与叠加视频，请等待。');
  try{await request(api('report'),{});if(id!==state.runId)return;const data=await request(api());if(id!==state.runId)return;state.data=data;render();notify('报告已生成。');}catch(e){if(id===state.runId){notify(e.message,true);render();}}
};
$('report-player').onchange=()=>select($('report-player').value);$('sort').onchange=table;
$('speed').onclick=(e)=>{if(!speedRange)return;const box=e.currentTarget.getBoundingClientRect(),x=(e.clientX-box.left)*640/box.width;const time=speedRange.start+Math.max(0,Math.min(1,(x-45)/570))*(speedRange.end-speedRange.start);showFrame(Math.round(time*state.data.manifest.video.fps)).catch((error)=>notify(error.message,true));};
$('speed').onkeydown=(e)=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();showFrame(state.frame+(e.key==='ArrowRight'?1:-1)).catch((error)=>notify(error.message,true));}};
initialize(render,draw).catch((e)=>notify(e.message,true));
