import {$,state,escape,option,notify,request,api,bindingFor,playerName,drawVideo,showFrame,initialize} from './workspace.js';
let selected=null;
function draw(){
  const rows=state.data.tracks.filter((r)=>r.frame_index===state.frame);
  drawVideo(rows,selected);
  $('frame-info').textContent=`当前帧 ${rows.length} 条人物轨迹 · 选中 ${selected===null?'无':`#${selected}`} · 点击画面中的人物框可选择轨迹`;
}
function choose(id,jump=false){
  selected=Number(id);$('track').value=selected;
  const record=state.data.catalog.find((r)=>r.track_id===selected);if(!record)return;
  $('start').value=record.start_frame;$('end').value=record.end_frame;
  document.querySelectorAll('[data-track]').forEach((button)=>button.classList.toggle('selected',Number(button.dataset.track)===selected));
  if(jump)showFrame(record.start_frame).catch((e)=>notify(e.message,true));else draw();
}
function render(){
  $('track').replaceChildren(...state.data.catalog.map((r)=>option(r.track_id,`#${r.track_id} · ${r.observations} 次观测`)));
  $('player').replaceChildren(option('','选择球员档案'),...state.data.players.map((p)=>option(p.player_id,`${p.display_name} / ${p.player_id}`)));
  $('track-list').innerHTML=state.data.catalog.map((r)=>`<button data-track="${r.track_id}" title="跳转轨迹首帧">#${r.track_id}</button>`).join('');
  $('track-list').querySelectorAll('button').forEach((b)=>b.onclick=()=>choose(b.dataset.track,true));
  $('history').innerHTML=state.data.bindings.bindings.map((b,i)=>`<tr><td>#${b.track_id}</td><td>[${b.start_frame}, ${b.end_frame})</td><td>${escape(b.player_id?playerName(b.player_id):'—')}</td><td>${b.action==='exclude'?'已排除':({home:'主队',away:'客队',unknown:'未知'}[b.team_id])}</td><td><button data-remove="${i}">移除</button></td></tr>`).join('') || '<tr><td colspan="5">尚无绑定。先创建球员档案，再选择轨迹与区间。</td></tr>';
  $('history').querySelectorAll('[data-remove]').forEach((button)=>button.onclick=()=>save(state.data.bindings.bindings.filter((_,i)=>i!==Number(button.dataset.remove))).catch((e)=>notify(e.message,true)));
  const hasTracks=state.data.catalog.length>0;for(const id of ['track','player','exclude'])$(id).disabled=!hasTracks;
  if(hasTracks)choose(state.data.catalog.some((r)=>r.track_id===selected)?selected:state.data.catalog[0].track_id);
}
async function save(bindings){
  const id=state.runId;
  await request(api('bindings'),{run_id:id,bindings});if(id!==state.runId)return;
  state.data.bindings.bindings=bindings;render();draw();notify('身份区间已保存。已有报告需要重新生成以更新统计。');
}
async function add(action){
  if(!state.data||selected===null)throw new Error('请先选择人物轨迹');
  const entry={track_id:selected,start_frame:Number($('start').value),end_frame:Number($('end').value),action};
  if(action==='bind'){entry.player_id=$('player').value;entry.team_id=$('team').value;}
  await save([...state.data.bindings.bindings,entry]);
}
$('track').onchange=()=>choose($('track').value,true);
$('binding-form').onsubmit=(e)=>{e.preventDefault();add('bind').catch((err)=>notify(err.message,true));};
$('exclude').onclick=()=>add('exclude').catch((e)=>notify(e.message,true));
$('player-form').onsubmit=async(e)=>{
  e.preventDefault();if(!state.data)return;const id=state.runId;
  const player={player_id:$('player-id').value.trim(),display_name:$('player-name').value.trim()};
  try{await request(api('players'),player);if(id!==state.runId)return;state.data.players=state.data.players.filter((p)=>p.player_id!==player.player_id).concat(player);render();$('player').value=player.player_id;notify('球员档案已保存。');}catch(error){notify(error.message,true);}
};
$('video-canvas').onclick=(e)=>{
  if(!state.data)return;const rect=e.target.getBoundingClientRect(),x=(e.clientX-rect.left)*e.target.width/rect.width,y=(e.clientY-rect.top)*e.target.height/rect.height;
  const rows=state.data.tracks.filter((r)=>r.frame_index===state.frame&&r.bbox_x1<=x&&x<=r.bbox_x2&&r.bbox_y1<=y&&y<=r.bbox_y2).sort((a,b)=>(a.bbox_x2-a.bbox_x1)*(a.bbox_y2-a.bbox_y1)-(b.bbox_x2-b.bbox_x1)*(b.bbox_y2-b.bbox_y1));
  if(rows.length)choose(rows[0].track_id);
};
initialize(()=>{selected=null;render();},draw).catch((e)=>notify(e.message,true));
