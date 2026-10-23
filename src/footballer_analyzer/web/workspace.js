export const $ = (id) => document.getElementById(id);
export const state = { data: null, runId: null, frame: 0, image: null, generation: 0, frameRequest: 0, playing: false };
export const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const option = (value, label) => { const node = document.createElement('option'); node.value = value; node.textContent = label; return node; };
export function notify(message, error = false) { $('notice').textContent = message; $('notice').classList.toggle('error', error); }
export async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求失败');
  return result;
}
export const api = (action = '') => `/api/runs/${encodeURIComponent(state.runId)}${action ? '/' + action : ''}`;
export const bindingFor = (row) => state.data.bindings.bindings.find((b) => b.track_id === row.track_id && b.start_frame <= row.frame_index && row.frame_index < b.end_frame);
export const playerName = (id) => state.data.players.find((p) => p.player_id === id)?.display_name || id;
export const colors = { home: '#36D9AB', away: '#84AEBF', unknown: '#fff', cluster_0: '#36D9AB', cluster_1: '#84AEBF' };
export function drawVideo(rows = [], selected = null) {
  const canvas = $('video-canvas'), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  ctx.font = `${Math.max(16, canvas.width / 85)}px sans-serif`;
  for (const row of rows) {
    const binding = bindingFor(row);
    ctx.strokeStyle = colors[binding?.team_id || row.team_id] || '#84AEBF';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = row.track_id === selected ? 5 : 2;
    ctx.strokeRect(row.bbox_x1, row.bbox_y1, row.bbox_x2-row.bbox_x1, row.bbox_y2-row.bbox_y1);
    ctx.fillText(`#${row.track_id}${binding?.player_id ? ' ' + playerName(binding.player_id) : ''}${binding?.action === 'exclude' ? ' 已排除' : ''}`, row.bbox_x1, Math.max(20,row.bbox_y1-7));
  }
}
export function svg(tag, attrs = {}, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key,value));
  if (text !== undefined) node.textContent = text;
  return node;
}
export function pitch(id) {
  const node = $(id), definition = state.data.pitch_definition;
  const {length_m: length, width_m: width} = definition.spec;
  node.replaceChildren(); node.setAttribute('viewBox', `-4 -4 ${length+8} ${width+8}`);
  node.append(svg('rect',{x:0,y:0,width:length,height:width,fill:'#174c43',stroke:'#84aebf','stroke-width':0.35}));
  for (const curve of definition.lines || []) node.append(svg('polyline',{points:curve.points.map((p)=>p.join(',')).join(' '),fill:'none',stroke:'#84aebf','stroke-width':0.3}));
  return node;
}
export function stop() { state.playing = false; $('play').textContent = '▶'; }
let draw = () => {};
export async function showFrame(frame) {
  if (!state.data) return;
  const m = state.data.manifest, run = state.runId, token = ++state.frameRequest;
  const target = Math.max(m.start_frame,Math.min(m.end_frame-1,Math.round(Number(frame))));
  if (!Number.isFinite(target)) return;
  const image = new Image();
  await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('视频帧读取失败，请检查源视频'));image.src=api(`frames/${target}`);});
  if (token !== state.frameRequest || run !== state.runId) return;
  state.frame=target;state.image=image;
  $('frame-slider').value=target;$('frame-number').value=target;
  $('frame-label').textContent=`源帧 ${target} / ${m.end_frame-1} · ${((target-m.start_frame)/m.video.fps).toFixed(2)} s`;
  draw();
}
export async function initialize(onLoad, onDraw) {
  draw = onDraw;
  const params=new URLSearchParams(location.search), clip=params.get('clip');
  if(clip)document.querySelectorAll('[data-page]').forEach((a)=>{if(a.dataset.page!=='materials')a.href=`/${a.dataset.page}?clip=${encodeURIComponent(clip)}`;});
  const key=(value)=>String(value).replace(/\.[^.]+$/,'');
  const runs=(await request('/api/runs')).filter((run)=>run.status==='complete' && (!clip || key(run.clip_id)===key(clip)));
  $('run-select').replaceChildren(...runs.map((run)=>option(run.run_id,`${run.clip_id} / ${run.run_id}`)));
  async function load(id) {
    stop();const token=++state.generation;++state.frameRequest;state.image=null;state.data=null;$('work-area').hidden=true;
    state.runId=id;
    try {
      const data=await request(api());if(token!==state.generation)return;
      state.data=data;const m=data.manifest;state.frame=m.start_frame;
      const canvas=$('video-canvas');canvas.width=m.video.width;canvas.height=m.video.height;
      canvas.parentElement.style.aspectRatio=`${m.video.width} / ${m.video.height}`;
      for (const element of [$('frame-slider'),$('frame-number')]) {element.min=m.start_frame;element.max=m.end_frame-1;element.value=m.start_frame;}
      const url=new URL(location.href);url.searchParams.set('clip',key(m.config.clip_id));url.searchParams.set('run',id);history.replaceState(null,'',url);
      document.querySelectorAll('[data-page]').forEach((a)=>{if(a.dataset.page!=='materials')a.href=`/${a.dataset.page}?clip=${encodeURIComponent(key(m.config.clip_id))}`;});
      $('material-meta').textContent=`素材 ${m.config.clip_id} · 比赛 ${m.config.match_id} · ${id} · ${data.calibration?.reviewed ? '标定已确认' : '标定未确认'}`;
      $('work-area').hidden=false;notify('已载入对应素材的完成运行。');
      await onLoad();if(token!==state.generation)return;await showFrame(state.frame);
    } catch(error) { if(token===state.generation)notify(error.message,true); }
  }
  $('run-select').onchange=()=>load($('run-select').value);
  $('previous').onclick=()=>{stop();showFrame(state.frame-1).catch((e)=>notify(e.message,true));};
  $('next').onclick=()=>{stop();showFrame(state.frame+1).catch((e)=>notify(e.message,true));};
  $('jump').onclick=()=>{stop();showFrame($('frame-number').value).catch((e)=>notify(e.message,true));};
  $('frame-slider').oninput=()=>{stop();showFrame($('frame-slider').value).catch((e)=>notify(e.message,true));};
  $('play').onclick=async()=>{
    if(state.playing)return stop();if(!state.data)return;
    state.playing=true;$('play').textContent='Ⅱ';const generation=state.generation;
    while(state.playing && generation===state.generation){
      const m=state.data.manifest,step=Math.max(1,Math.round(m.video.fps/10));
      if(state.frame+step>=m.end_frame){stop();break;}
      try{await showFrame(state.frame+step);}catch(e){notify(e.message,true);stop();}
      await new Promise((resolve)=>setTimeout(resolve,100));
    }
  };
  if(!runs.length){$('empty').hidden=false;$('run-select').disabled=true;notify(clip ? `素材 ${clip} 没有已完成运行。` : '暂无已完成运行。');return;}
  const requested=params.get('run');await load(runs.some((r)=>r.run_id===requested)?requested:runs[0].run_id);
}
