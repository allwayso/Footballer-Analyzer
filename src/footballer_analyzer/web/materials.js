const $ = (selector) => document.querySelector(selector);
const state = { clips: [], page: 1, pageSize: 8, query: '', sort: 'id', generation: 0 };
let toastTimer;
function notify(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3200);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
function duration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const total = Math.round(seconds);
  return [Math.floor(total / 3600), Math.floor(total / 60) % 60, total % 60].map((part) => String(part).padStart(2, '0')).join(':');
}
function bytes(value) {
  return value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GB` : `${(value / 1024 ** 2).toFixed(1)} MB`;
}
const entryIcon = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M11 4h5v5M16 4l-8 8M8 4H4v12h12v-4"/></svg>';
function entryButton(label, id, clip) {
  const clipKey = clip.clip_id.replace(/\.[^.]+$/, '');
  const routes = { '场地标定': '/calibration', '人物识别': '/identity', '足球追踪': '/tracking', '分析报告': '/report' };
  const completed = label === '场地标定' ? clip.calibrated_runs > 0 : label === '人物识别' ? clip.tracking_runs > 0 : label === '足球追踪' ? clip.ball_runs > 0 : clip.report_runs > 0;
  const status = label === '场地标定' ? (completed ? '已标定' : '未标定') : label === '人物识别' ? (completed ? '已识别' : '未识别') : label === '足球追踪' ? (completed ? '已追踪' : '未追踪') : (completed ? '查看报告' : '生成报告');
  const kind = completed ? 'complete' : 'pending';
  return `<a class="entry-button ${label === '场地标定' ? `calibration-${kind}` : `workflow-${kind}`}" href="${routes[label]}?clip=${encodeURIComponent(clipKey)}" title="${status}，打开${label}" aria-label="${id} ${status}，打开${label}">${entryIcon}${status}</a>`;
}
function render() {
  const filtered = state.clips.filter((clip) => `${clip.id} ${clip.material_id} ${clip.clip_id}`.toLowerCase().includes(state.query));
  filtered.sort((a, b) => {
    if (state.sort === 'id') return a.id.localeCompare(b.id);
    if (!Number.isFinite(a.duration_s)) return Number.isFinite(b.duration_s) ? 1 : a.id.localeCompare(b.id);
    if (!Number.isFinite(b.duration_s)) return -1;
    return (state.sort === 'duration-desc' ? -1 : 1) * (a.duration_s - b.duration_s) || a.id.localeCompare(b.id);
  });
  const pages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
  state.page = Math.min(state.page, pages);
  const start = (state.page - 1) * state.pageSize;
  $('#material-rows').innerHTML = filtered.slice(start, start + state.pageSize).map((clip) => {
    const name = escapeHtml(clip.clip_id);
    return `<tr><td><div class="thumbnail"><img loading="lazy" src="/api/clips/${encodeURIComponent(clip.clip_id)}/preview" alt="${name} 的首帧截图"><span class="frame-label">首帧</span></div></td><td><button class="material-id" data-copy-id="M-${clip.material_id.toUpperCase()}" title="点击复制完整素材编号" aria-label="复制完整素材编号 M-${clip.material_id.toUpperCase()}">${clip.id}</button><span class="file-name" title="${name}">${name}</span></td><td><span class="missing">待补充</span></td><td><span class="missing">待补充</span></td><td><span class="missing">待补充</span></td><td><span class="missing">—</span></td><td><span class="time-value" title="${clip.durationError ? '无法读取视频时长' : '源视频时长'}">${duration(clip.duration_s)}</span></td>${['场地标定', '人物识别', '足球追踪', '分析报告'].map((label) => `<td>${entryButton(label, clip.id, clip)}</td>`).join('')}<td><button class="more-button" data-placeholder="素材操作" data-material="${clip.id}" aria-label="${clip.id} 更多操作">···</button></td></tr>`;
  }).join('');
  $('#material-rows').querySelectorAll('img').forEach((img) => img.addEventListener('error', () => {
    const fallback = document.createElement('span');
    fallback.textContent = '首帧暂不可用';
    img.parentElement.replaceChildren(fallback);
  }));
  $('#load-message').hidden = filtered.length > 0;
  if (!filtered.length) {
    $('#load-message').classList.remove('error');
    $('#load-message').textContent = state.query ? '没有匹配的素材，请尝试其他关键词。' : '素材库为空，将视频放入 data/clips 后刷新列表。';
  }
  $('#range-summary').textContent = filtered.length ? `显示 ${start + 1}–${Math.min(start + state.pageSize, filtered.length)} 条，共 ${filtered.length} 条素材` : '共 0 条素材';
  $('#page-number').textContent = `${state.page} / ${pages}`;
  $('#previous').disabled = state.page <= 1;
  $('#next').disabled = state.page >= pages;
}
async function load() {
  const generation = ++state.generation;
  $('#refresh').disabled = true;
  $('#load-message').hidden = false;
  $('#load-message').classList.remove('error');
  $('#load-message').textContent = '正在读取本地视频…';
  try {
    const response = await fetch('/api/clips');
    if (!response.ok) throw new Error('素材列表读取失败');
    const clips = await response.json();
    state.clips = clips.map((clip) => ({ ...clip, id: `M-${clip.material_id.slice(0, 12).toUpperCase()}`, duration_s: null }));
    state.page = 1;
    $('#total-count').textContent = state.clips.length;
    $('#nav-count').textContent = state.clips.length;
    $('#storage-summary').textContent = `${state.clips.length} 段视频，占用 ${bytes(state.clips.reduce((total, clip) => total + clip.size_bytes, 0))}`;
    render();
    // Limit concurrent video probes, so the first frames and UI remain responsive.
    const queue = [...state.clips];
    await Promise.all(Array.from({ length: 3 }, async () => {
      while (queue.length && generation === state.generation) {
        const clip = queue.shift();
        try {
          const response = await fetch(`/api/clips/${encodeURIComponent(clip.clip_id)}/metadata`);
          if (!response.ok) throw new Error('视频信息读取失败');
          clip.duration_s = (await response.json()).duration_s;
        } catch { clip.durationError = true; }
      }
    }));
    if (generation === state.generation) render();
  } catch {
    $('#material-rows').replaceChildren();
    $('#load-message').hidden = false;
    $('#load-message').classList.add('error');
    $('#load-message').textContent = '无法读取素材库，请确认本地服务运行正常，然后点击刷新。';
  } finally { $('#refresh').disabled = false; }
}
document.addEventListener('click', (event) => {
  const copyButton = event.target.closest('[data-copy-id]');
  if (copyButton) {
    navigator.clipboard.writeText(copyButton.dataset.copyId).then(
      () => notify('已复制完整素材编号'),
      () => notify('复制失败，请检查浏览器剪贴板权限')
    );
    return;
  }
  const button = event.target.closest('[data-placeholder]');
  if (button) notify(`${button.dataset.material ? `${button.dataset.material}：` : ''}${button.dataset.placeholder}功能待接入`);
});
$('#search').addEventListener('input', (event) => { state.query = event.target.value.trim().toLowerCase(); state.page = 1; render(); });
$('#sort').addEventListener('change', (event) => { state.sort = event.target.value; state.page = 1; render(); });
$('#page-size').addEventListener('change', (event) => { state.pageSize = Number(event.target.value); state.page = 1; render(); });
$('#previous').addEventListener('click', () => { state.page--; render(); });
$('#next').addEventListener('click', () => { state.page++; render(); });
$('#refresh').addEventListener('click', load);
load();
