const $ = (id) => document.getElementById(id);
const state = { runId: null, data: null, frame: 0, image: null, timer: null, selectedRecordId: null, mergeSourceId: null, frameRequest: 0 };

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

function notify(message, error = false) {
  $("review-notice").textContent = message;
  $("review-notice").classList.toggle("error", error);
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function renderRecords() {
  const segments = state.data.field_segments || [];
  const byId = new Map(segments.map((item) => [item.record_id, item]));
  if (!byId.has(state.selectedRecordId)) state.selectedRecordId = null;
  $("segment-status").textContent = !state.data.calibration
    ? "先完成场地标定以筛出场内观测"
    : state.data.calibration.reviewed !== true
      ? "场地标定尚未人工确认，暂不纳入校对记录"
      : `${segments.length} 条有效场内记录 · 按轨迹断点整理`;
  const rows = segments.map((item) => {
    const row = document.createElement("tr");
    row.className = `review-record${item.record_id === state.selectedRecordId ? " selected" : ""}`;
    const selectRecord = (frame) => {
      state.selectedRecordId = item.record_id;
      renderRecords();
      stopPlayback();
      showFrame(frame).catch((error) => notify(error.message, true));
    };
    const start = document.createElement("button");
    start.className = "frame-link"; start.textContent = item.start_frame; start.title = `跳转到源帧 ${item.start_frame}`;
    start.addEventListener("click", () => selectRecord(item.start_frame));
    const end = document.createElement("button");
    end.className = "frame-link"; end.textContent = item.end_frame; end.title = `跳转到源帧 ${item.end_frame}`;
    end.addEventListener("click", () => selectRecord(item.end_frame));
    const frames = document.createElement("td"); frames.className = "review-frames"; frames.append(start, document.createTextNode(" / "), end);
    const merged = item.merge_group_size > 1 ? byId.get(item.merge_group_id) : null;
    const merge = document.createElement("button"); merge.className = "secondary"; merge.textContent = "并入";
    merge.disabled = segments.length < 2;
    merge.addEventListener("click", () => openMergeDialog(item));
    const action = document.createElement("td"); action.append(merge);
    row.append(document.createElement("td"), frames, document.createElement("td"), document.createElement("td"), document.createElement("td"), action);
    row.children[0].textContent = `#${item.track_id}`;
    row.children[2].textContent = item.observation_count;
    row.children[3].textContent = item.team_id || "unknown";
    row.children[4].textContent = merged ? `#${merged.track_id} · ${merged.start_frame}` : "—";
    return row;
  });
  if (!rows.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.colSpan = 6; cell.className = "empty-row"; cell.textContent = !state.data.calibration
      ? "完成场地标定后显示场内片段。"
      : state.data.calibration.reviewed !== true
        ? "人工确认场地标定后，才会显示场内片段。"
        : "没有通过场地范围校验的球员观测。";
    row.append(cell); rows.push(row);
  }
  $("field-segment-table").replaceChildren(...rows);
  const history = (state.data.field_review_history || []).map((operation) => {
    const row = document.createElement("tr"), source = byId.get(operation.source_record_id), destination = byId.get(operation.target_record_id);
    const destinationLabel = destination ? `#${destination.track_id} · ${destination.start_frame}` : operation.target_record_id;
    const values = [new Date(operation.created_at).toLocaleString(), source ? `#${source.track_id} · ${source.start_frame}` : operation.source_record_id, operation.target_frame === undefined ? destinationLabel : `${destinationLabel} · 帧 ${operation.target_frame}`, operation.status === "active" ? "已生效" : operation.status === "undone" ? "已撤销" : "记录已变化"];
    for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    const cell = document.createElement("td"), undo = document.createElement("button");
    undo.className = "remove-button"; undo.textContent = "撤销"; undo.disabled = operation.status !== "active";
    undo.addEventListener("click", async () => {
      try {
        await request(`/api/runs/${encodeURIComponent(state.runId)}/field-merge-undo`, { operation_id: operation.operation_id });
        await loadRun(state.runId); notify("融合操作已撤销。");
      } catch (error) { notify(error.message, true); }
    });
    cell.append(undo); row.append(cell); return row;
  });
  if (!history.length) { const row = document.createElement("tr"), cell = document.createElement("td"); cell.colSpan = 5; cell.className = "empty-row"; cell.textContent = "尚无融合操作。"; row.append(cell); history.push(row); }
  $("field-merge-history").replaceChildren(...history);
}

async function loadRun(runId) {
  stopPlayback();
  const data = await request(`/api/runs/${encodeURIComponent(runId)}`);
  state.runId = runId; state.data = data;
  if (!(data.field_segments || []).some((item) => item.record_id === state.selectedRecordId)) state.selectedRecordId = null;
  const selector = $("run-select"); selector.value = runId;
  const frame = Math.max(data.manifest.start_frame, Math.min(state.frame || data.manifest.start_frame, data.manifest.end_frame - 1));
  $("frame-number").min = data.manifest.start_frame; $("frame-number").max = data.manifest.end_frame - 1;
  $("frame-slider").min = data.manifest.start_frame; $("frame-slider").max = data.manifest.end_frame - 1;
  renderRecords(); await showFrame(frame);
}

async function showFrame(frameIndex) {
  const manifest = state.data.manifest;
  const parsedFrame = Number(frameIndex);
  const frame = Math.max(manifest.start_frame, Math.min(manifest.end_frame - 1, Number.isFinite(parsedFrame) ? Math.round(parsedFrame) : manifest.start_frame));
  const requestId = ++state.frameRequest;
  const runId = state.runId;
  const image = new Image();
  image.src = `/api/runs/${encodeURIComponent(runId)}/frames/${frame}?t=${Date.now()}`;
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("源视频帧读取失败")); });
  if (requestId !== state.frameRequest || runId !== state.runId) return;
  state.frame = frame; state.image = image;
  const canvas = $("review-canvas"), context = canvas.getContext("2d");
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; context.drawImage(image, 0, 0);
  const activeSegments = (state.data.field_segments || []).filter((segment) => segment.field_frames.includes(frame));
  const fieldTrackIds = new Set(activeSegments.map((segment) => segment.track_id));
  const selected = activeSegments.find((segment) => segment.record_id === state.selectedRecordId);
  for (const row of state.data.tracks.filter((entry) => entry.frame_index === frame && fieldTrackIds.has(entry.track_id))) {
    const isSelected = selected?.track_id === row.track_id;
    context.strokeStyle = isSelected ? "#ff2020" : row.team_id === "cluster_0" ? "#28cfac" : row.team_id === "cluster_1" ? "#f9b26a" : "#e9f0e8";
    context.lineWidth = isSelected ? Math.max(5, image.naturalWidth / 360) : Math.max(2, image.naturalWidth / 900);
    context.strokeRect(row.bbox_x1, row.bbox_y1, row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1);
    context.fillStyle = context.strokeStyle; context.font = `bold ${Math.max(14, image.naturalWidth / 85)}px sans-serif`; context.fillText(`#${row.track_id}`, row.bbox_x1, Math.max(18, row.bbox_y1 - 4));
  }
  const seconds = frame / manifest.video.fps;
  $("frame-label").textContent = `FRAME ${frame}`; $("time-label").textContent = `${seconds.toFixed(2)} / ${(manifest.video.frame_count / manifest.video.fps).toFixed(2)} s`;
  $("frame-number").value = frame; $("frame-slider").value = frame;
}

function stopPlayback() { clearTimeout(state.timer); state.timer = null; $("play").textContent = "▶"; }
function openMergeDialog(source) {
  stopPlayback();
  state.mergeSourceId = source.record_id;
  $("merge-source").textContent = `来源轨迹 #${source.track_id} · ${source.start_frame}–${source.end_frame} 帧`;
  const frame = Math.max(state.data.manifest.start_frame, Math.min(state.frame, state.data.manifest.end_frame - 1));
  $("merge-frame").min = state.data.manifest.start_frame;
  $("merge-frame").max = state.data.manifest.end_frame - 1;
  $("merge-frame").value = frame;
  refreshMergeCandidates();
  $("merge-dialog").showModal();
}
function refreshMergeCandidates() {
  const source = (state.data.field_segments || []).find((item) => item.record_id === state.mergeSourceId);
  const frameValue = Number($("merge-frame").value);
  const validFrame = Number.isInteger(frameValue) && frameValue >= state.data.manifest.start_frame && frameValue < state.data.manifest.end_frame;
  const candidates = validFrame ? (state.data.field_segments || []).filter((item) => item.record_id !== source?.record_id
    && item.merge_group_id !== source?.merge_group_id && item.field_frames.includes(frameValue)) : [];
  const select = $("merge-target");
  const previousTarget = select.value;
  select.replaceChildren(option("", candidates.length ? "选择目标轨迹" : "该帧没有可合并的轨迹"), ...candidates.map((item) => option(item.record_id, `#${item.track_id} · ${item.start_frame}–${item.end_frame} 帧`)));
  if (candidates.some((item) => item.record_id === previousTarget)) select.value = previousTarget;
  select.disabled = !candidates.length;
  $("merge-candidates").textContent = validFrame
    ? candidates.length ? `源帧 ${frameValue} 有 ${candidates.length} 条可选场内轨迹。` : `源帧 ${frameValue} 没有其他可合并的场内轨迹。`
    : `请输入 ${state.data.manifest.start_frame}–${state.data.manifest.end_frame - 1} 范围内的源帧。`;
  $("confirm-merge").disabled = !candidates.length || !select.value;
}
$("merge-target").addEventListener("change", () => { $("confirm-merge").disabled = !$("merge-target").value; });
$("merge-frame").addEventListener("input", refreshMergeCandidates);
$("merge-frame").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); $("locate-merge-frame").click(); } });
$("locate-merge-frame").addEventListener("click", () => {
  refreshMergeCandidates(); stopPlayback();
  showFrame($("merge-frame").value).catch((error) => notify(error.message, true));
});
$("cancel-merge").addEventListener("click", () => $("merge-dialog").close());
$("close-merge").addEventListener("click", () => $("merge-dialog").close());
$("merge-dialog").addEventListener("click", (event) => { if (event.target === $("merge-dialog")) $("merge-dialog").close(); });
$("merge-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const source = (state.data.field_segments || []).find((item) => item.record_id === state.mergeSourceId);
  const targetId = $("merge-target").value;
  const targetFrame = Number($("merge-frame").value);
  if (!source || !targetId || !Number.isInteger(targetFrame)) return;
  $("confirm-merge").disabled = true;
  try {
    await request(`/api/runs/${encodeURIComponent(state.runId)}/field-merges`, { source_record_id: source.record_id, target_record_id: targetId, target_frame: targetFrame });
    $("merge-dialog").close();
    state.mergeSourceId = null;
    await loadRun(state.runId);
    notify(`轨迹 #${source.track_id} 已合并到所选目标。`);
  } catch (error) { notify(error.message, true); refreshMergeCandidates(); }
});
$("run-select").addEventListener("change", () => loadRun($("run-select").value).catch((error) => notify(error.message, true)));
$("jump-frame").addEventListener("click", () => { stopPlayback(); showFrame($("frame-number").value).catch((error) => notify(error.message, true)); });
$("frame-number").addEventListener("keydown", (event) => { if (event.key === "Enter") $("jump-frame").click(); });
$("frame-slider").addEventListener("input", () => { stopPlayback(); showFrame($("frame-slider").value).catch((error) => notify(error.message, true)); });
$("previous-frame").addEventListener("click", () => { stopPlayback(); showFrame(state.frame - state.data.manifest.config.stride).catch((error) => notify(error.message, true)); });
$("next-frame").addEventListener("click", () => { stopPlayback(); showFrame(state.frame + state.data.manifest.config.stride).catch((error) => notify(error.message, true)); });
$("play").addEventListener("click", () => {
  if (state.timer) return stopPlayback();
  $("play").textContent = "Ⅱ";
  const advance = async () => {
    const next = state.frame + state.data.manifest.config.stride;
    if (next >= state.data.manifest.end_frame) { stopPlayback(); return; }
    try {
      await showFrame(next);
      if (state.timer) state.timer = setTimeout(advance, Math.max(40, 1000 / state.data.manifest.video.fps * state.data.manifest.config.stride));
    } catch (error) { stopPlayback(); notify(error.message, true); }
  };
  state.timer = setTimeout(advance, 0);
});

try {
  const runs = (await request("/api/runs")).filter((run) => run.status === "complete");
  $("run-select").replaceChildren(...runs.map((run) => option(run.run_id, `${run.run_id} · ${run.clip_id} · ${run.status}`)));
  if (!runs.length) notify("还没有已完成的分析运行，请先到追踪工作台运行视频。", true);
  else { const requested = new URLSearchParams(location.search).get("run"); const runId = runs.some((run) => run.run_id === requested) ? requested : runs[0].run_id; await loadRun(runId); notify("选择首末帧复核片段；确认同一球员后可合并轨迹。 "); }
} catch (error) { notify(error.message, true); }
