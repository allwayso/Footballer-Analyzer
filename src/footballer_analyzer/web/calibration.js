const $ = (id) => document.getElementById(id);
const state = { runId: null, data: null, definition: null, image: null, mask: null, frame: 0, referenceFrame: 0, points: [], roi: [], selectedLandmark: null, pointMode: false, roiMode: false, dirty: false, preview: null, requestId: 0, geometryId: 0, playback: false, jobTimer: null, pendingJob: false };

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

function notify(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

function api(action = "") {
  return `/api/runs/${encodeURIComponent(state.runId)}${action ? `/${action}` : ""}`;
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function calibration() {
  return state.dirty ? state.preview : state.data?.calibration || null;
}

function matrixInverse(matrix) {
  const determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  if (Math.abs(determinant) < 1e-12) return null;
  const inverse = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) inverse[column][row] = (matrix[(row + 1) % 3][(column + 1) % 3] * matrix[(row + 2) % 3][(column + 2) % 3] - matrix[(row + 1) % 3][(column + 2) % 3] * matrix[(row + 2) % 3][(column + 1) % 3]) / determinant;
  return inverse;
}

function setDirty() {
  state.dirty = true;
  state.preview = null;
  if ($("method").value === "manual") $("reviewed").checked = false;
  $("save").disabled = true;
  renderPoints();
  draw();
}

function renderPitch() {
  const surface = $("reference-pitch");
  surface.replaceChildren();
  if (!state.definition) return;
  const { length_m: length, width_m: width } = state.definition.spec;
  surface.setAttribute("viewBox", `-4 -4 ${length + 8} ${width + 8}`);
  surface.append(svgNode("rect", { x: -4, y: -4, width: length + 8, height: width + 8, fill: "#0f2940" }));
  surface.append(svgNode("rect", { x: 0, y: 0, width: length, height: width, fill: "#164b3d", stroke: "#84aebf", "stroke-width": 0.32 }));
  for (const curve of state.definition.lines || []) surface.append(svgNode("polyline", { points: curve.points.map((point) => point.join(",")).join(" "), fill: "none", stroke: "#84aebf", "stroke-width": 0.28 }));
  for (const landmark of state.definition.landmarks || []) {
    const paired = state.points.findIndex((point) => point.landmark_id === landmark.id);
    const selected = state.selectedLandmark === landmark.id;
    const group = svgNode("g", { role: "button", tabindex: 0, class: "reference-point", "aria-label": landmark.label });
    group.append(svgNode("title", {}, landmark.label));
    group.append(svgNode("circle", { cx: landmark.pitch[0], cy: landmark.pitch[1], r: selected ? 1.55 : 1.1, fill: selected ? "#36d9ab" : paired >= 0 ? "#84aebf" : "#fff", stroke: "#0d0d0d", "stroke-width": 0.38 }));
    if (paired >= 0) group.append(svgNode("text", { x: landmark.pitch[0] + 1.4, y: landmark.pitch[1] - 1.1, fill: "#36d9ab", "font-size": 2.5, "font-weight": "bold" }, paired + 1));
    const choose = () => chooseLandmark(landmark.id).catch((error) => notify(error.message, true));
    group.addEventListener("click", choose);
    group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); } });
    surface.append(group);
  }
  $("landmark-select").replaceChildren(option("", "选择场地参考点"), ...(state.definition.landmarks || []).map((landmark) => option(landmark.id, landmark.label)));
  $("landmark-select").value = state.selectedLandmark || "";
}

function renderPoints() {
  const rows = state.points.map((point, index) => {
    const row = document.createElement("tr");
    const landmark = state.definition?.landmarks.find((item) => item.id === point.landmark_id);
    const name = document.createElement("td"); name.textContent = landmark?.label || `参考点 ${index + 1}`;
    const coords = document.createElement("td"); coords.textContent = point.image.map(Math.round).join(" / ");
    const actionCell = document.createElement("td");
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "删除"; remove.setAttribute("aria-label", `删除配对 ${index + 1}`);
    remove.addEventListener("click", () => { state.points.splice(index, 1); setDirty(); });
    actionCell.append(remove); row.append(name, coords, actionCell); return row;
  });
  if (!rows.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.className = "empty-points"; cell.colSpan = 3; cell.textContent = "尚无配对点"; row.append(cell); rows.push(row);
  }
  $("point-rows").replaceChildren(...rows);
  $("landmark-status").textContent = state.pointMode && state.selectedLandmark ? `已选择 ${state.definition?.landmarks.find((item) => item.id === state.selectedLandmark)?.label}，点击左侧原图中的对应地面位置。` : `已配对 ${state.points.length} 点。至少需要四对分布充分的地面点。`;
  renderPitch();
}

function updateFrameControls() {
  const { manifest } = state.data;
  $("frame-label").textContent = `源帧 ${state.frame} / ${manifest.end_frame - 1}`;
  $("frame-slider").value = state.frame;
  $("frame-number").value = state.frame;
  $("time-label").textContent = `${((state.frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} / ${((manifest.end_frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} s`;
}

function draw() {
  if (!state.data) return;
  const canvas = $("video-canvas"), context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) context.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  const active = calibration();
  if ($("show-overlay").checked && state.mask && active?.mode === "white_lines") {
    context.globalAlpha = 0.7; context.drawImage(state.mask, 0, 0, canvas.width, canvas.height); context.globalAlpha = 1;
  }
  if ($("show-overlay").checked && active && state.frame >= active.start_frame && state.frame < active.end_frame) {
    const record = active.mode === "white_lines" ? active.frames?.[String(state.frame)] : active;
    const inverse = record?.homography && (active.mode !== "white_lines" || record.status === "valid") ? matrixInverse(record.homography) : null;
    if (inverse) {
      context.strokeStyle = "#36d9ab"; context.lineWidth = Math.max(2, canvas.width / 800);
      for (const curve of state.definition?.lines || []) {
        context.beginPath(); let connected = false;
        for (const [x, y] of curve.points) {
          const divisor = inverse[2][0] * x + inverse[2][1] * y + inverse[2][2];
          const px = (inverse[0][0] * x + inverse[0][1] * y + inverse[0][2]) / divisor;
          const py = (inverse[1][0] * x + inverse[1][1] * y + inverse[1][2]) / divisor;
          if (!Number.isFinite(px) || !Number.isFinite(py) || Math.abs(px) > canvas.width * 4 || Math.abs(py) > canvas.height * 4) { connected = false; continue; }
          if (connected) context.lineTo(px, py); else context.moveTo(px, py);
          connected = true;
        }
        context.stroke();
      }
    }
    if (active.roi?.length) {
      context.strokeStyle = "#84aebf"; context.lineWidth = Math.max(2, canvas.width / 800); context.setLineDash([10, 8]); context.beginPath();
      active.roi.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y)); context.closePath(); context.stroke(); context.setLineDash([]);
    }
  }
  if (state.frame === state.referenceFrame) state.points.forEach((point, index) => {
    context.fillStyle = "#36d9ab"; context.beginPath(); context.arc(point.image[0], point.image[1], Math.max(7, canvas.width / 220), 0, Math.PI * 2); context.fill();
    context.fillStyle = "#0d0d0d"; context.font = `bold ${Math.max(18, canvas.width / 70)}px sans-serif`; context.fillText(String(index + 1), point.image[0] + 11, point.image[1] - 10);
  });
  if (state.roi.length && state.frame === state.referenceFrame) {
    context.strokeStyle = "#36d9ab"; context.lineWidth = Math.max(2, canvas.width / 800); context.beginPath();
    state.roi.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
    if (!state.roiMode) context.closePath(); context.stroke();
  }
  updateFrameControls();
}

async function showFrame(frame) {
  if (!state.data) return;
  const manifest = state.data.manifest;
  if ((state.pointMode || state.roiMode) && frame !== state.frame) return notify("请先结束当前点选，再切换视频帧。", true);
  const bounded = Math.min(manifest.end_frame - 1, Math.max(manifest.start_frame, Math.round(frame)));
  const requestId = ++state.requestId;
  const image = new Image();
  const loaded = new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("无法读取视频帧，请确认源视频仍存在。")); });
  image.src = api(`frames/${bounded}`);
  let mask = null;
  const active = calibration();
  if (active?.mode === "white_lines" && bounded >= active.start_frame && bounded < active.end_frame) {
    mask = new Image();
    const maskLoaded = new Promise((resolve, reject) => { mask.onload = resolve; mask.onerror = () => reject(new Error("白线识别缓存读取失败，请重新运行标定。")); });
    mask.src = `${api(`white-lines/${bounded}.png`)}?version=${encodeURIComponent(active.calibration_id)}`;
    await Promise.all([loaded, maskLoaded]);
  } else await loaded;
  if (requestId !== state.requestId) return;
  state.frame = bounded; state.image = image; state.mask = mask;
  const canvas = $("video-canvas");
  canvas.width = manifest.video.width; canvas.height = manifest.video.height;
  $("video-canvas").parentElement.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
  draw();
}

function stopPlayback() {
  state.playback = false; $("play").textContent = "▶"; $("play").setAttribute("aria-label", "播放");
}

async function play() {
  if (state.playback) return stopPlayback();
  if (!state.data || state.pointMode || state.roiMode) return;
  state.playback = true; $("play").textContent = "Ⅱ"; $("play").setAttribute("aria-label", "暂停");
  const step = Math.max(1, Math.round(state.data.manifest.video.fps / 10));
  while (state.playback) {
    if (state.frame + step >= state.data.manifest.end_frame) { stopPlayback(); break; }
    try { await showFrame(state.frame + step); } catch (error) { notify(error.message, true); stopPlayback(); }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function chooseLandmark(id) {
  if (!state.data || !id) return;
  stopPlayback();
  state.pointMode = false; state.roiMode = false;
  if (state.points.length && state.frame !== state.referenceFrame) await showFrame(state.referenceFrame);
  if (!state.points.length) state.referenceFrame = state.frame;
  state.selectedLandmark = id; state.roiMode = false; state.pointMode = true;
  $("point-mode").textContent = "结束点选";
  renderPoints(); draw();
}

async function refreshDefinition(reset = false) {
  if (!state.data) return;
  const geometryId = ++state.geometryId;
  const model = $("pitch-model").value;
  const query = new URLSearchParams({ model, length: $("pitch-length").value, width: $("pitch-width").value });
  const result = await request(`/api/pitch-models?${query}`);
  if (geometryId !== state.geometryId) return;
  state.definition = result.models[0];
  if (reset) { state.points = []; state.roi = []; state.selectedLandmark = null; }
  setDirty();
  if (model === "eight") $("method").value = "manual";
  renderState();
}

function payload() {
  if (!state.definition) throw new Error("请先选择有效的场地尺寸。");
  const result = { pitch_model: $("pitch-model").value, pitch_length_m: state.definition.spec.length_m, pitch_width_m: state.definition.spec.width_m, reference_frame: state.referenceFrame, start_frame: Number($("start-frame").value), end_frame: Number($("end-frame").value), reviewed: $("reviewed").checked, note: "场地参考点手动固定标定" };
  if (state.points.every((point) => point.landmark_id)) result.point_pairs = state.points.map((point) => ({ landmark_id: point.landmark_id, image: point.image }));
  else { result.image_points = state.points.map((point) => point.image); result.pitch_points = state.points.map((point) => point.pitch); }
  if (state.roi.length) result.roi = state.roi;
  return result;
}

function renderState() {
  const active = calibration();
  const manual = $("method").value === "manual";
  $("method").querySelector('[value="white_lines"]').disabled = $("pitch-model").value === "eight";
  $("manual-controls").hidden = !manual;
  $("point-mode").textContent = state.pointMode ? "结束点选" : "开始点选";
  $("roi-mode").textContent = state.roiMode ? "结束区域点选" : "点选可靠区域";
  $("auto-controls").hidden = manual;
  $("calibration-status").textContent = !active ? "待标定" : active.reviewed ? "已验收" : "待验收";
  if (!active) $("calibration-summary").textContent = `${state.points.length} 对参考点 · 预览后保存`;
  else if (active.mode === "white_lines") $("calibration-summary").textContent = `${active.summary.valid_frames} / ${active.summary.frame_count} 帧有效 · ${active.reviewed ? "已人工验收" : "等待人工验收"}`;
  else $("calibration-summary").textContent = `${active.image_points.length} 对参考点 · 拟合残差 ${active.fit_rmse_m.toFixed(3)} m · ${active.reviewed ? "已验收" : "草稿"}`;
  $("save").hidden = !manual;
  $("save").disabled = !state.preview || state.points.length < 4;
  $("save-review").disabled = active?.mode !== "white_lines";
  $("run-auto").disabled = $("pitch-model").value === "eight";
  $("job-status").textContent = active?.mode === "white_lines" ? `${active.summary.valid_frames} / ${active.summary.frame_count} 帧匹配有效。请回看运动过程，核对模型线与画面白线。` : "适用于运动镜头，逐帧匹配球场白线。";
}

async function loadRun(runId, preserveFrame = false) {
  stopPlayback();
  clearTimeout(state.jobTimer);
  state.runId = runId; state.preview = null; state.dirty = false; state.pointMode = false; state.roiMode = false; state.requestId++;
  const loadGeometryId = ++state.geometryId;
  state.selectedLandmark = null;
  const data = await request(api());
  if (state.runId !== runId) return;
  state.data = data;
  const manifest = data.manifest, saved = data.calibration;
  const model = saved?.pitch_model || data.pitch_definition.spec.model || manifest.config.pitch_model || "eleven";
  $("pitch-model").value = model;
  $("method").value = saved ? saved.mode === "white_lines" ? "white_lines" : "manual" : manifest.config.calibration_method || (model === "eight" ? "manual" : "white_lines");
  $("pitch-length").value = saved?.pitch_length_m || manifest.config.pitch_length_m || data.pitch_definition.spec.length_m;
  $("pitch-width").value = saved?.pitch_width_m || manifest.config.pitch_width_m || data.pitch_definition.spec.width_m;
  $("start-frame").value = saved?.start_frame ?? manifest.start_frame;
  $("end-frame").value = saved?.end_frame ?? manifest.end_frame;
  const query = new URLSearchParams({ model, length: $("pitch-length").value, width: $("pitch-width").value });
  const definitions = await request(`/api/pitch-models?${query}`);
  if (state.runId !== runId || state.geometryId !== loadGeometryId) return;
  state.definition = definitions.models[0];
  if (state.runId !== runId) return;
  $("reviewed").checked = saved?.reviewed === true;
  state.referenceFrame = saved?.reference_frame ?? manifest.start_frame;
  state.points = saved?.image_points ? saved.image_points.map((image, index) => ({ image: [...image], pitch: [...saved.pitch_points[index]], landmark_id: saved.point_pairs?.[index]?.landmark_id || state.definition.landmarks.find((item) => item.pitch.every((value, axis) => Math.abs(value - saved.pitch_points[index][axis]) < 1e-6))?.id })) : [];
  state.roi = saved?.roi?.map((point) => [...point]) || [];
  $("run-description").textContent = `${manifest.config.match_id} · ${manifest.video.width}×${manifest.video.height} · ${manifest.video.fps.toFixed(2)} FPS`;
  $("clip-label").textContent = `${manifest.config.clip_id} · ${runId.slice(-8)}`;
  const slider = $("frame-slider"); slider.min = manifest.start_frame; slider.max = manifest.end_frame - 1; slider.step = 1;
  $("frame-number").min = manifest.start_frame; $("frame-number").max = manifest.end_frame - 1;
  renderPoints(); renderState();
  await showFrame(preserveFrame ? Math.max(manifest.start_frame, Math.min(state.frame, manifest.end_frame - 1)) : saved?.reference_frame ?? manifest.start_frame);
  notify(saved?.mode === "white_lines" ? "逐帧标定已加载。播放并核对模型与画面白线的重合情况。" : saved ? "固定标定已加载。可检查参考点、预览或重新验收。" : "选择模型上的参考点，再点击左侧画面中的对应地面位置。");
  watchCalibration(runId).catch((error) => notify(error.message, true));
}

async function watchCalibration(runId) {
  clearTimeout(state.jobTimer);
  const job = await request(`/api/runs/${encodeURIComponent(runId)}/calibration-job`);
  if (runId !== state.runId) return;
  const running = job.status === "running";
  $("run-auto").disabled = running || $("pitch-model").value === "eight";
  $("run-auto").textContent = running ? "正在逐帧识别…" : "运行逐帧白线标定";
  if (running) {
    state.pendingJob = true;
    $("job-status").textContent = `已处理 ${job.processed_frames} / ${job.total_frames} 帧。`;
    state.jobTimer = setTimeout(() => watchCalibration(runId).catch((error) => notify(error.message, true)), 1000);
  } else if (job.status === "failed") $("job-status").textContent = `标定失败：${job.error}。可尝试固定镜头参考点配对。`;
  else if (state.pendingJob && job.status === "complete") { state.pendingJob = false; await loadRun(runId, true); notify("逐帧标定完成，请回看并保存验收状态。"); }
}

async function initialize(preferredId, preferredClip) {
  const runs = await request("/api/runs");
  const clipKey = (value) => String(value).replace(/\.[^.]+$/, "");
  const completed = runs.filter((run) => run.status === "complete" && (!preferredClip || clipKey(run.clip_id) === clipKey(preferredClip)));
  $("run-select").replaceChildren(...completed.map((run) => option(run.run_id, `${run.clip_id} · ${run.run_id.slice(-8)}`)));
  if (!completed.length) {
    $("run-select").append(option("", "尚无已完成运行")); $("run-select").disabled = true;
    document.querySelectorAll(".calibration-layout button, .calibration-layout input, .calibration-layout select").forEach((node) => { node.disabled = true; });
    $("clip-label").textContent = "没有可标定的运行"; notify("请先在追踪工作台完成一个视频运行，再返回此处标定。"); return;
  }
  $("run-select").disabled = false;
  const matchingClip = preferredClip && completed.find((run) => clipKey(run.clip_id) === clipKey(preferredClip));
  if (preferredClip && !matchingClip) {
    $("run-select").prepend(option("", "该素材暂无已完成运行"));
    $("run-select").value = "";
    document.querySelectorAll(".calibration-layout button, .calibration-layout input, .calibration-layout select").forEach((node) => { node.disabled = true; });
    $("clip-label").textContent = preferredClip;
    $("run-description").textContent = "该素材暂无可标定的分析运行。";
    notify(`素材“${preferredClip}”暂无已完成分析运行，请先完成分析。`, true);
    return;
  }
  const chosen = completed.some((run) => run.run_id === preferredId) ? preferredId : matchingClip?.run_id || completed[0].run_id;
  $("run-select").value = chosen;
  await loadRun(chosen);
}

$("run-select").addEventListener("change", () => {
  if (!$("run-select").value) return;
  document.querySelectorAll(".calibration-layout button, .calibration-layout input, .calibration-layout select").forEach((node) => { node.disabled = false; });
  loadRun($("run-select").value).catch((error) => notify(error.message, true));
});
$("previous").addEventListener("click", () => showFrame(state.frame - 1).catch((error) => notify(error.message, true)));
$("next").addEventListener("click", () => showFrame(state.frame + 1).catch((error) => notify(error.message, true)));
$("play").addEventListener("click", play);
$("jump").addEventListener("click", () => showFrame(Number($("frame-number").value)).catch((error) => notify(error.message, true)));
$("frame-number").addEventListener("keydown", (event) => { if (event.key === "Enter") $("jump").click(); });
$("frame-slider").addEventListener("input", () => showFrame(Number($("frame-slider").value)).catch((error) => notify(error.message, true)));
$("show-overlay").addEventListener("change", draw);
$("landmark-select").addEventListener("change", () => chooseLandmark($("landmark-select").value).catch((error) => notify(error.message, true)));
$("point-mode").addEventListener("click", () => {
  if (state.roiMode) { state.roiMode = false; }
  else if (!state.selectedLandmark && !state.pointMode) return notify("请先选择场地模型上的参考点。", true);
  else state.pointMode = !state.pointMode;
  $("point-mode").textContent = state.pointMode ? "结束点选" : "开始点选";
  $("video-canvas").classList.toggle("pointing", state.pointMode || state.roiMode);
  renderPoints(); draw();
});
$("video-canvas").addEventListener("click", (event) => {
  if (!state.data || (!state.pointMode && !state.roiMode)) return;
  const bounds = event.currentTarget.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * event.currentTarget.width / bounds.width;
  const y = (event.clientY - bounds.top) * event.currentTarget.height / bounds.height;
  if (state.roiMode) { state.roi.push([x, y]); setDirty(); return; }
  if (!state.selectedLandmark) return notify("请先选择场地模型上的参考点。", true);
  const landmark = state.definition.landmarks.find((item) => item.id === state.selectedLandmark);
  const existing = state.points.findIndex((point) => point.landmark_id === landmark.id);
  const point = { image: [x, y], pitch: [...landmark.pitch], landmark_id: landmark.id };
  if (existing >= 0) state.points[existing] = point; else state.points.push(point);
  state.selectedLandmark = null; state.pointMode = false;
  $("point-mode").textContent = "开始点选"; $("video-canvas").classList.remove("pointing");
  setDirty(); notify(`已记录 ${landmark.label} 的图像坐标。继续选择下一个场地参考点。`);
});
$("undo-point").addEventListener("click", () => { if (state.points.length) { state.points.pop(); setDirty(); } });
$("clear-points").addEventListener("click", () => { state.points = []; state.selectedLandmark = null; state.pointMode = false; $("point-mode").textContent = "开始点选"; setDirty(); });
$("roi-mode").addEventListener("click", async () => {
  stopPlayback();
  if (state.frame !== state.referenceFrame) { state.pointMode = false; state.roiMode = false; await showFrame(state.referenceFrame).catch((error) => notify(error.message, true)); }
  if (state.roiMode) { state.roiMode = false; $("roi-mode").textContent = "点选可靠区域"; }
  else { state.roi = []; state.roiMode = true; state.pointMode = false; $("roi-mode").textContent = "结束区域点选"; }
  $("video-canvas").classList.toggle("pointing", state.pointMode || state.roiMode); draw();
});
$("clear-roi").addEventListener("click", () => { state.roi = []; state.roiMode = false; $("roi-mode").textContent = "点选可靠区域"; setDirty(); });
$("pitch-model").addEventListener("change", async () => {
  try {
  const preset = await request(`/api/pitch-models?model=${encodeURIComponent($("pitch-model").value)}`);
  $("pitch-length").value = preset.models[0].spec.length_m; $("pitch-width").value = preset.models[0].spec.width_m;
  if ($("pitch-model").value === "eight") $("method").value = "manual";
  await refreshDefinition(true);
  } catch (error) { notify(error.message, true); }
});
for (const id of ["pitch-length", "pitch-width"]) $(id).addEventListener("change", () => refreshDefinition(true).catch((error) => notify(error.message, true)));
$("method").addEventListener("change", () => { stopPlayback(); state.pointMode = false; state.roiMode = false; renderState(); });
$("start-frame").addEventListener("change", setDirty);
$("end-frame").addEventListener("change", setDirty);
$("preview").addEventListener("click", async () => {
  if (!state.data) return;
  stopPlayback(); state.pointMode = false; state.roiMode = false;
  $("point-mode").textContent = "开始点选"; $("roi-mode").textContent = "点选可靠区域"; $("video-canvas").classList.remove("pointing");
  try {
    const runId = state.runId, body = payload(), snapshot = JSON.stringify(body);
    const result = await request(api("calibration-preview"), body);
    if (runId !== state.runId || snapshot !== JSON.stringify(payload())) return;
    state.preview = result;
    state.dirty = true; $("reviewed").checked = false; renderState(); draw();
    $("calibration-summary").textContent = `${state.points.length} 对参考点 · ${state.preview.inlier_count} 个内点 · 残差 ${state.preview.fit_rmse_px.toFixed(2)} px`;
    notify("预览已加载。请检查其他帧的场线贴合，再确认保存。拟合残差不能替代独立精度验收。");
  } catch (error) { notify(error.message, true); }
});
$("save").addEventListener("click", async () => {
  if (!state.preview) return notify("请先预览当前参考点配对。", true);
  try {
    const body = { ...payload(), reviewed: $("reviewed").checked };
    await request(api("calibration"), body);
    await loadRun(state.runId, true);
    notify(body.reviewed ? "固定标定已确认并保存。" : "固定标定草稿已保存。回看验收后再确认。 ");
  } catch (error) { notify(error.message, true); }
});
$("discard").addEventListener("click", () => state.runId && loadRun(state.runId, true).catch((error) => notify(error.message, true)));
$("run-auto").addEventListener("click", async () => {
  if (!state.runId) return;
  try {
    const geometry = await request(`/api/pitch-models?model=${encodeURIComponent($("pitch-model").value)}&length=${encodeURIComponent($("pitch-length").value)}&width=${encodeURIComponent($("pitch-width").value)}`);
    const spec = geometry.models[0].spec;
    await request(api("calibration-auto"), { pitch_model: spec.model, pitch_length_m: spec.length_m, pitch_width_m: spec.width_m });
    state.pendingJob = true; await watchCalibration(state.runId); notify("逐帧白线标定已启动，完成后会自动载入结果。");
  } catch (error) { notify(error.message, true); }
});

$("save-review").addEventListener("click", async () => {
  const active = state.data?.calibration;
  if (active?.mode !== "white_lines") return notify("请先运行逐帧白线标定。", true);
  if (Number($("pitch-length").value) !== active.pitch_length_m || Number($("pitch-width").value) !== active.pitch_width_m || $("pitch-model").value !== state.data.pitch_definition.spec.model) return notify("场地设置已改变，请重新运行逐帧白线标定后再验收。", true);
  try { await request(api("calibration-review"), { calibration_id: active.calibration_id, reviewed: $("reviewed").checked }); await loadRun(state.runId, true); notify("逐帧标定验收状态已保存。"); }
  catch (error) { notify(error.message, true); }
});

const initialParams = new URLSearchParams(window.location.search);
initialize(initialParams.get("run"), initialParams.get("clip")).catch((error) => notify(error.message, true));
