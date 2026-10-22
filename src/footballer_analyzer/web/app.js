const element = (id) => document.getElementById(id);
const state = { runId: null, data: null, frame: 0, selected: null, image: null, whiteImage: null, playing: false, pointMode: false, roiMode: false, roi: [], points: [], requestId: 0, runRequestId: 0, frameRows: new Map(), detectionRows: new Map(), motionRows: new Map(), draftDirty: false, calibrationTimer: null };
const colors = { home: "#28cfac", away: "#f9b26a", cluster_0: "#28cfac", cluster_1: "#f9b26a", unknown: "#a6b3b2" };
const reasons = { unbound: "身份尚未确认", excluded: "人工排除", outside_roi: "超出可靠统计区域", outside_pitch: "投影超出球场", clipped_feet: "脚点被画面裁切", uncalibrated: "尚未标定", calibration_unreliable: "本帧白线几何不足，暂停场地投影", outside_calibration_window: "超出标定有效区间", invalid_projection: "投影无效", speed_jump: "异常速度跳变，已切段" };
const analysis = { clips: [], timer: null, running: false, activeClip: null, handledId: null, runId: null };
const analysisStages = { queued: "准备分析", inspecting: "核实视频可解码区间", sampling: "采样球衣颜色", tracking: "人物检测与持续追踪", awaiting_calibration: "人物追踪完成，待场地标定", complete: "分析完成" };
const ball = { models: [], result: null, rows: new Map(), timer: null, requestId: 0, running: false, handledId: null };
Object.assign(state, { draftPitch: null, preview: null, selectedLandmark: null, draftVersion: 0, geometryRequest: 0, pendingCalibrations: new Set() });
let pitchModels = [];

function activeCalibration() {
  return state.draftDirty ? state.preview : state.data?.calibration;
}

function activePitch() {
  return state.draftDirty ? state.draftPitch : state.data?.pitch_definition;
}

function notify(message, error = false) {
  element("notice").textContent = message;
  element("notice").classList.toggle("error", error);
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

function api(action = "") {
  return `/api/runs/${encodeURIComponent(state.runId)}${action ? `/${action}` : ""}`;
}

function make(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function option(value, label) {
  const node = make("option", label);
  node.value = value;
  return node;
}

function bindingFor(row) {
  return state.data.bindings.bindings.find((entry) => entry.track_id === row.track_id && entry.start_frame <= row.frame_index && row.frame_index < entry.end_frame);
}

function calibrationAt(frameIndex) {
  const calibration = activeCalibration();
  if (calibration?.mode !== "white_lines") return calibration;
  const record = calibration.frames?.[String(frameIndex)];
  return record?.status === "valid" && record.homography ? { ...calibration, homography: record.homography, roi: record.roi } : null;
}

function project(row) {
  const calibration = calibrationAt(row.frame_index);
  const binding = bindingFor(row);
  const base = { row, binding, color: colors[binding?.team_id || row.team_id] || colors.unknown, position: null, reason: null };
  if (!calibration) return { ...base, reason: activeCalibration()?.mode === "white_lines" ? "calibration_unreliable" : "uncalibrated" };
  if (row.frame_index < calibration.start_frame || row.frame_index >= calibration.end_frame) return { ...base, reason: "outside_calibration_window" };
  const footX = (row.bbox_x1 + row.bbox_x2) / 2;
  const footY = row.bbox_y2;
  const matrix = calibration.homography;
  const divisor = matrix[2][0] * footX + matrix[2][1] * footY + matrix[2][2];
  if (Math.abs(divisor) < 1e-8) return { ...base, reason: "invalid_projection" };
  const worldX = (matrix[0][0] * footX + matrix[0][1] * footY + matrix[0][2]) / divisor;
  const worldY = (matrix[1][0] * footX + matrix[1][1] * footY + matrix[1][2]) / divisor;
  if (!Number.isFinite(worldX) || !Number.isFinite(worldY)) return { ...base, reason: "invalid_projection" };
  base.position = [worldX, worldY];
  if (footY >= calibration.height - 2) base.reason = "clipped_feet";
  else if (!pointInPolygon(footX, footY, calibration.roi)) base.reason = "outside_roi";
  else if (worldX < 0 || worldX > calibration.pitch_length_m || worldY < 0 || worldY > calibration.pitch_width_m) base.reason = "outside_pitch";
  else if (binding?.action === "exclude") base.reason = "excluded";
  else if (!binding) base.reason = "unbound";
  const motion = state.motionRows.get(`${row.frame_index}:${row.track_id}`);
  if (motion && !motion.valid && !state.draftDirty && !state.data.latest_report?.stale) base.reason = motion.reason;
  return base;
}

function pointInPolygon(horizontal, vertical, polygon) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [currentX, currentY] = polygon[index];
    const [previousX, previousY] = polygon[previous];
    const cross = (horizontal - currentX) * (previousY - currentY) - (vertical - currentY) * (previousX - currentX);
    if (Math.abs(cross) < 1e-6 && horizontal >= Math.min(currentX, previousX) && horizontal <= Math.max(currentX, previousX) && vertical >= Math.min(currentY, previousY) && vertical <= Math.max(currentY, previousY)) return true;
    if ((currentY > vertical) !== (previousY > vertical) && horizontal < (previousX - currentX) * (vertical - currentY) / (previousY - currentY) + currentX) inside = !inside;
  }
  return inside;
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function pitchShape(tag, attributes, text) {
  element("pitch").append(svgNode(tag, attributes, text));
}

function pitchDisplayPoint(point, spec) {
  return spec.model === "eight" ? [point[1], spec.length_m - point[0]] : [...point];
}

function landmarkLabel(landmark) {
  if (!landmark) return "历史自定义参考点";
  if (state.draftPitch?.spec.model !== "eight") return landmark.label;
  return landmark.label.replace(/^左侧/, "近端").replace(/^右侧/, "远端").replace(/上/g, "左").replace(/下/g, "右");
}

function manualMapEditing() {
  return element("calibration-method").value === "manual" && !element("calibration-tab").hidden;
}

function currentBall() {
  return element("show-ball").checked ? ball.rows.get(state.frame)?.selected : null;
}

function ballPosition() {
  const candidate = currentBall();
  const calibration = calibrationAt(state.frame);
  if (!candidate || !calibration || state.frame < calibration.start_frame || state.frame >= calibration.end_frame) return null;
  const x = (candidate[0] + candidate[2]) / 2;
  const y = (candidate[1] + candidate[3]) / 2;
  if (!pointInPolygon(x, y, calibration.roi)) return null;
  const matrix = calibration.homography;
  const divisor = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  if (Math.abs(divisor) < 1e-8) return null;
  const position = [(matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / divisor, (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / divisor];
  return position.every(Number.isFinite) && position[0] >= 0 && position[0] <= calibration.pitch_length_m && position[1] >= 0 && position[1] <= calibration.pitch_width_m ? position : null;
}

function drawPitch(observations) {
  const calibration = activeCalibration();
  const definition = activePitch();
  const spec = definition?.spec || { model: element("pitch-model").value, length_m: Number(element("pitch-length").value), width_m: Number(element("pitch-width").value) };
  const { length_m: length, width_m: width } = spec;
  const portrait = spec.model === "eight";
  const displayWidth = portrait ? width : length;
  const displayHeight = portrait ? length : width;
  const displayPoints = (points) => points.map((point) => pitchDisplayPoint(point, spec).join(",")).join(" ");
  const pitch = element("pitch");
  pitch.replaceChildren();
  pitch.classList.toggle("portrait", portrait);
  pitch.setAttribute("viewBox", portrait ? `-5 -7 ${displayWidth + 10} ${displayHeight + 14}` : `-4 -4 ${displayWidth + 8} ${displayHeight + 8}`);
  pitchShape("rect", { x: -5, y: -7, width: displayWidth + 10, height: displayHeight + 14, fill: "#214e37" });
  for (let index = 0; index < 10; index++) {
    const start = index * length / 10;
    const end = (index + 1) * length / 10;
    pitchShape("polygon", { points: displayPoints([[start, 0], [end, 0], [end, width], [start, width]]), fill: index % 2 ? "#2d5a3d" : "#285539" });
  }
  const line = { fill: "none", stroke: "#83a17c", "stroke-width": 0.32 };
  for (const curve of definition?.lines || []) pitchShape("polyline", { points: displayPoints(curve.points), ...line });
  pitchShape("circle", { cx: displayWidth / 2, cy: displayHeight / 2, r: 0.6, fill: "#83a17c" });
  if (!calibration && !manualMapEditing()) {
    pitchShape("text", { x: displayWidth / 2, y: displayHeight / 2 + 18, "text-anchor": "middle", fill: "#b6ccab", "font-size": 3 }, "等待手动标定");
  }
  for (const observation of observations) {
    if (!observation.position || observation.reason === "excluded") continue;
    const [worldX, worldY] = observation.position;
    if (worldX < 0 || worldX > length || worldY < 0 || worldY > width) continue;
    const [horizontal, vertical] = pitchDisplayPoint(observation.position, spec);
    const selected = observation.row.track_id === state.selected;
    const group = svgNode("g", { class: "pitch-point", "data-track-id": observation.row.track_id, tabindex: 0, role: "button", "aria-label": `选中轨迹 ${observation.row.track_id}` });
    if (selected) group.append(svgNode("circle", { cx: horizontal, cy: vertical, r: 2.8, fill: "none", stroke: "white", "stroke-width": 0.4 }));
    group.append(svgNode("circle", { cx: horizontal, cy: vertical, r: selected ? 1.8 : 1.4, fill: observation.reason ? "#214e37" : observation.color, stroke: observation.color, "stroke-width": 0.6 }));
    group.append(svgNode("text", { x: horizontal + 2, y: vertical - 1.2, fill: "#e5efdd", "font-size": 2.6, "font-family": "monospace" }, observation.row.track_id));
    group.addEventListener("click", () => selectTrack(observation.row.track_id));
    group.addEventListener("keydown", (event) => { if (event.key === "Enter") selectTrack(observation.row.track_id); });
    pitch.append(group);
  }
  if (portrait) {
    pitchShape("text", { x: displayWidth / 2, y: -2.5, "text-anchor": "middle", fill: "#d8efe6", "font-size": 2.5 }, "远端");
    pitchShape("text", { x: displayWidth / 2, y: displayHeight + 4, "text-anchor": "middle", fill: "#d8efe6", "font-size": 2.5 }, "近端 / 相机侧");
  } else {
    pitchShape("text", { x: 0, y: width + 2.8, fill: "#89a47e", "font-size": 1.7 }, `0,0  → x / ${length} × ${width} m`);
  }
  const football = ballPosition();
  if (football) {
    const [x, y] = pitchDisplayPoint(football, spec);
    pitchShape("polygon", { points: `${x - 1.6},${y - 3} ${x + 1.6},${y - 3} ${x},${y}`, fill: "#ffea42", stroke: "#172920", "stroke-width": 0.45, "data-ball-marker": "true" });
  }
  if (definition && manualMapEditing()) drawLandmarks(definition);
}

function drawLandmarks(definition) {
  for (const landmark of definition.landmarks) {
    const [horizontal, vertical] = pitchDisplayPoint(landmark.pitch, definition.spec);
    const selected = state.selectedLandmark === landmark.id;
    const paired = state.points.findIndex((point) => point.landmark_id === landmark.id);
    const label = landmarkLabel(landmark);
    const group = svgNode("g", { role: "button", tabindex: 0, "aria-label": label, "aria-pressed": selected, "data-landmark": landmark.id, class: "reference-point" });
    group.append(svgNode("title", {}, label));
    group.append(svgNode("circle", { cx: horizontal, cy: vertical, r: 1.8, fill: "transparent", "pointer-events": "all" }));
    group.append(svgNode("circle", { cx: horizontal, cy: vertical, r: selected ? 1.5 : 0.9, fill: selected ? "#ffad48" : paired >= 0 ? "#ffd58f" : "#d8efe6", stroke: selected ? "#fff" : "#214e37", "stroke-width": 0.35 }));
    if (paired >= 0) group.append(svgNode("text", { x: horizontal + (horizontal > definition.spec.width_m / 2 && definition.spec.model === "eight" ? -1.5 : 1.5), y: vertical - 1.3, "text-anchor": definition.spec.model === "eight" && horizontal > definition.spec.width_m / 2 ? "end" : "start", fill: "#ffd58f", "font-size": 2.7, "font-weight": "bold", "paint-order": "stroke", stroke: "#214e37", "stroke-width": 0.6 }, paired + 1));
    const choose = () => chooseLandmark(landmark.id).catch((error) => notify(error.message, true));
    group.addEventListener("click", choose);
    group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); } });
    element("pitch").append(group);
  }
}

function detectionKey(row) {
  return [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2, row.confidence].join(":");
}

function drawVideo(observations) {
  const canvas = element("video-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) context.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  if (state.whiteImage && element("show-white-lines").checked && activeCalibration()?.mode === "white_lines") {
    context.globalAlpha = 0.7;
    context.drawImage(state.whiteImage, 0, 0, canvas.width, canvas.height);
    context.globalAlpha = 1;
  }
  const rawDetections = state.detectionRows.get(state.frame) || [];
  const matched = new Set(observations.map((observation) => detectionKey(observation.row)));
  const unmatched = rawDetections.filter((row) => !matched.has(detectionKey(row)));
  element("detection-summary").textContent = state.data.detections == null ? "旧运行没有原始检测缓存；重新执行 track 后可区分漏检与关联失败。" : `检测 ${rawDetections.length} · 已关联 ${observations.length} · 未关联 ${unmatched.length}（包含场外人员与噪声）。红色虚线：有检测但未形成轨迹；可见人物无任何框：检测阶段漏检。`;
  if (element("show-unmatched").checked) {
    context.strokeStyle = "#ff7575"; context.fillStyle = "#ff7575";
    context.lineWidth = 2; context.setLineDash([6, 4]); context.font = "17px sans-serif";
    for (const row of unmatched) {
      context.strokeRect(row.bbox_x1, row.bbox_y1, row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1);
      context.fillText(`未关联 ${row.confidence.toFixed(2)}`, row.bbox_x1, Math.max(18, row.bbox_y1 - 5));
    }
    context.setLineDash([]);
  }
  const calibration = calibrationAt(state.frame);
  if (calibration && element("show-pitch-model").checked && state.frame >= calibration.start_frame && state.frame < calibration.end_frame) {
    context.strokeStyle = "#d4eac488";
    context.lineWidth = 2;
    context.setLineDash([10, 8]);
    context.beginPath();
    calibration.roi.forEach(([horizontal, vertical], index) => index ? context.lineTo(horizontal, vertical) : context.moveTo(horizontal, vertical));
    context.closePath(); context.stroke(); context.setLineDash([]);
    const inverse = invert3(calibration.homography);
    if (inverse) {
      context.strokeStyle = "#ffe55a";
      for (const { points } of activePitch()?.lines || []) {
        context.beginPath();
        let connected = false;
        points.forEach(([horizontal, vertical], index) => {
          const divisor = inverse[2][0] * horizontal + inverse[2][1] * vertical + inverse[2][2];
          const pixelX = (inverse[0][0] * horizontal + inverse[0][1] * vertical + inverse[0][2]) / divisor;
          const pixelY = (inverse[1][0] * horizontal + inverse[1][1] * vertical + inverse[1][2]) / divisor;
          if (!Number.isFinite(pixelX) || !Number.isFinite(pixelY) || Math.abs(pixelX) > 1e6 || Math.abs(pixelY) > 1e6) { connected = false; return; }
          if (connected) context.lineTo(pixelX, pixelY); else context.moveTo(pixelX, pixelY);
          connected = true;
        });
        context.stroke();
      }
    }
  }
  if (element("show-boxes").checked) for (const observation of observations) {
    const row = observation.row;
    const selected = row.track_id === state.selected;
    context.strokeStyle = observation.reason === "excluded" ? "#fb7474" : observation.color;
    context.lineWidth = selected ? 5 : 2;
    context.strokeRect(row.bbox_x1, row.bbox_y1, row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1);
    const label = `#${row.track_id}${observation.binding?.player_id ? ` · ${observation.binding.player_id}` : ""}${element("show-unmatched").checked ? ` ${row.confidence.toFixed(2)}` : ""}`;
    context.font = "19px sans-serif";
    const labelWidth = context.measureText(label).width + 12;
    context.fillStyle = selected ? "#f3ffed" : "#092d25dd";
    context.fillRect(row.bbox_x1, Math.max(0, row.bbox_y1 - 27), labelWidth, 25);
    context.fillStyle = selected ? "#123d2f" : observation.color;
    context.fillText(label, row.bbox_x1 + 6, Math.max(20, row.bbox_y1 - 8));
    context.beginPath(); context.arc((row.bbox_x1 + row.bbox_x2) / 2, row.bbox_y2, selected ? 6 : 4, 0, Math.PI * 2); context.fill();
  }
  const football = currentBall();
  if (football) {
    const x = (football[0] + football[2]) / 2;
    const y = (football[1] + football[3]) / 2;
    const size = Math.max(12, canvas.width / 100);
    context.beginPath(); context.moveTo(x - size / 2, y - size - 3); context.lineTo(x + size / 2, y - size - 3); context.lineTo(x, y); context.closePath();
    context.fillStyle = "#ffea42"; context.strokeStyle = "#18211b"; context.lineWidth = Math.max(2, size / 10); context.fill(); context.stroke();
  }
  if (state.frame === Number(element("calibration-reference").value) && (state.pointMode || !element("calibration-tab").hidden)) state.points.forEach((point, index) => {
    context.fillStyle = "#ffd58f"; context.beginPath(); context.arc(point.image[0], point.image[1], 7, 0, Math.PI * 2); context.fill();
    context.font = "bold 24px sans-serif"; context.fillText(String(index + 1), point.image[0] + 10, point.image[1] - 10);
  });
  if (element("calibration-method").value === "manual" && !element("calibration-tab").hidden && state.roi.length) {
    context.strokeStyle = "#ffe48e"; context.lineWidth = 3; context.beginPath();
    state.roi.forEach(([horizontal, vertical], index) => index ? context.lineTo(horizontal, vertical) : context.moveTo(horizontal, vertical));
    if (!state.roiMode) context.closePath();
    context.stroke();
  }
}

function invert3(matrix) {
  const determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  if (Math.abs(determinant) < 1e-12) return null;
  const inverse = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) inverse[column][row] = (matrix[(row + 1) % 3][(column + 1) % 3] * matrix[(row + 2) % 3][(column + 2) % 3] - matrix[(row + 1) % 3][(column + 2) % 3] * matrix[(row + 2) % 3][(column + 1) % 3]) / determinant;
  return inverse;
}

function displayName(playerId) {
  return state.data.players.find((player) => player.player_id === playerId)?.display_name || playerId;
}

function draw() {
  if (!state.data) return;
  element("export-download").hidden = true;
  const observations = (state.frameRows.get(state.frame) || []).map(project);
  drawVideo(observations);
  drawPitch(observations);
  const ballRecord = ball.rows.get(state.frame);
  element("ball-result-status").textContent = !ball.result ? "当前模型尚无结果。点击运行足球追踪；其他模型的已有结果可直接切换查看。" : `本帧 ${ballRecord?.selected ? `▼ 检出 · 分数 ${ballRecord.selected[4].toFixed(2)}` : "缺失，不显示标记"} · 原始候选 ${ballRecord?.candidates.length || 0} · ${ball.result.manifest.observed_frames}/${ball.result.frames.length} 帧有选中观测（不代表准确率）。倒三角尖端指向球心；地图为地面近似投影。`;
  const calibration = activeCalibration();
  if (calibration?.mode === "white_lines") {
    const record = calibration.frames?.[String(state.frame)];
    const valid = record?.status === "valid";
    element("stat-calibration").textContent = valid ? "逐帧有效" : "本帧失效";
    element("pitch-mode").textContent = valid ? calibration.reviewed ? "逐帧米制投影" : "逐帧 · 待验收" : "白线不足 · 暂停投影";
    element("calibration-detail").textContent = valid ? `白线支持 ${(record.support * 100).toFixed(1)}% · 中位误差 ${record.error_px.toFixed(1)} px` : "本帧不使用上一帧标定";
    element("white-line-status").textContent = `青色：本帧提取的白色线索（可能包含白衣）；黄色：匹配的球场模型。源帧 ${state.frame} · ${valid ? `有效 · ${record.curve_count} 条曲线 · P90 误差 ${record.p90_error_px.toFixed(1)} px` : "几何匹配失败，隐藏模型与球员投影"}。`;
    element("white-line-status").classList.toggle("calibration-warning", !valid);
  } else {
    element("stat-calibration").textContent = calibration ? calibration.reviewed ? "固定 · 已确认" : "固定 · 预览" : "待标定";
    element("pitch-mode").textContent = calibration ? calibration.reviewed ? "固定米制投影" : "标定预览 · 未确认" : "等待标定预览";
    element("calibration-detail").textContent = calibration ? `${calibration.image_points.length} 对点 · 残差 ${calibration.fit_rmse_m.toFixed(3)} m` : "选择模型参考点并在原图配对";
    element("white-line-status").textContent = calibration ? `黄色：手动固定场地模型 · 源帧 [${calibration.start_frame}, ${calibration.end_frame})。不依赖逐帧白线识别。` : "当前没有有效标定预览；请完成参考点配对并点击预览。";
    element("white-line-status").classList.remove("calibration-warning");
  }
  element("stat-observations").textContent = observations.length;
  element("visible-count").textContent = observations.length;
  element("stat-bound").textContent = observations.filter((observation) => observation.binding?.action === "bind").length;
  element("track-list").replaceChildren(...observations.map((observation) => {
    const button = make("button", undefined, `track-item${observation.row.track_id === state.selected ? " selected" : ""}`);
    const marker = make("span", undefined, "marker"); marker.style.background = observation.color;
    button.append(marker, make("span", observation.binding?.player_id ? displayName(observation.binding.player_id) : `轨迹 #${observation.row.track_id}`), make("small", `#${observation.row.track_id}`));
    button.addEventListener("click", () => selectTrack(observation.row.track_id));
    return button;
  }));
  const selected = observations.find((observation) => observation.row.track_id === state.selected);
  element("selected-name").textContent = selected?.binding?.player_id ? displayName(selected.binding.player_id) : state.selected ? `轨迹 #${state.selected}` : "选择一名球员";
  element("selected-id").textContent = state.selected ? `#${state.selected}` : "—";
  element("selected-position").textContent = selected?.position ? selected.position.map((coordinate) => coordinate.toFixed(2)).join(" / ") : "—";
  element("selected-reason").textContent = selected ? (selected.reason ? reasons[selected.reason] || selected.reason : "当前点有效；报告仅累加连续有效观测。") : state.selected ? "该轨迹在当前帧没有观测。" : "从画面、场地圆点或左侧列表选择。";
  const manifest = state.data.manifest;
  element("frame-label").textContent = `FRAME ${String(state.frame).padStart(5, "0")} / ${manifest.end_frame - 1}`;
  element("frame-slider").value = state.frame;
  element("frame-number").value = state.frame;
  element("time-label").textContent = `${((state.frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} / ${((manifest.end_frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} s`;
}

async function showFrame(frameIndex) {
  if (!state.data) return;
  if ((state.pointMode || state.roiMode) && frameIndex !== Number(element("calibration-reference").value)) return notify("点选期间锁定参考帧，请先结束点选再切换画面。", true);
  const manifest = state.data.manifest;
  const bounded = Math.min(manifest.end_frame - 1, Math.max(manifest.start_frame, frameIndex));
  const step = ball.result || state.data.calibration?.mode === "white_lines" ? 1 : manifest.config.stride;
  const aligned = manifest.start_frame + Math.floor((bounded - manifest.start_frame) / step) * step;
  const requestId = ++state.requestId;
  const image = new Image();
  const loaded = new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("无法读取视频帧，请检查源视频是否仍存在")); });
  image.src = api(`frames/${aligned}`);
  let whiteImage = null;
  const calibration = activeCalibration();
  if (calibration?.mode === "white_lines" && aligned >= calibration.start_frame && aligned < calibration.end_frame) {
    whiteImage = new Image();
    const whiteLoaded = new Promise((resolve, reject) => { whiteImage.onload = resolve; whiteImage.onerror = () => reject(new Error("白线识别缓存读取失败，请重新标定")); });
    whiteImage.src = `${api(`white-lines/${aligned}.png`)}?version=${encodeURIComponent(calibration.calibration_id)}`;
    await Promise.all([loaded, whiteLoaded]);
  } else await loaded;
  if (requestId !== state.requestId) return;
  state.frame = aligned; state.image = image; state.whiteImage = whiteImage; draw();
}

function selectTrack(trackId) {
  state.selected = trackId;
  element("binding-track").value = trackId;
  const catalog = state.data.catalog.find((entry) => entry.track_id === trackId);
  if (catalog) {
    element("binding-start").value = catalog.start_frame;
    element("binding-end").value = catalog.end_frame;
  }
  draw();
}

function stopPlayback() {
  state.playing = false; element("play").textContent = "▶"; element("play").setAttribute("aria-label", "播放");
}

async function play() {
  if (state.playing) return stopPlayback();
  if (!state.data || state.pointMode || state.roiMode) return;
  state.playing = true; element("play").textContent = "Ⅱ"; element("play").setAttribute("aria-label", "暂停");
  while (state.playing) {
    const manifest = state.data.manifest;
    const sampling = ball.result || state.data.calibration?.mode === "white_lines" ? 1 : manifest.config.stride;
    const step = Math.max(sampling, Math.round(manifest.video.fps / 10 * Number(element("playback-rate").value) / sampling) * sampling);
    if (state.frame + step >= manifest.end_frame) { stopPlayback(); break; }
    try { await showFrame(state.frame + step); } catch (error) { notify(error.message, true); stopPlayback(); }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

function renderBindings() {
  element("binding-player").replaceChildren(option("", "选择球员档案"), ...state.data.players.map((player) => option(player.player_id, `${player.display_name} / ${player.player_id}`)));
  const entries = state.data.bindings.bindings;
  const nodes = entries.map((entry, index) => {
    const row = make("tr");
    row.append(make("td", `#${entry.track_id}`), make("td", `[${entry.start_frame}, ${entry.end_frame})`), make("td", entry.player_id ? displayName(entry.player_id) : "—"), make("td", entry.action === "exclude" ? "已排除" : entry.team_id));
    const cell = make("td"); const remove = make("button", "移除", "remove-button");
    remove.addEventListener("click", async () => {
      try { await request(api("bindings"), { run_id: state.runId, bindings: entries.filter((_, entryIndex) => entryIndex !== index) }); await loadRun(state.runId, true); notify("区间已移除；重新生成报告可更新统计。"); } catch (error) { notify(error.message, true); }
    });
    cell.append(remove); row.append(cell); return row;
  });
  if (!nodes.length) { const row = make("tr"); const cell = make("td", "尚未绑定。先建立档案，再从画面选择轨迹。", "empty-row"); cell.colSpan = 5; row.append(cell); nodes.push(row); }
  element("binding-table").replaceChildren(...nodes);
}

function markDraftDirty() {
  state.draftDirty = true;
  state.preview = null;
  state.draftVersion++;
  element("calibration-reviewed").checked = false;
  if (element("calibration-method").value === "manual") element("save-calibration").disabled = true;
}

function renderCalibrationPoints() {
  element("calibration-points").replaceChildren(...state.points.map((point, index) => {
    const landmark = state.draftPitch?.landmarks.find((entry) => entry.id === point.landmark_id);
    const row = make("tr"); row.append(make("td", String(index + 1)), make("td", landmarkLabel(landmark)), make("td", point.image.map(Math.round).join(" / ")));
    const cell = make("td"); const remove = make("button", "×", "remove-button");
    remove.setAttribute("aria-label", `删除配对 ${index + 1}`);
    remove.addEventListener("click", () => { state.points.splice(index, 1); markDraftDirty(); renderCalibrationPoints(); draw(); });
    if (point.landmark_id) {
      const replace = make("button", "重选", "remove-button");
      replace.addEventListener("click", () => chooseLandmark(point.landmark_id).catch((error) => notify(error.message, true)));
      cell.append(replace);
    }
    cell.append(remove); row.append(cell); return row;
  }));
  renderReferenceControls();
}

function updateEditing() {
  const locked = state.pointMode || state.roiMode;
  for (const id of ["play", "previous-frame", "next-frame", "frame-slider", "frame-number", "jump-frame"]) element(id).disabled = locked;
  element("point-mode").textContent = state.pointMode ? "结束点选" : "开始点选图像点";
  element("roi-mode").textContent = state.roiMode ? "结束区域点选" : "点选可靠区域";
}

function updateCalibrationUI() {
  const manual = element("calibration-method").value === "manual";
  element("calibration-method").querySelector('[value="white_lines"]').disabled = element("pitch-model").value === "eight";
  element("manual-calibration").hidden = !manual;
  element("reference-controls").hidden = !manualMapEditing();
  element("auto-calibration").hidden = manual;
  element("next-calibration-failure").hidden = manual;
  element("save-calibration").textContent = manual ? "确认并保存标定" : "保存验收状态";
  element("save-calibration").disabled = manual && (state.points.length < 4 || (!state.preview && state.draftDirty));
  if (!manual) { state.pointMode = false; state.roiMode = false; }
  updateEditing();
  renderReferenceControls();
}

function renderReferenceControls() {
  const definition = state.draftPitch;
  element("landmark-select").replaceChildren(option("", "点击地图圆点或选择名称"), ...(definition?.landmarks || []).map((point) => option(point.id, landmarkLabel(point))));
  element("landmark-select").value = state.selectedLandmark || "";
  const selected = definition?.landmarks.find((point) => point.id === state.selectedLandmark);
  element("landmark-status").textContent = selected ? `待选：${landmarkLabel(selected)}。请在原图点击对应地面点。` : `已配对 ${state.points.length} 点。点击上方地图参考点，再点击原图；橙色表示当前选择。`;
}

async function chooseLandmark(identifier) {
  if (!state.data) return;
  stopPlayback();
  state.selectedLandmark = identifier;
  state.pointMode = false; state.roiMode = false;
  if (state.frame !== Number(element("calibration-reference").value)) await showFrame(Number(element("calibration-reference").value));
  state.pointMode = true;
  updateEditing(); renderReferenceControls(); draw();
}

async function refreshDraftPitch(reset = false) {
  if (!state.data) return;
  markDraftDirty();
  state.draftPitch = null;
  if (reset) { state.points = []; state.roi = []; state.selectedLandmark = null; }
  renderCalibrationPoints(); updateCalibrationUI(); draw();
  const requestId = ++state.geometryRequest;
  const model = element("pitch-model").value;
  const query = new URLSearchParams({model, length: element("pitch-length").value, width: element("pitch-width").value});
  const result = await request(`/api/pitch-models?${query}`);
  if (requestId !== state.geometryRequest) return;
  state.draftPitch = result.models[0];
  renderCalibrationPoints(); updateCalibrationUI(); draw();
}

function manualCalibrationPayload() {
  if (!state.draftPitch) throw new Error("请先填写有效的场地尺寸。");
  const payload = { pitch_model: element("pitch-model").value, pitch_length_m: state.draftPitch.spec.length_m, pitch_width_m: state.draftPitch.spec.width_m, reference_frame: Number(element("calibration-reference").value), start_frame: Number(element("calibration-start").value), end_frame: Number(element("calibration-end").value), reviewed: element("calibration-reviewed").checked, note: "模型参考点手动固定标定" };
  if (state.points.every((point) => point.landmark_id)) payload.point_pairs = state.points.map((point) => ({landmark_id: point.landmark_id, image: point.image}));
  else { payload.image_points = state.points.map((point) => point.image); payload.pitch_points = state.points.map((point) => point.pitch); }
  if (state.roi.length) payload.roi = state.roi;
  return payload;
}

function renderStatistics() {
  const report = state.data.latest_report;
  element("open-report").hidden = !report;
  if (report) element("open-report").href = `/reports/${state.runId}/${report.report_id}/report.html`;
  element("report-note").textContent = report ? report.stale ? "身份、标定或档案已修改，以下为旧版结果，请重新生成报告。" : `已生成 ${report.report_id}。结果为已分析片段的有效观测。` : "完成场地标定确认和球员绑定后，即可生成报告。";
  element("statistics-table").replaceChildren(...(report?.summary.players || []).map((summary) => {
    const row = make("tr");
    row.append(make("td", displayName(summary.player_id)), make("td", summary.distance_m === null ? "数据不足" : `${summary.distance_m.toFixed(2)} m`), make("td", `${summary.valid_seconds.toFixed(2)} s`), make("td", `${(summary.coverage * 100).toFixed(1)}%`), make("td", summary.avg_speed_mps === null ? "—" : `${(summary.avg_speed_mps * 3.6).toFixed(2)} km/h`));
    row.addEventListener("click", () => {
      const chart = make("img"); chart.alt = `${displayName(summary.player_id)} 的速度曲线与热图`;
      const encoded = [...new TextEncoder().encode(summary.player_id)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
      chart.src = `/reports/${state.runId}/${report.report_id}/${encoded}.png`; element("player-chart").replaceChildren(chart);
    }); return row;
  }));
  element("player-chart").replaceChildren();
}

async function loadRun(runId, preserve = false) {
  stopPlayback();
  ball.requestId++; ball.result = null; ball.rows.clear();
  state.data = null; state.image = null; state.whiteImage = null;
  for (const id of ["start-ball", "ball-model", "open-calibration-step", "next-ball-missing"]) element(id).disabled = true;
  state.pointMode = false; state.roiMode = false; state.preview = null; state.selectedLandmark = null;
  state.geometryRequest++; state.draftVersion++;
  const oldFrame = state.frame;
  state.requestId++;
  state.runId = runId;
  const runRequestId = ++state.runRequestId;
  const data = await request(api());
  if (runRequestId !== state.runRequestId) return;
  state.data = data;
  element("open-calibration-step").disabled = false;
  const lastModel = localStorage.getItem("football-model");
  const resultModel = data.ball_versions?.[0]?.model_id;
  const preferredModel = ball.models.some((entry) => entry.id === lastModel) ? lastModel : resultModel || "football_v11n_tile";
  element("ball-model").value = preferredModel;
  state.frameRows.clear(); state.detectionRows.clear(); state.motionRows.clear();
  for (const row of data.tracks) { if (!state.frameRows.has(row.frame_index)) state.frameRows.set(row.frame_index, []); state.frameRows.get(row.frame_index).push(row); }
  for (const row of data.detections || []) { if (!state.detectionRows.has(row.frame_index)) state.detectionRows.set(row.frame_index, []); state.detectionRows.get(row.frame_index).push(row); }
  element("show-unmatched").disabled = data.detections == null;
  if (data.detections == null) element("show-unmatched").checked = false;
  for (const row of data.latest_report?.motion || []) state.motionRows.set(`${row.frame_index}:${row.track_id}`, row);
  const manifest = data.manifest;
  const calibration = data.calibration;
  state.draftPitch = data.pitch_definition;
  element("pitch-model").value = data.pitch_definition.spec.model;
  element("calibration-method").value = calibration ? calibration.mode === "white_lines" ? "white_lines" : "manual" : manifest.config.calibration_method || (data.pitch_definition.spec.model === "eight" ? "manual" : "white_lines");
  const sourceStep = 1;
  const slider = element("frame-slider"); slider.min = manifest.start_frame; slider.max = manifest.end_frame - 1; slider.step = sourceStep;
  element("frame-number").min = manifest.start_frame; element("frame-number").max = manifest.end_frame - 1; element("frame-number").step = sourceStep;
  element("video-canvas").width = manifest.video.width; element("video-canvas").height = manifest.video.height;
  element("source-description").textContent = `${manifest.config.clip_id} · ${manifest.config.match_id} · ${data.catalog.length} 条轨迹`;
  element("video-spec").textContent = `${manifest.video.width}×${manifest.video.height} · ${manifest.video.fps.toFixed(0)} FPS`;
  element("stat-duration").textContent = `${((manifest.end_frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} s`;
  element("stat-frame-info").textContent = `源帧 [${manifest.start_frame}, ${manifest.end_frame}) · stride ${manifest.config.stride}`;
  element("stat-calibration").textContent = calibration ? calibration.reviewed ? "已确认" : "草稿预览" : "未标定";
  element("calibration-detail").textContent = calibration?.mode === "white_lines" ? "逐帧白线几何匹配" : calibration ? `${calibration.image_points.length} 对点 · 拟合残差 ${calibration.fit_rmse_m.toFixed(3)} m` : "先建立图像点与场地对应关系";
  element("pitch-mode").textContent = calibration ? calibration.reviewed ? "米制投影" : "草稿 · 未确认" : "等待标定";
  element("pitch-length").value = calibration?.pitch_length_m || manifest.config.pitch_length_m;
  element("pitch-width").value = calibration?.pitch_width_m || manifest.config.pitch_width_m;
  element("calibration-reference").value = calibration?.reference_frame ?? manifest.start_frame;
  element("calibration-start").value = calibration?.start_frame ?? manifest.start_frame;
  element("calibration-end").value = calibration?.end_frame ?? manifest.end_frame;
  element("calibration-reviewed").checked = calibration?.reviewed || false;
  element("calibration-fit").textContent = calibration?.mode === "white_lines" ? `${calibration.summary.valid_frames} / ${calibration.summary.frame_count} 帧匹配有效 · 耗时 ${calibration.summary.elapsed_s.toFixed(1)} s。需回看移动过程的线条重合情况；支持率不代表独立精度验收。` : calibration ? `拟合点残差 ${calibration.fit_rmse_m.toFixed(3)} m，仅用于检查点的拟合情况，不能证明轨迹精度。${calibration.note || ""}` : "尚无拟合结果。";
  state.points = calibration?.image_points ? calibration.image_points.map((image, index) => ({ image: [...image], pitch: [...calibration.pitch_points[index]], landmark_id: calibration.point_pairs?.[index]?.landmark_id || data.pitch_definition.landmarks.find((point) => point.pitch.every((value, axis) => Math.abs(value - calibration.pitch_points[index][axis]) < 1e-6))?.id })) : [];
  state.roi = calibration?.roi ? calibration.roi.map((point) => [...point]) : [];
  element("save-calibration").textContent = calibration?.mode === "white_lines" ? "保存验收状态" : "保存标定";
  element("show-white-lines").disabled = calibration?.mode !== "white_lines";
  element("next-calibration-failure").disabled = !Object.values(calibration?.frames || {}).some((record) => record.status !== "valid");
  state.draftDirty = false;
  if (!preserve) { state.selected = null; element("binding-start").value = manifest.start_frame; element("binding-end").value = manifest.end_frame; }
  renderBindings(); renderCalibrationPoints(); renderStatistics(); updateCalibrationUI();
  let ballError = null;
  try { await loadBallResult(false); } catch (error) { ballError = error; }
  if (runRequestId !== state.runRequestId) return;
  await showFrame(preserve ? oldFrame : manifest.start_frame);
  if (runRequestId !== state.runRequestId) return;
  notify(calibration?.mode === "white_lines" ? "逐帧白线结果已加载。播放移动过程，核对黄色模型与原始白线是否持续重合。" : calibration ? "固定标定已加载。点击球员，对照原图脚点与场地位置。" : "轨迹已加载。请在“场地标定”选择模型和方式，完成位置标定。");
  if (ballError) notify(`人物结果已加载，足球结果读取失败：${ballError.message}。可刷新重试。`, true);
  watchCalibration(state.runId).catch((error) => notify(error.message, true));
  watchBall().catch((error) => notify(error.message, true));
}

async function loadBallResult(redraw = true) {
  const runId = state.runId;
  const modelId = element("ball-model").value;
  const requestId = ++ball.requestId;
  ball.result = null; ball.rows.clear();
  element("next-ball-missing").disabled = true;
  if (redraw) draw();
  if (!runId || !modelId) return;
  const result = await request(`/api/runs/${encodeURIComponent(runId)}/ball?model=${encodeURIComponent(modelId)}`);
  if (requestId !== ball.requestId || state.runId !== runId || element("ball-model").value !== modelId) return;
  ball.result = result;
  for (const row of result?.frames || []) ball.rows.set(row.frame, row);
  element("next-ball-missing").disabled = !result?.frames.some((row) => !row.selected);
  element("frame-slider").step = result || state.data.calibration?.mode === "white_lines" ? 1 : state.data.manifest.config.stride;
  element("frame-number").step = element("frame-slider").step;
  if (redraw) draw();
}

async function watchBall() {
  clearTimeout(ball.timer);
  try {
    const job = await request("/api/ball-job");
    ball.running = job.status === "running";
    element("start-ball").disabled = ball.running || !state.data || !ball.models.find((entry) => entry.id === element("ball-model").value)?.available;
    element("ball-model").disabled = !state.data;
    element("start-ball").textContent = ball.running ? "足球追踪进行中…" : "运行足球追踪";
    element("ball-progress").hidden = !ball.running;
    element("ball-progress").max = Math.max(1, job.total_frames || 0);
    element("ball-progress").value = job.processed_frames || 0;
    const label = element("ball-job-status");
    label.classList.toggle("error", job.status === "failed");
    label.textContent = ball.running ? `${job.run_id} · ${job.stage === "ball_filtering" ? "跨帧筛选" : "逐帧检测"} · ${job.processed_frames}/${job.total_frames}。可继续复核已有结果。` : job.status === "failed" ? `足球追踪失败：${job.error}。已有结果仍可查看，可重试。` : job.status === "complete" ? `${job.run_id} · 足球追踪完成，选择对应模型查看。` : "为左侧当前运行追加足球追踪；场地标定可独立完成。";
    if (ball.running) ball.timer = setTimeout(watchBall, 1000);
    else if (job.status === "complete" && ball.handledId !== job.job_id) {
      ball.handledId = job.job_id;
      if (job.run_id === state.runId && job.model_id === element("ball-model").value) await loadBallResult();
    }
  } catch (error) {
    element("ball-job-status").textContent = `读取足球进度失败：${error.message}，正在重试。`;
    ball.timer = setTimeout(watchBall, 2000);
  }
}

function showClip() {
  const clip = analysis.clips.find((entry) => entry.clip_id === element("clip-select").value);
  const preview = element("clip-preview");
  preview.hidden = true;
  element("clip-preview-status").hidden = false;
  element("clip-preview-status").textContent = clip ? "正在加载首帧…" : "没有符合条件的素材";
  element("clip-detail").textContent = clip ? `${(clip.size_bytes / 1024 / 1024).toFixed(1)} MB · ${clip.tracking_runs ? `${clip.tracking_runs} 次人物运行，${clip.calibrated_runs} 次附带场地标定；可新建运行重新分析。` : "尚无已完成人物运行；仅白线验收不计入人物追踪。"}` : "可取消筛选查看已有素材，或将新视频放入 data/clips 后刷新。";
  element("start-analysis").disabled = analysis.running || !clip;
  if (clip) preview.src = `/api/clips/${encodeURIComponent(clip.clip_id)}/preview`;
  else preview.removeAttribute("src");
}

function renderClips() {
  const previous = element("clip-select").value;
  const clips = analysis.clips.filter((clip) => !element("unprocessed-only").checked || !clip.tracking_runs || clip.clip_id === analysis.activeClip);
  element("clip-select").replaceChildren(...clips.map((clip) => option(clip.clip_id, `${clip.clip_id} · ${clip.tracking_runs ? "有人物运行" : "未追踪人物"}`)));
  if (!clips.length) element("clip-select").append(option("", "没有符合条件的素材"));
  else if (clips.some((clip) => clip.clip_id === previous)) element("clip-select").value = previous;
  showClip();
}

async function refreshClips() {
  analysis.clips = await request("/api/clips");
  renderClips();
}

async function initialize(preferredId = null) {
  const [runs, clips, definitions, models] = await Promise.all([request("/api/runs"), request("/api/clips"), request("/api/pitch-models"), request("/api/ball-models")]);
  ball.models = models.models;
  element("ball-model").replaceChildren(...ball.models.map((entry) => {
    const node = option(entry.id, entry.label + (entry.available ? "" : " · 未安装权重"));
    return node;
  }));
  pitchModels = definitions.models;
  analysis.clips = clips;
  renderClips();
  const completed = runs.filter((run) => run.status === "complete");
  element("run-select").replaceChildren(...completed.map((run) => option(run.run_id, `${run.clip_id} · ${run.run_id.slice(-8)}`)));
  element("run-select").disabled = !completed.length;
  if (!completed.length) {
    element("run-select").append(option("", "尚无已完成运行"));
    notify("在上方选择视频，点击“开始人物追踪”，完成后分别进行场地标定与足球追踪。");
  } else {
    const candidate = preferredId || state.runId;
    const selectedId = completed.some((run) => run.run_id === candidate) ? candidate : completed[0].run_id;
    element("run-select").value = selectedId;
    await loadRun(selectedId);
  }
  await watchAnalysis();
}

async function watchAnalysis() {
  clearTimeout(analysis.timer);
  try {
    const job = await request("/api/analysis-job");
    analysis.running = job.status === "running";
    analysis.runId = job.run_id;
    analysis.activeClip = analysis.running ? job.clip_id : null;
    if (analysis.running) {
      if (!Array.from(element("clip-select").options).some((entry) => entry.value === job.clip_id)) renderClips();
      element("clip-select").value = job.clip_id;
      if (element("clip-preview").getAttribute("src") !== `/api/clips/${encodeURIComponent(job.clip_id)}/preview`) showClip();
      for (const [id, key] of [["analysis-model", "pitch_model"], ["analysis-method", "calibration_method"], ["analysis-length", "pitch_length_m"], ["analysis-width", "pitch_width_m"], ["analysis-imgsz", "imgsz"], ["analysis-device", "device"], ["analysis-start", "start_frame"], ["analysis-end", "end_frame"]]) element(id).value = job.config[key] ?? "";
    }
    element("start-analysis").disabled = analysis.running || !element("clip-select").value;
    element("start-analysis").textContent = analysis.running ? "正在追踪人物…" : "开始人物追踪";
    for (const id of ["clip-select", "unprocessed-only", "analysis-model", "analysis-method", "analysis-length", "analysis-width", "analysis-imgsz", "analysis-device", "analysis-start", "analysis-end"]) element(id).disabled = analysis.running;
    element("analysis-method").querySelector('[value="white_lines"]').disabled = element("analysis-model").value === "eight";
    element("open-analysis").hidden = !job.run_id || analysis.running;
    element("analysis-progress").hidden = !analysis.running;
    element("analysis-progress").max = Math.max(1, job.total_frames || 0);
    element("analysis-progress").value = job.processed_frames || 0;
    const label = element("analysis-status");
    label.classList.toggle("error", job.status === "failed");
    if (analysis.running) {
      const deviceName = job.config?.device === "gpu" ? "GPU 核显" : "CPU";
      label.textContent = `${job.clip_id} · ${analysisStages[job.stage] || job.stage} · ${job.processed_frames} / ${job.total_frames || "待确定"} 帧。正在用${deviceName}分析，可继续查看已有运行；刷新页面后会恢复进度。`;
      analysis.timer = setTimeout(watchAnalysis, 1000);
    } else if (job.status === "failed") {
      label.textContent = `${job.clip_id} · ${analysisStages[job.stage] || job.stage}失败：${job.error}${job.run_id ? "。人物缓存已保留，可打开结果并重新运行场地标定。" : "。可调整参数后重新开始。"}`;
    } else if (job.status === "complete") {
      label.textContent = `${job.clip_id} · 人物追踪完成。通过步骤 2 单独完成场地标定，或选择足球模型运行步骤 3。`;
      if (analysis.handledId !== job.job_id) {
        analysis.handledId = job.job_id;
        await initialize(job.run_id);
      }
    }
  } catch (error) {
    element("analysis-status").textContent = `读取分析进度失败：${error.message}，正在重试。`;
    analysis.timer = setTimeout(watchAnalysis, 2000);
  }
}

async function watchCalibration(runId) {
  clearTimeout(state.calibrationTimer);
  const job = await request(`/api/runs/${encodeURIComponent(runId)}/calibration-job`);
  if (runId !== state.runId) return;
  const running = job.status === "running";
  if (running) state.pendingCalibrations.add(runId);
  element("auto-calibration").disabled = running || element("pitch-model").value === "eight";
  element("auto-calibration").textContent = running ? "正在逐帧识别…" : "运行逐帧白线标定";
  element("calibration-progress").textContent = running ? `已处理 ${job.processed_frames} / ${job.total_frames} 帧；完成后自动载入。` : job.status === "failed" ? `标定失败：${job.error}。可切换手动固定标定。` : "八人制采用独立参考点；固定标定保存后用于整个有效区间。";
  if (running) state.calibrationTimer = setTimeout(() => watchCalibration(runId).catch((error) => notify(error.message, true)), 1000);
  else if (state.pendingCalibrations.delete(runId) && job.status === "complete" && state.data.calibration?.calibration_id !== job.calibration_id) await loadRun(runId, true);
}

element("run-select").addEventListener("change", () => loadRun(element("run-select").value).catch((error) => notify(error.message, true)));
element("reload").addEventListener("click", () => initialize().catch((error) => notify(error.message, true)));
element("clip-select").addEventListener("change", showClip);
element("unprocessed-only").addEventListener("change", renderClips);
element("refresh-clips").addEventListener("click", () => refreshClips().catch((error) => notify(error.message, true)));
element("clip-preview").addEventListener("load", () => { element("clip-preview").hidden = false; element("clip-preview-status").hidden = true; });
element("clip-preview").addEventListener("error", () => { element("clip-preview").hidden = true; element("clip-preview-status").hidden = false; element("clip-preview-status").textContent = "首帧预览不可用，请检查视频是否可解码。"; });
element("open-analysis").addEventListener("click", () => initialize(analysis.runId).catch((error) => notify(error.message, true)));
function updateAnalysisOptions(reset = false) {
  if (reset) {
    const preset = pitchModels.find((definition) => definition.spec.model === element("analysis-model").value);
    if (preset) { element("analysis-length").value = preset.spec.length_m; element("analysis-width").value = preset.spec.width_m; }
    element("analysis-method").value = element("analysis-model").value === "eight" ? "manual" : "white_lines";
  }
  element("analysis-method").querySelector('[value="white_lines"]').disabled = element("analysis-model").value === "eight";
  if (!analysis.running) {
    element("start-analysis").textContent = "开始人物追踪";
    element("analysis-status").textContent = "此步仅运行人物追踪。完成后，分别点击步骤 2 场地标定与步骤 3 足球追踪。";
  }
}
element("analysis-model").addEventListener("change", () => updateAnalysisOptions(true));
element("analysis-method").addEventListener("change", () => updateAnalysisOptions());
element("start-analysis").addEventListener("click", async () => {
  const endText = element("analysis-end").value;
  const payload = { clip_id: element("clip-select").value, pitch_model: element("analysis-model").value, calibration_method: element("analysis-method").value, pitch_length_m: Number(element("analysis-length").value), pitch_width_m: Number(element("analysis-width").value), imgsz: Number(element("analysis-imgsz").value), device: element("analysis-device").value, start_frame: Number(element("analysis-start").value), end_frame: endText === "" ? null : Number(endText) };
  if (!payload.clip_id) return;
  element("start-analysis").disabled = true;
  try {
    await request("/api/analysis", payload);
    await watchAnalysis();
  } catch (error) {
    notify(error.message, true);
    await watchAnalysis();
  }
});
element("show-boxes").addEventListener("change", draw);
element("show-ball").addEventListener("change", draw);
element("ball-model").addEventListener("change", async () => {
  localStorage.setItem("football-model", element("ball-model").value);
  stopPlayback();
  try { await loadBallResult(); await watchBall(); } catch (error) { notify(error.message, true); }
});
element("start-ball").addEventListener("click", async () => {
  if (!state.runId) return;
  element("start-ball").disabled = true;
  try { await request(api("ball"), { model_id: element("ball-model").value }); await watchBall(); }
  catch (error) { notify(error.message, true); await watchBall(); }
});
element("open-calibration-step").addEventListener("click", () => {
  document.querySelector('[data-tab="calibration"]').click();
  element("calibration-tab").scrollIntoView({ block: "start", behavior: "smooth" });
});
element("next-ball-missing").addEventListener("click", () => {
  const missing = ball.result?.frames.filter((row) => !row.selected) || [];
  const next = missing.find((row) => row.frame > state.frame) || missing[0];
  if (next) { stopPlayback(); showFrame(next.frame).catch((error) => notify(error.message, true)); }
});
element("show-unmatched").addEventListener("change", draw);
element("show-white-lines").addEventListener("change", draw);
element("show-pitch-model").addEventListener("change", draw);
element("auto-calibration").addEventListener("click", async () => {
  if (!state.data) return;
  stopPlayback();
  const runId = state.runId;
  element("auto-calibration").disabled = true;
  try {
    await request(api("calibration-auto"), { pitch_model: element("pitch-model").value, pitch_length_m: Number(element("pitch-length").value), pitch_width_m: Number(element("pitch-width").value) });
    state.pendingCalibrations.add(runId);
    await watchCalibration(runId);
    notify("正在逐源帧识别白线并匹配场地几何，完成后可播放验收。");
  } catch (error) { element("auto-calibration").disabled = false; notify(error.message, true); }
});
element("next-calibration-failure").addEventListener("click", () => {
  const failures = Object.values(state.data?.calibration?.frames || {}).filter((record) => record.status !== "valid").map((record) => record.frame_index).sort((first, second) => first - second);
  if (!failures.length) return;
  stopPlayback();
  showFrame(failures.find((frame) => frame > state.frame) ?? failures[0]).catch((error) => notify(error.message, true));
});
element("export-frame").addEventListener("click", async () => {
  if (!state.data || !state.image) return;
  stopPlayback();
  const frameIndex = state.frame;
  const runId = state.runId;
  const selectedName = element("selected-name").textContent;
  const selectedDetails = `轨迹 ${element("selected-id").textContent} · 坐标 ${element("selected-position").textContent} m`;
  const calibrationNote = state.data.calibration?.reviewed ? "标定已确认 · 精度待独立验证" : "标定草稿 · 仅供位置核对";
  const canvas = document.createElement("canvas"); canvas.width = 1600; canvas.height = 760;
  const context = canvas.getContext("2d");
  context.fillStyle = "#f2f5f3"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#193632"; context.font = "bold 30px sans-serif"; context.fillText("TOUCHLINE / 球员追踪与位置对拍", 40, 55);
  context.font = "18px sans-serif"; context.fillStyle = "#668273";
  context.fillText(`${state.data.manifest.config.clip_id} · 源帧 ${state.frame} · 时间 ${(state.frame / state.data.manifest.video.fps).toFixed(2)} s`, 40, 92);
  context.drawImage(element("video-canvas"), 40, 130, 1000, 562.5);
  const pitch = element("pitch").cloneNode(true);
  const portrait = activePitch()?.spec.model === "eight";
  pitch.setAttribute("width", portrait ? "700" : "980"); pitch.setAttribute("height", portrait ? "880" : "680");
  const serialized = new XMLSerializer().serializeToString(pitch);
  const image = new Image();
  try {
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("图片生成超时")), 5000);
      image.onload = () => { clearTimeout(timeout); resolve(); };
      image.onerror = () => { clearTimeout(timeout); reject(new Error("球场图片生成失败")); };
      image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(serialized)}`;
    });
    context.drawImage(image, portrait ? 1140 : 1070, portrait ? 130 : 190, portrait ? 350 : 490, portrait ? 440 : 340);
    context.fillStyle = "#193632"; context.font = "20px sans-serif";
    context.fillText(selectedName, 1080, 590);
    context.font = "17px sans-serif"; context.fillStyle = "#668273";
    context.fillText(selectedDetails, 1080, 625);
    context.fillText(calibrationNote, 1080, 660);
    context.font = "16px sans-serif"; context.fillText("同帧对应；颜色为球队或待确认颜色簇，数字为本次运行的轨迹 ID。", 40, 735);
    if (state.frame !== frameIndex || state.runId !== runId || element("selected-name").textContent !== selectedName) return;
    const anchor = element("export-download"); anchor.download = `touchline-frame-${frameIndex}.png`; anchor.href = canvas.toDataURL("image/png"); anchor.hidden = false;
    notify(`源帧 ${frameIndex} 的对拍图已生成，点击“保存 PNG”下载。`);
  } catch { notify("导出失败，请重试。", true); }
});
element("play").addEventListener("click", play);
for (const [id, direction] of [["previous-frame", -1], ["next-frame", 1]]) element(id).addEventListener("click", () => { stopPlayback(); if (state.data) showFrame(state.frame + direction * (ball.result || state.data.calibration?.mode === "white_lines" ? 1 : state.data.manifest.config.stride)).catch((error) => notify(error.message, true)); });
element("frame-slider").addEventListener("input", () => { stopPlayback(); showFrame(Number(element("frame-slider").value)).catch((error) => notify(error.message, true)); });
element("frame-number").addEventListener("change", () => { stopPlayback(); showFrame(Number(element("frame-number").value)).catch((error) => notify(error.message, true)); });
element("jump-frame").addEventListener("click", () => { stopPlayback(); showFrame(Number(element("frame-number").value)).catch((error) => notify(error.message, true)); });
element("frame-number").addEventListener("keydown", (event) => { if (event.key === "Enter") element("jump-frame").click(); });
document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-tab]").forEach((tab) => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", tab === button ? "true" : "false"); });
  for (const name of ["identity", "calibration", "statistics"]) element(`${name}-tab`).hidden = button.dataset.tab !== name;
  if (button.dataset.tab !== "calibration") { state.pointMode = false; state.roiMode = false; }
  updateCalibrationUI();
  draw();
}));
element("point-mode").addEventListener("click", async () => {
  if (!state.data) return;
  state.roiMode = false; element("roi-mode").textContent = "点选可靠区域";
  stopPlayback(); state.pointMode = !state.pointMode;
  updateEditing();
  if (state.pointMode) {
    try { await showFrame(Number(element("calibration-reference").value)); notify("先点击球场位置地图的参考点，再点击原图对应地面位置；全部点来自同一参考帧。"); } catch (error) { state.pointMode = false; updateEditing(); notify(error.message, true); }
  }
});
element("roi-mode").addEventListener("click", async () => {
  if (!state.data) return;
  stopPlayback(); state.pointMode = false; element("point-mode").textContent = "开始点选图像点";
  state.roiMode = !state.roiMode; element("roi-mode").textContent = state.roiMode ? "结束区域点选" : "点选可靠区域";
  if (state.roiMode) {
    state.roi = []; markDraftDirty();
    try { await showFrame(Number(element("calibration-reference").value)); notify("按顺序点击可靠地面区域的边界（至少三点），完成后点“结束区域点选”。区域之外不计入统计。"); } catch (error) { state.roiMode = false; notify(error.message, true); }
  }
  updateEditing();
  draw();
});
element("video-canvas").addEventListener("click", (event) => {
  if (!state.data) return;
  const canvas = element("video-canvas"), rectangle = canvas.getBoundingClientRect();
  const horizontal = (event.clientX - rectangle.left) / rectangle.width * canvas.width;
  const vertical = (event.clientY - rectangle.top) / rectangle.height * canvas.height;
  if (state.roiMode) {
    if (state.frame !== Number(element("calibration-reference").value)) { notify("请回到参考帧后点选可靠区域。", true); return; }
    state.roi.push([Number(horizontal.toFixed(2)), Number(vertical.toFixed(2))]); markDraftDirty(); draw(); return;
  }
  if (state.pointMode) {
    if (state.frame !== Number(element("calibration-reference").value)) { notify("请回到参考帧后点选，不能混用不同帧的图像点。", true); return; }
    const landmark = state.draftPitch?.landmarks.find((point) => point.id === state.selectedLandmark);
    if (!landmark) return notify("请先在球场位置地图选择参考点。", true);
    const point = { image: [Number(horizontal.toFixed(2)), Number(vertical.toFixed(2))], pitch: [...landmark.pitch], landmark_id: landmark.id };
    const existing = state.points.findIndex((entry) => entry.landmark_id === landmark.id);
    if (existing >= 0) state.points[existing] = point; else state.points.push(point);
    state.selectedLandmark = null;
    markDraftDirty(); renderCalibrationPoints(); draw(); return;
  }
  const hits = (state.frameRows.get(state.frame) || []).filter((row) => row.bbox_x1 <= horizontal && row.bbox_x2 >= horizontal && row.bbox_y1 <= vertical && row.bbox_y2 >= vertical).sort((first, second) => (first.bbox_x2 - first.bbox_x1) * (first.bbox_y2 - first.bbox_y1) - (second.bbox_x2 - second.bbox_x1) * (second.bbox_y2 - second.bbox_y1));
  if (hits.length) selectTrack(hits[0].track_id);
});
element("player-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (!state.data) return;
  const playerId = element("player-id").value.trim();
  try { await request(api("players"), { player_id: playerId, display_name: element("player-name").value.trim() }); await loadRun(state.runId, true); element("binding-player").value = playerId; notify("球员档案已保存，可用于此后比赛。"); } catch (error) { notify(error.message, true); }
});
async function addBinding(action) {
  if (!state.data) return;
  const entry = { track_id: Number(element("binding-track").value), start_frame: Number(element("binding-start").value), end_frame: Number(element("binding-end").value), action };
  if (action === "bind") { entry.player_id = element("binding-player").value; entry.team_id = element("binding-team").value; }
  const bindings = [...state.data.bindings.bindings, entry];
  await request(api("bindings"), { run_id: state.runId, bindings });
  await loadRun(state.runId, true); notify("区间已保存。修改身份后只需重算报告，原始轨迹保持不变。");
}
element("binding-form").addEventListener("submit", (event) => { event.preventDefault(); addBinding("bind").catch((error) => notify(error.message, true)); });
element("exclude-track").addEventListener("click", () => addBinding("exclude").catch((error) => notify(error.message, true)));
for (const id of ["calibration-start", "calibration-end"]) element(id).addEventListener("input", () => { markDraftDirty(); draw(); });
for (const id of ["pitch-length", "pitch-width"]) {
  element(id).addEventListener("input", () => { state.geometryRequest++; state.draftPitch = null; state.points = []; state.roi = []; state.selectedLandmark = null; markDraftDirty(); renderCalibrationPoints(); draw(); });
  element(id).addEventListener("change", () => refreshDraftPitch(true).catch((error) => notify(error.message, true)));
}
element("pitch-model").addEventListener("change", () => {
  const preset = pitchModels.find((definition) => definition.spec.model === element("pitch-model").value);
  element("pitch-length").value = preset.spec.length_m; element("pitch-width").value = preset.spec.width_m;
  element("calibration-method").value = preset.spec.model === "eight" ? "manual" : "white_lines";
  state.pointMode = false; state.roiMode = false;
  refreshDraftPitch(true).catch((error) => notify(error.message, true));
});
element("calibration-method").addEventListener("change", () => { state.pointMode = false; state.roiMode = false; markDraftDirty(); updateCalibrationUI(); draw(); });
element("landmark-select").addEventListener("change", () => { if (element("landmark-select").value) chooseLandmark(element("landmark-select").value).catch((error) => notify(error.message, true)); });
element("calibration-reference").addEventListener("change", async () => {
  state.pointMode = false; state.roiMode = false; state.points = []; state.roi = []; state.selectedLandmark = null;
  markDraftDirty(); updateEditing(); renderCalibrationPoints();
  try { await showFrame(Number(element("calibration-reference").value)); element("calibration-reference").value = state.frame; } catch (error) { notify(error.message, true); }
});
element("use-current-frame").addEventListener("click", () => { element("calibration-reference").value = state.frame; element("calibration-reference").dispatchEvent(new Event("change")); document.querySelector(".viewer-grid").scrollIntoView({block: "start", behavior: "smooth"}); });
element("clear-roi").addEventListener("click", () => { state.roi = []; state.roiMode = false; markDraftDirty(); updateEditing(); draw(); });
element("discard-calibration").addEventListener("click", () => { if (state.runId) loadRun(state.runId, true).catch((error) => notify(error.message, true)); });
element("clear-points").addEventListener("click", () => { state.points = []; state.selectedLandmark = null; markDraftDirty(); renderCalibrationPoints(); draw(); });

element("preview-calibration").addEventListener("click", async () => {
  if (!state.data) return;
  const version = state.draftVersion;
  const runId = state.runId;
  state.pointMode = false; state.roiMode = false; updateEditing(); stopPlayback();
  try {
    const calibration = await request(api("calibration-preview"), manualCalibrationPayload());
    if (version !== state.draftVersion || runId !== state.runId) return;
    state.preview = calibration; state.draftDirty = true;
    element("save-calibration").disabled = false;
    element("calibration-fit").textContent = `${state.points.length} 对点 · ${calibration.inlier_count} 内点 · 拟合残差 ${calibration.fit_rmse_px.toFixed(2)} px。请用未参与标定的白线交点独立核对，并检查首、中、末帧。`;
    draw(); notify("标定预览已显示。可播放核对整段视频，检查后确认保存。少量或集中参考点的拟合残差不能证明整场精度。");
  } catch (error) { notify(error.message, true); }
});
element("save-calibration").addEventListener("click", async () => {
  if (!state.data) return;
  if (element("calibration-method").value === "white_lines") {
    const calibration = state.data.calibration;
    if (calibration?.mode !== "white_lines") return notify("请先运行逐帧白线标定。", true);
    if (Number(element("pitch-length").value) !== calibration.pitch_length_m || Number(element("pitch-width").value) !== calibration.pitch_width_m) return notify("尺寸已改变，请重新运行逐帧标定后再验收。", true);
    try {
      await request(api("calibration-review"), { calibration_id: calibration.calibration_id, reviewed: element("calibration-reviewed").checked });
      await loadRun(state.runId, true);
      notify("逐帧标定验收状态已保存；失败帧仍不参与场地投影与跑动统计。");
    } catch (error) { notify(error.message, true); }
    return;
  }
  if (state.draftDirty && !state.preview) return notify("请先预览当前参考点标定。", true);
  try {
    const payload = manualCalibrationPayload();
    await request(api("calibration"), payload); state.pointMode = false; state.roiMode = false;
    await loadRun(state.runId, true);
    notify(payload.reviewed ? "固定标定已确认，整段视频可查看投影并生成统计报告。" : "固定标定草稿已保存，刷新后可继续验收。");
  } catch (error) { notify(error.message, true); }
});
element("generate-report").addEventListener("click", async () => {
  if (!state.data) return;
  if (state.draftDirty) return notify("请先保存当前标定草稿，再生成报告。", true);
  const button = element("generate-report"); button.disabled = true; button.classList.add("busy"); button.textContent = "正在计算与渲染…"; stopPlayback();
  try { await request(api("report"), {}); await loadRun(state.runId, true); notify("新版本报告已生成。点击统计表中的球员查看速度曲线与热图。"); } catch (error) { notify(error.message, true); }
  finally { button.disabled = false; button.classList.remove("busy"); button.textContent = "生成 / 重算报告"; }
});
initialize().catch((error) => notify(error.message, true));
