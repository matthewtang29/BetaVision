// BetaVision web app: upload -> follow the camera -> mark the route -> analyze -> results.
// The hold picker mirrors the OpenCV picker in climb_holds.py; the server runs the
// same detection functions on every change and sends back the outlines.
"use strict";

const $ = (id) => document.getElementById(id);
const STEPS = ["step-upload", "step-holds", "step-run", "step-results", "step-error"];
const VIDEO_EXT = [".mp4", ".mov", ".avi"];
const COLORS = { auto: "#3fae5a", manual: "#e38b2c", start: "#1fb8c9", end: "#d0473b", vol: "#ebebeb", cut: "#ff69b4" };

let job = null;          // {id, name, frames, fps, width, height, pano, camera_moved}
let P = null;            // picker state sent to the server (see jobs.Job.build_route)
let holds = [], vols = [], skippedVols = 0;
let mode = "AUTO", tag = null, poly = [], history = [];
let img = new Image(), view = { s: 1, ox: 0, oy: 0, fit: 1 };
let pollTimer = null, jobId = null;

const STEP_NUM = { "step-upload": 1, "step-holds": 2, "step-results": 3 };

function show(step, num) {
  STEPS.forEach((s) => ($(s).hidden = s !== step));
  setSteps(num || STEP_NUM[step]);
  window.scrollTo({ top: 0 });
}

// header step indicator: earlier steps get a tick, the current one is filled
function setSteps(n) {
  if (!n) return;
  document.querySelectorAll(".steps .step").forEach((el) => {
    const i = +el.dataset.step;
    el.classList.toggle("done", i < n);
    el.classList.toggle("current", i === n);
  });
  document.querySelectorAll(".steps .dash").forEach((el) => el.classList.toggle("done", +el.dataset.after < n));
}
setSteps(1);

// light / dark toggle; with no saved choice the page follows the system setting
$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("bv-theme", root.dataset.theme); } catch (_) { /* storage blocked */ }
});

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Something went wrong (${r.status})`);
  return data;
}

function fail(msg) {
  clearInterval(pollTimer);
  $("error-message").textContent = msg;
  show("step-error");
}

// ------------------------------------------------------------------ 1. upload
const drop = $("drop"), fileInput = $("file");

["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(fileInput.files[0]); });
// a file dropped anywhere else shouldn't make the browser navigate away
["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => {
  if (!drop.contains(e.target)) e.preventDefault();
}));

function upload(file) {
  const err = $("upload-error");
  err.hidden = true;
  if (!VIDEO_EXT.some((x) => file.name.toLowerCase().endsWith(x))) {
    err.textContent = "That isn't a video BetaVision can read. Choose an .mp4, .mov or .avi file.";
    err.hidden = false;
    return;
  }
  drop.hidden = true;
  $("still-row").hidden = true;
  $("upload-progress").hidden = false;
  setBar("upload", 0, "Uploading…");

  const form = new FormData();
  form.append("file", file);
  form.append("still", $("still").checked ? "true" : "false");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const f = e.loaded / e.total;
    setBar("upload", f, f < 1 ? "Uploading…" : "Opening the video…");
  };
  xhr.onload = () => {
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
    if (xhr.status !== 200) return uploadFailed(data.detail || "The upload didn't work. Try again.");
    jobId = data.id;
    $("video-name").textContent = file.name;
    $("video-name").title = file.name;
    startPolling("Reading the wall…", 1);
  };
  xhr.onerror = () => uploadFailed("Lost connection to BetaVision. Is its window still open?");
  xhr.send(form);
}

function uploadFailed(msg) {
  drop.hidden = false;
  $("still-row").hidden = false;
  $("upload-progress").hidden = true;
  fileInput.value = "";
  $("upload-error").textContent = msg;
  $("upload-error").hidden = false;
}

function setBar(which, f, label) {
  const pct = Math.round(Math.max(0, Math.min(1, f)) * 100);
  $(`${which}-bar`).style.width = pct + "%";
  $(`${which}-pct`).textContent = pct + "%";
  if (label !== undefined) $(`${which}-label`).textContent = label;
}

// ------------------------------------------------------------------ status polling (preparing + analysis)
const STAGES = {
  camera: "Following the camera", stitch: "Stitching the wall", queued: "Waiting", model: "Downloading",
  lock: "Locking holds", track: "Tracking", draw: "Drawing", encode: "Encoding",
};

function startPolling(title, step) {
  $("run-title").textContent = title;
  $("run-message").textContent = "";
  $("run-stage").textContent = "Starting";
  setBar("run", 0);
  show("step-run", step);
  clearInterval(pollTimer);
  pollTimer = setInterval(poll, 600);
}

async function poll() {
  let s;
  try { s = await api(`/api/jobs/${jobId}`); } catch (e) { return fail(e.message); }
  $("run-stage").textContent = STAGES[s.stage] || "Working";
  $("run-message").textContent = s.message || "";
  setBar("run", ["stitch", "encode", "model", "queued"].includes(s.stage) ? (s.stage === "encode" ? 1 : 0) : s.progress);
  $("run-bar").parentElement.classList.toggle("busy", ["stitch", "encode", "model"].includes(s.stage));
  if (s.state === "error") fail(s.message);
  else if (s.state === "picking") { clearInterval(pollTimer); startPicker(s.info); }
  else if (s.state === "done") { clearInterval(pollTimer); showResults(s.results); }
}

// ------------------------------------------------------------------ 2. hold picker
const canvas = $("canvas"), ctx = canvas.getContext("2d");

function startPicker(info) {
  job = info;
  P = { frame: 0, tol: 22, min_size: 3, auto: [], manual: [], deleted: [], starts: [], end: null,
        vols_manual: [], vols_deleted: [] };
  holds = []; vols = []; history = []; poly = []; tag = null; setMode("AUTO");
  $("frame-field").hidden = job.pano;
  if (job.pano) $("video-name").textContent = "Whole wall (stitched)";
  $("frame").max = Math.max(0, job.frames - 1);
  $("frame").value = 0;
  $("tol").value = 22; $("minsize").value = 3;
  $("tol-label").textContent = 22; $("minsize-label").textContent = 3;
  $("lock-row").hidden = !job.camera_moved;
  $("notes-error").hidden = true;
  updateFrameLabel(0);
  show("step-holds");
  loadPicture(0, true);
}

function loadPicture(i, fit) {
  const next = new Image();
  next.onload = () => {
    img = next;
    P.frame = i;
    if (fit) fitView();
    draw();
    refreshRoute();
  };
  next.src = `/api/jobs/${job.id}/frame/${job.pano ? 0 : i}`;
}

function updateFrameLabel(i) {
  $("frame-label").textContent = `${i} · ${(i / job.fps).toFixed(1)} s`;
}

// ---------- view: zoom + pan (image px at the canvas's top-left = ox, oy)
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
}

function fitView() {
  resizeCanvas();
  const r = canvas.getBoundingClientRect();
  view.fit = Math.min(r.width / job.width, r.height / job.height);
  view.s = view.fit;
  clampView();
}

function clampView() {
  const r = canvas.getBoundingClientRect();
  const vw = r.width / view.s, vh = r.height / view.s;
  view.ox = vw >= job.width ? (job.width - vw) / 2 : Math.min(Math.max(view.ox, 0), job.width - vw);
  view.oy = vh >= job.height ? (job.height - vh) / 2 : Math.min(Math.max(view.oy, 0), job.height - vh);
}

function zoomAt(factor, cx, cy) {
  const r = canvas.getBoundingClientRect();
  if (cx === undefined) { cx = r.width / 2; cy = r.height / 2; }
  const ix = view.ox + cx / view.s, iy = view.oy + cy / view.s;
  view.s = Math.min(Math.max(view.s * factor, view.fit), view.fit * 8);
  view.ox = ix - cx / view.s; view.oy = iy - cy / view.s;
  clampView(); draw();
}

function toImage(e) {
  const r = canvas.getBoundingClientRect();
  const x = Math.round(view.ox + (e.clientX - r.left) / view.s);
  const y = Math.round(view.oy + (e.clientY - r.top) / view.s);
  return x >= 0 && y >= 0 && x < job.width && y < job.height ? [x, y] : null;
}

// ---------- drawing
function path(pts, close = true) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (close) ctx.closePath();
}

function draw() {
  if (!job) return;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const k = view.s * dpr;
  ctx.setTransform(k, 0, 0, k, -view.ox * k, -view.oy * k);
  ctx.imageSmoothingQuality = "high";
  ctx.lineJoin = "round";
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, job.width, job.height);

  for (const v of vols) {                        // volumes: white outline on a dark edge, cut-outs pink
    path(v.contour);
    ctx.lineWidth = 4 / view.s; ctx.strokeStyle = "rgba(30,30,30,0.85)"; ctx.stroke();
    ctx.lineWidth = 2 / view.s; ctx.strokeStyle = COLORS.vol; ctx.stroke();
    for (const e of v.exclude || []) {
      path(e); ctx.lineWidth = 2 / view.s; ctx.strokeStyle = COLORS.cut; ctx.stroke();
    }
  }
  for (const h of holds) {
    const color = h.role ? COLORS[h.role] : COLORS[h.source] || COLORS.auto;
    path(h.contour);
    if (h.role) { ctx.fillStyle = color + "33"; ctx.fill(); }
    ctx.lineWidth = (h.role ? 3 : 2) / view.s;
    ctx.strokeStyle = color;
    ctx.stroke();
  }
  for (const c of P.auto) {                      // where you clicked (AUTO samples)
    if (!job.pano && c.frame !== P.frame) continue;
    const r = 7 / view.s;
    ctx.lineWidth = 2 / view.s;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath(); ctx.moveTo(c.x - r, c.y); ctx.lineTo(c.x + r, c.y);
    ctx.moveTo(c.x, c.y - r); ctx.lineTo(c.x, c.y + r); ctx.stroke();
  }
  if (poly.length) {                             // volume corners clicked so far
    path(poly, false);
    ctx.lineWidth = 2 / view.s; ctx.strokeStyle = COLORS.vol; ctx.stroke();
    for (const [x, y] of poly) {
      ctx.beginPath(); ctx.arc(x, y, 4 / view.s, 0, 7);
      ctx.fillStyle = COLORS.vol; ctx.fill(); ctx.lineWidth = 1.5 / view.s; ctx.strokeStyle = "#222"; ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(poly[0][0], poly[0][1], 18 / view.s, 0, 7);
    ctx.lineWidth = 1 / view.s; ctx.strokeStyle = COLORS.vol; ctx.stroke();
  }
  // labels in screen pixels so they stay readable at any zoom
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = "700 12px Manrope, Segoe UI, system-ui, sans-serif";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,0,0,0.75)";
  const label = (text, x, y, color) => {
    const sx = (x - view.ox) * view.s, sy = (y - view.oy) * view.s;
    ctx.strokeText(text, sx, sy); ctx.fillStyle = color; ctx.fillText(text, sx, sy);
  };
  for (const v of vols) {
    const top = v.contour.reduce((a, b) => (b[1] < a[1] ? b : a));
    label(v.id, top[0] - 10, top[1] - 8, COLORS.vol);
  }
  for (const h of holds) {
    const text = h.id + (h.role === "start" ? " START" : h.role === "end" ? " END" : "");
    label(text, h.center[0] + 8 / view.s, h.center[1] + 4 / view.s, h.role ? COLORS[h.role] : "#ffffff");
  }
}

// ---------- geometry for clicks (same rules as climb_holds.py)
function insidePoly(p, [x, y]) {
  let c = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i], [xj, yj] = p[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
const inside = (h, pt) => insidePoly(h.contour, pt);

function edgeDistance(h, [x, y]) {
  let best = Infinity;
  const p = h.contour;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [ax, ay] = p[j], [bx, by] = p[i];
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L));
    best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
  }
  return inside(h, [x, y]) ? 0 : best;
}

let msgTimer = null;
function flash(msg) {
  const el = $("canvas-msg");
  el.textContent = msg; el.hidden = false;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => (el.hidden = true), 2800);
}

// size of one video frame (thresholds are relative to it, also on the stitched wall)
const frameMin = () => (job.pano ? Math.min(job.frame_h || job.height, job.frame_w || job.width) : Math.min(job.width, job.height));

function tagHold(pt) {
  let hit = holds.find((h) => inside(h, pt));
  if (!hit && holds.length) {                    // forgiving: nearest hold if the click was close
    const near = holds.reduce((a, b) => (edgeDistance(a, pt) <= edgeDistance(b, pt) ? a : b));
    if (edgeDistance(near, pt) <= 0.04 * frameMin()) hit = near;
  }
  if (!hit) { flash("No hold there. Click inside an outline."); return; }
  const p = inside(hit, hit.center) ? hit.center : pt;
  if (tag === "start") {
    history.push(["starts", P.starts.slice()]);
    const same = P.starts.filter((s) => inside(hit, s));
    if (same.length) P.starts = P.starts.filter((s) => !inside(hit, s));   // click again to untag
    else { P.starts.push(p); if (P.starts.length > 2) P.starts.shift(); }
  } else {
    history.push(["end", P.end]);
    P.end = p;
  }
  setTag(null);
}

function finishPoly() {
  if (poly.length >= 3) {
    P.vols_manual.push({ kind: "poly", points: poly.slice() });
    history.push("vol_manual");
  }
  poly = [];
  setTag(null);
  refreshRoute();
}

function onClick(pt, right) {
  if (!right && tag === "poly") {
    if (poly.length >= 3 && Math.hypot(pt[0] - poly[0][0], pt[1] - poly[0][1]) < 18 / view.s) return finishPoly();
    poly.push(pt); updateStatus(); draw(); return;
  }
  if (!right && tag === "volume") {
    P.vols_manual.push({ kind: "click", x: pt[0], y: pt[1], frame: P.frame });
    history.push("vol_manual");
    setTag(null);
  } else if (!right && tag) tagHold(pt);
  else if (!right && mode === "AUTO") { P.auto.push({ x: pt[0], y: pt[1], frame: P.frame }); history.push("sample"); }
  else if (!right) { P.manual.push({ x: pt[0], y: pt[1], frame: P.frame }); history.push("manual"); }
  else {                                         // right click: delete a hold, else a volume
    const m = holds.find((h) => h.manual_index != null && inside(h, pt));
    const mv = vols.find((v) => v.manual_index != null && inside(v, pt));
    if (m) {
      history.push(["unmanual", P.manual[m.manual_index], m.manual_index]);
      P.manual.splice(m.manual_index, 1);
    } else if (holds.some((h) => inside(h, pt))) {
      P.deleted.push(pt); history.push("deleted");
    } else if (mv) {
      history.push(["unvol", P.vols_manual[mv.manual_index], mv.manual_index]);
      P.vols_manual.splice(mv.manual_index, 1);
    } else if (vols.some((v) => inside(v, pt))) {
      P.vols_deleted.push(pt); history.push("vol_deleted");
    } else { flash("Nothing to delete there."); return; }
  }
  refreshRoute();
}

function undo() {
  const last = history.pop();
  if (!last) return;
  if (last === "sample") P.auto.pop();
  else if (last === "manual") P.manual.pop();
  else if (last === "deleted") P.deleted.pop();
  else if (last === "vol_manual") P.vols_manual.pop();
  else if (last === "vol_deleted") P.vols_deleted.pop();
  else if (last[0] === "starts") P.starts = last[1];
  else if (last[0] === "end") P.end = last[1];
  else if (last[0] === "unmanual") P.manual.splice(last[2], 0, last[1]);
  else if (last[0] === "unvol") P.vols_manual.splice(last[2], 0, last[1]);
  refreshRoute();
}

function clearAll() {
  Object.assign(P, { auto: [], manual: [], deleted: [], starts: [], end: null, vols_manual: [], vols_deleted: [] });
  history = []; poly = []; setTag(null);
  refreshRoute();
}

// ask the server for outlines; ignore answers that arrive out of order
let seq = 0, refreshTimer = null;
function refreshRoute() {
  updateStatus();
  draw();
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => {
    const mine = ++seq;
    try {
      const data = await api(`/api/jobs/${job.id}/holds`, P);
      if (mine !== seq) return;
      if (data.failed_volumes.length) {          // a V click that found no volume: drop it again
        data.failed_volumes.slice().reverse().forEach((k) => {
          P.vols_manual.splice(k, 1);
          const at = history.lastIndexOf("vol_manual");
          if (at >= 0) history.splice(at, 1);
        });
        flash("Couldn't find a volume there. Try its middle, or trace its corners (P).");
        return refreshRoute();
      }
      holds = data.holds; vols = data.volumes; skippedVols = data.skipped_volumes;
      updateStatus(); draw();
    } catch (e) { flash(e.message); }
  }, 60);
}

function setMode(m) {
  mode = m;
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  setTag(null);
}

function setTag(t) {
  if (tag === "poly" && t !== "poly") poly = [];
  tag = t;
  for (const [id, name] of [["tag-start", "start"], ["tag-end", "end"], ["tag-volume", "volume"], ["tag-poly", "poly"]]) {
    $(id).classList.toggle("on", t === name);
  }
  updateStatus();
  draw();
}

function updateStatus() {
  if (!P) return;
  const nStart = holds.filter((h) => h.role === "start").length;
  const hasEnd = holds.some((h) => h.role === "end");
  const nCut = vols.reduce((n, v) => n + (v.exclude || []).length, 0);
  $("mode-status").textContent =
    tag === "start" ? "Now: click a start hold" :
    tag === "end" ? "Now: click the end hold" :
    tag === "volume" ? "Now: click a volume to add it" :
    tag === "poly" ? `Now: click the volume's corners (${poly.length} so far)` :
    mode === "AUTO" ? "Now: click a hold to find all of its colour" : "Now: click a hold to add just that one";
  $("n-holds").textContent = `${holds.length} hold${holds.length === 1 ? "" : "s"}`;
  $("n-start").textContent = `Start ${nStart}/2`;
  $("n-end").textContent = `End ${hasEnd ? "✓" : "—"}`;
  $("n-vols").textContent = `${vols.length} volume${vols.length === 1 ? "" : "s"}` +
    (skippedVols ? ` (${skippedVols} painted, skipped)` : "");
  $("n-cuts").textContent = nCut ? `${nCut} off-route cut out` : "";
  $("analyze").disabled = holds.length === 0;
  $("undo").disabled = history.length === 0;
}

// ---------- canvas mouse: click = mark, drag = pan, wheel = zoom
let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy, moved: false, button: e.button };
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 5) return;
  drag.moved = true;
  canvas.classList.add("dragging");
  view.ox = drag.ox - dx / view.s; view.oy = drag.oy - dy / view.s;
  clampView(); draw();
});
canvas.addEventListener("pointerup", (e) => {
  canvas.classList.remove("dragging");
  if (drag && !drag.moved) {
    const pt = toImage(e);
    if (pt) onClick(pt, drag.button === 2);
  }
  drag = null;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
window.addEventListener("resize", () => { if (!$("step-holds").hidden) { fitView(); draw(); } });

// ---------- controls
document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
for (const [id, name] of [["tag-start", "start"], ["tag-end", "end"], ["tag-volume", "volume"], ["tag-poly", "poly"]]) {
  $(id).addEventListener("click", () => setTag(tag === name ? null : name));
}
$("undo").addEventListener("click", undo);
$("clear").addEventListener("click", clearAll);
$("zoom-in").addEventListener("click", () => zoomAt(1.25));
$("zoom-out").addEventListener("click", () => zoomAt(1 / 1.25));
$("zoom-fit").addEventListener("click", () => { fitView(); draw(); });

let frameTimer = null;
$("frame").addEventListener("input", (e) => {
  const i = +e.target.value;
  updateFrameLabel(i);
  clearTimeout(frameTimer);
  frameTimer = setTimeout(() => loadPicture(i, false), 120);
});
$("tol").addEventListener("input", (e) => { P.tol = +e.target.value; $("tol-label").textContent = P.tol; refreshRoute(); });
$("minsize").addEventListener("input", (e) => { P.min_size = +e.target.value; $("minsize-label").textContent = P.min_size; refreshRoute(); });

document.addEventListener("keydown", (e) => {
  if ($("step-holds").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
  if (["SELECT", "TEXTAREA", "INPUT"].includes(e.target.tagName) && e.target.type !== "range") return;
  const k = e.key.toLowerCase();
  if (k === "m") setMode(mode === "AUTO" ? "MANUAL" : "AUTO");
  else if (k === "s") setTag("start");
  else if (k === "e") setTag("end");
  else if (k === "v") setTag("volume");
  else if (k === "p") setTag("poly");
  else if (k === "z") undo();
  else if (k === "r") clearAll();
  else if (k === "escape") setTag(null);
  else if (k === "+" || k === "=") zoomAt(1.25);
  else if (k === "-" || k === "_") zoomAt(1 / 1.25);
  else if (k === "0") { fitView(); draw(); }
  else if (k === "enter" && tag === "poly") finishPoly();
  else if (k === "enter" && holds.length) analyze(true);
  else return;
  e.preventDefault();
});

$("analyze").addEventListener("click", () => analyze(true));
$("skip").addEventListener("click", () => analyze(false));

// ------------------------------------------------------------------ 3. run
async function analyze(withHolds) {
  $("notes-error").hidden = true;
  try {
    await api(`/api/jobs/${job.id}/start`, {
      picker: withHolds ? P : null, model: $("model").value, notes: $("notes").value, lock: $("lock").checked,
    });
  } catch (e) {
    if (e.message.startsWith("Sticky notes")) { $("notes-error").textContent = e.message; $("notes-error").hidden = false; $("notes").focus(); }
    else flash(e.message);
    return;
  }
  startPolling("Reading the beta…", 3);
}

// ------------------------------------------------------------------ 4. results
function fmt(t) { return `${t.toFixed(1)} s`; }

function stat(value, label, cls) {
  return `<div class="stat ${cls || ""}"><div class="v">${value}</div><div class="k">${label}</div></div>`;
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function showResults(r) {
  const player = $("player");
  player.hidden = false;
  $("player-note").hidden = true;
  player.onerror = () => { player.hidden = true; $("player-note").hidden = false; };
  player.src = r.video_url;
  $("download").href = r.download_url;
  $("csv").href = r.csv_url;
  const name = job ? job.name : "";
  $("result-sub").textContent = `${name} · ${fmt(r.duration_s)} · ${r.frames} frames at ${r.fps} fps`;

  let top;
  if (r.topped_s != null) top = stat(fmt(r.topped_s), "Topped out. Nice send!", "topped");
  else if (r.has_end) top = stat("Not yet", "Both hands never matched the end hold");
  else top = stat("—", r.holds_total ? "No end hold tagged" : "Body tracking only");
  $("stats").innerHTML =
    stat(`${r.detected_pct}%`, `Climber tracked (${r.detected}/${r.frames} frames)`) + top +
    stat(r.holds_total ? `${r.holds_used}/${r.holds_total}` : "—",
         "Route holds used" + (r.volumes ? ` · ${r.volumes} volume${r.volumes === 1 ? "" : "s"}` : ""));

  const seek = (t) => { player.currentTime = t; player.play(); };
  const moves = $("moves");
  moves.innerHTML = r.moves.length ? r.moves.map((m, i) =>
    `<li data-i="${i}"><span class="n">${i + 1}</span>` +
    `<span class="what"><span>${esc(m.limb)} on ${m.volume ? "volume" : "hold"} ${esc(m.hold)}${m.role ? `<span class="role">${esc(m.role)}</span>` : ""}</span>` +
    `<span class="t">[${fmt(m.t)}]</span></span><span class="d">${fmt(m.dur)}</span></li>`).join("")
    : `<li class="empty">${r.holds_total ? "No hand or foot stayed on a marked hold long enough to count." : "No holds were marked, so there's no hold sequence."}</li>`;
  $("moves-count").textContent = r.moves.length ? `${r.moves.length} move${r.moves.length === 1 ? "" : "s"}` : "";
  moves.querySelectorAll("li[data-i]").forEach((li) => li.addEventListener("click", () => seek(r.moves[+li.dataset.i].t)));

  $("notes-block").hidden = !r.notes.length;
  $("notes-list").innerHTML = r.notes.map((n, i) =>
    `<li data-i="${i}"><span class="n">${i + 1}</span><span class="what"><span>${esc(n.text)}</span>` +
    `<span class="t">[${fmt(n.start)}] · ${esc(n.to)}</span></span></li>`).join("");
  $("notes-list").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => seek(r.notes[+li.dataset.i].start)));

  $("angles").innerHTML = r.angles.length
    ? "<tr><th>Joint</th><th>Most bent</th><th>Typical</th><th>Straightest</th></tr>" +
      r.angles.map((a) => `<tr><td>${a.joint}</td><td>${a.min}°</td><td>${a.median}°</td><td>${a.max}°</td></tr>`).join("")
    : "<tr><td>The climber wasn't tracked in any frame.</td></tr>";
  show("step-results");
}

// ------------------------------------------------------------------ start over
function restart() {
  clearInterval(pollTimer);
  $("player").removeAttribute("src");
  $("player").load();
  job = null; P = null; holds = []; vols = []; jobId = null;
  fileInput.value = "";
  drop.hidden = false;
  $("still-row").hidden = false;
  $("upload-progress").hidden = true;
  $("upload-error").hidden = true;
  show("step-upload");
}
["restart-1", "restart-2", "restart-3"].forEach((id) => $(id).addEventListener("click", restart));
