/**
 * Flora web demo — application wiring.
 *
 * Modes of operation:
 *  1. Template clips   — predictions precomputed by the real PyTorch pipeline
 *  2. Dataset clips    — precomputed student vs teacher (True/Compare)
 *  3. In-Silico        — live fusion-model reruns with lesioned modalities
 *  4. Upload           — full live pipeline: decode → MobileViT → Whisper → Flora
 */

import { BrainViewer } from "./brain.js";
import {
  BASE, fetchBin, fetchJSON, f16ToF32, robustRange, fmtMs, tick,
} from "./util.js";
import {
  onBackend, webgpuAvailable, getBackend, runFusion, runVideoEncoder, runAudioEncoder,
  extractFrames, loadVideoElement,
} from "./inference.js";
import { computeWhisperMel, drawMel, drawWaveform, decodeAudio16k } from "./audio.js";
import { applyI18n, wireLang } from "./i18n.js";

/* ── ONNX Runtime Web env ─────────────────────────────────────────────── */
// NOTE: wasmPaths is intentionally NOT set — ORT resolves its wasm/glue files
// relative to the loaded bundle (vendor/ort/), which is where they live.
// (A relative override like "./vendor/ort/" gets doubled by ORT's resolver.)
ort.env.wasm.numThreads = 1;
if (webgpuAvailable()) {
  ort.env.webgpu.powerPreference = "high-performance";
}

/* ── Global state ─────────────────────────────────────────────────────── */
const S = {
  brain: null,
  manifest: null,
  clip: null,          // { id, title, T, predSeries:[T×F32(20484)], trueSeries|null, videoURL|null, dur }
  playing: false,
  pseudoT: 0,          // timer-based playback position for clips without video
  uploads: [],
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

/* ── Backend badge ────────────────────────────────────────────────────── */
onBackend((b) => {
  const badge = $("#backend-badge");
  badge.classList.remove("webgpu", "wasm");
  badge.classList.add(b);
  $("#backend-text").textContent = b === "webgpu" ? "Flora · WebGPU" : "Flora · WASM";
  $("#about-backend").innerHTML =
    `<strong>Backend: ${b === "webgpu" ? "WebGPU (FP16)" : "WASM SIMD fallback"}</strong>`;
});
if (!webgpuAvailable()) {
  $("#backend-text").textContent = "Flora · no WebGPU → WASM";
}

/* ── Boot ─────────────────────────────────────────────────────────────── */
applyI18n();
wireLang();
init();

async function init() {
  S.brain = await BrainViewer.create($("#brain-canvas"));
  window.addEventListener("resize", () => S.brain.resize());

  S.manifest = await fetchJSON(BASE + "data/templates.json");
  renderExamples();
  renderCompare();
  renderInSilicoChips();
  wireTabs();
  wireSegments();
  wireGuide();
  wireUploadModal();
  applyI18n();
  requestAnimationFrame(playbackLoop);

  // deep links: ?clip=ocean_waves&open=1&surface=inflated&tab=insilico
  const q = new URLSearchParams(location.search);
  if (q.get("open") === "1") $('#seg-open [data-open="1"]')?.click();
  if (q.get("surface") === "inflated") $('#seg-surface [data-surface="inflated"]')?.click();
  const tab = q.get("tab");
  if (tab) $$("#tabs .tab").find((b) => b.dataset.tab === tab)?.click();
  const clipId = q.get("clip");
  if (clipId) {
    const t = S.manifest.templates.find((x) => x.id === clipId);
    const c = S.manifest.dataset_clips.find((x) => x.id === clipId);
    if (t) playTemplate(t);
    else if (c) playDatasetClip(c);
  }

  if (location.search.includes("autotest")) {
    try {
      const { runSelfTest } = await import("./selftest.js");
      runSelfTest();
    } catch (e) {
      fetch("/report", { method: "POST", body: `AUTOTEST:IMPORT-FAIL ${e.message || e}` }).catch(() => {});
    }
  }
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
function wireTabs() {
  $$("#tabs .tab").forEach((btn) => btn.addEventListener("click", () => {
    $$("#tabs .tab").forEach((b) => b.classList.toggle("active", b === btn));
    $$(".tab-page").forEach((p) => p.classList.toggle("active", p.id === `page-${btn.dataset.tab}`));
  }));
}

/* ── Segmented controls ───────────────────────────────────────────────── */
function wireSegments() {
  $$("#seg-mode button").forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    $$("#seg-mode button").forEach((x) => x.classList.toggle("active", x === b));
    S.brain.setMode(b.dataset.mode);
    $("#brain-compare-labels").classList.toggle("hidden", b.dataset.mode !== "compare");
  }));
  $$("#seg-surface button").forEach((b) => b.addEventListener("click", () => {
    $$("#seg-surface button").forEach((x) => x.classList.toggle("active", x === b));
    S.brain.setSurface(b.dataset.surface);
  }));
  $$("#seg-open button").forEach((b) => b.addEventListener("click", () => {
    $$("#seg-open button").forEach((x) => x.classList.toggle("active", x === b));
    S.brain.setOpen(b.dataset.open === "1");
  }));
}

function setModeButtons({ hasTrue }) {
  $$("#seg-mode button").forEach((b) => {
    const m = b.dataset.mode;
    b.disabled = (m === "true" || m === "compare") ? !hasTrue : false;
  });
  // activate Predicted
  $$("#seg-mode button").forEach((x) => x.classList.toggle("active", x.dataset.mode === "predicted"));
  S.brain.setMode("predicted");
  $("#brain-compare-labels").classList.add("hidden");
}

/* ── Guide overlay ────────────────────────────────────────────────────── */
function wireGuide() {
  $("#guide-btn").addEventListener("click", () => $("#guide-overlay").classList.remove("hidden"));
  $("#guide-close").addEventListener("click", () => $("#guide-overlay").classList.add("hidden"));
  $(".guide-backdrop").addEventListener("click", () => $("#guide-overlay").classList.add("hidden"));
}

/* ── Template grid ────────────────────────────────────────────────────── */
function videoCard({ title, thumb, dur, onClick, badge }) {
  const el = document.createElement("div");
  el.className = "video-card";
  el.innerHTML = `
    <img src="${thumb}" alt="${title}" loading="lazy"/>
    <div class="vc-play"><div class="vc-circle">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="#fff"><path d="M8 5v14l11-7z"/></svg>
    </div></div>
    <div class="vc-meta"><span class="vc-title">${title}</span>
    <span class="vc-dur">${badge || dur || ""}</span></div>`;
  el.addEventListener("click", () => {
    $$(".video-card").forEach((c) => c.classList.remove("playing"));
    el.classList.add("playing");
    onClick(el);
  });
  return el;
}

function renderExamples() {
  const grid = $("#examples-grid");
  // upload card first (like the reference has the grid; upload is extra)
  const up = document.createElement("div");
  up.className = "video-card upload-card";
  up.innerHTML = `<div class="uc-inner"><div class="uc-plus">+</div>
    <div data-i18n="d.browse.upload">Upload your video</div><div class="dim small" data-i18n="d.browse.upload2">live WebGPU inference</div></div>`;
  up.addEventListener("click", openUploadModal);
  grid.appendChild(up);

  for (const t of S.manifest.templates) {
    grid.appendChild(videoCard({
      title: t.title, thumb: `${BASE}${t.thumb}`, dur: "0:08",
      onClick: () => playTemplate(t),
    }));
  }
}

/* ── Activation series helpers ────────────────────────────────────────── */

/** Build per-TR vertex series (Float32Array of nVertices) from parcel values. */
function parcelsToSeries(parcels, T) {
  const labels = S.brain.parcelLabels;
  const n = S.brain.nVertices;
  const series = [];
  for (let t = 0; t < T; t++) {
    const verts = new Float32Array(n);
    const row = t * 400;
    for (let v = 0; v < n; v++) {
      const p = labels[v];
      if (p > 0) verts[v] = parcels[row + p - 1];
    }
    series.push(verts);
  }
  return series;
}

function verticesToSeries(data, T) {
  const n = S.brain.nVertices;
  const series = [];
  for (let t = 0; t < T; t++) series.push(data.subarray(t * n, (t + 1) * n));
  return series;
}

/** Robust-normalize a whole series jointly to 0..1. */
function normalizeSeries(series) {
  let all = [];
  for (const s of series) all.push(...sampleEvery(s, 4));
  const [lo, hi] = robustRange(Float32Array.from(all), 1, 99);
  const range = hi - lo || 1;
  return series.map((s) => {
    const out = new Float32Array(s.length);
    for (let i = 0; i < s.length; i++) {
      out[i] = Math.max(0, Math.min(1, (s[i] - lo) / range));
    }
    return out;
  });
}
function sampleEvery(arr, k) {
  const out = [];
  for (let i = 0; i < arr.length; i += k) out.push(arr[i]);
  return out;
}

/* ── Clip loading & playback ──────────────────────────────────────────── */

async function loadPredSeries(entry) {
  const buf = await fetchBin(`${BASE}${entry.pred}`);
  const data = f16ToF32(buf);
  return entry.kind === "parcels"
    ? parcelsToSeries(data, entry.T)
    : verticesToSeries(data, entry.T);
}

async function playTemplate(t) {
  stopPlayback();
  $("#stage-empty-hint").style.opacity = 0;
  const pred = normalizeSeries(await loadPredSeries(t));
  S.clip = {
    id: t.id, title: t.title, T: t.T,
    predSeries: pred, trueSeries: null,
    videoURL: `${BASE}${t.video}`, dur: 8,
  };
  setModeButtons({ hasTrue: false });
  showVideoFloat(t.title, `${BASE}${t.video}`);
  S.playing = true;
}

async function playDatasetClip(c) {
  stopPlayback();
  $("#stage-empty-hint").style.opacity = 0;
  const [predRaw, trueRaw] = await Promise.all([
    fetchBin(`${BASE}${c.pred}`).then(f16ToF32),
    fetchBin(`${BASE}${c.true}`).then(f16ToF32),
  ]);
  // normalize pred & true jointly (they share the colorbar)
  const predSeries = verticesToSeries(predRaw, c.T);
  const trueSeries = verticesToSeries(trueRaw, c.T);
  const joined = normalizeSeries([...predSeries, ...trueSeries]);
  S.clip = {
    id: c.id, title: c.title, T: c.T,
    predSeries: joined.slice(0, c.T),
    trueSeries: joined.slice(c.T),
    videoURL: null, dur: 8,
  };
  setModeButtons({ hasTrue: true });
  // start in Compare mode
  $$("#seg-mode button").forEach((x) => x.classList.toggle("active", x.dataset.mode === "compare"));
  S.brain.setMode("compare");
  $("#brain-compare-labels").classList.remove("hidden");
  hideVideoFloat("stimulus replay — no video file for dataset clips");
  S.playing = true;
}

function stopPlayback() {
  S.playing = false;
  S.pseudoT = 0;
  const v = $("#vf-video");
  v.pause();
  S.brain.clearActivity();
}

/* ── Floating video player ────────────────────────────────────────────── */
function showVideoFloat(title, url) {
  $("#video-float").classList.remove("hidden");
  $("#video-float").classList.remove("note-mode");
  $("#vf-title").textContent = title;
  const v = $("#vf-video");
  v.src = url;
  v.loop = true;
  v.play().catch(() => {});
  // TR ticks
  const ticks = $("#vf-tr-ticks");
  ticks.innerHTML = "";
  for (let i = 0; i < (S.clip?.T || 5); i++) ticks.appendChild(document.createElement("i"));
  S.brain.setAutoRotate(false);
}
function hideVideoFloat(note) {
  const panel = $("#video-float");
  if (note) {
    panel.classList.remove("hidden");
    panel.classList.add("note-mode");
    $("#vf-title").textContent = note;
    const v = $("#vf-video");
    v.pause();
    v.removeAttribute("src");
    v.load();
  } else {
    panel.classList.add("hidden");
  }
  S.brain.setAutoRotate(true);
}
$("#vf-close").addEventListener("click", () => {
  stopPlayback();
  hideVideoFloat();
  $$(".video-card").forEach((c) => c.classList.remove("playing"));
  $("#stage-empty-hint").style.opacity = 1;
  S.clip = null;
  S.brain.setAutoRotate(true);
});

/* ── Playback loop: sync brain with video time ────────────────────────── */
let lerpScratch = null; // reused across frames — setActivity copies synchronously

function playbackLoop() {
  requestAnimationFrame(playbackLoop);
  if (!S.playing || !S.clip) return;

  const v = $("#vf-video");
  let phase; // 0..1
  if (v.src && v.duration && Number.isFinite(v.duration)) {
    phase = v.currentTime / v.duration;
    $("#vf-progress").style.width = `${phase * 100}%`;
  } else {
    // timer-based loop for clips without video
    S.pseudoT = (S.pseudoT + 1 / 60) % 8;
    phase = S.pseudoT / 8;
  }

  const T = S.clip.T;
  const f = Math.min(T - 1e-4, phase * T);
  const i0 = Math.floor(f);
  const i1 = Math.min(T - 1, i0 + 1);
  const w = f - i0;

  const lerp = (series) => {
    if (!series) return null;
    const a = series[i0], b = series[i1];
    if (!lerpScratch || lerpScratch.length !== a.length) {
      lerpScratch = new Float32Array(a.length);
    }
    const out = lerpScratch;
    for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * w;
    return out;
  };

  S.brain.setActivity(lerp(S.clip.predSeries), "pred");
  if (S.clip.trueSeries) S.brain.setActivity(lerp(S.clip.trueSeries), "true");
}

/* ── Compare tab ──────────────────────────────────────────────────────── */
function renderCompare() {
  const clips = S.manifest.dataset_clips;
  if (!clips.length) return;
  const rs = clips.map((c) => c.parcel_r);
  $("#stat-best-r").textContent = Math.max(...rs).toFixed(3);
  $("#stat-mean-r").textContent = (rs.reduce((a, b) => a + b, 0) / rs.length).toFixed(3);

  const list = $("#compare-list");
  for (const c of clips) {
    const el = document.createElement("div");
    el.className = "compare-item";
    el.innerHTML = `
      <div><div class="ci-title">${c.title}</div>
      <div class="dim small">${c.desc}</div></div>
      <canvas width="150" height="44"></canvas>
      <div class="ci-r">r = ${c.parcel_r.toFixed(3)}</div>`;
    // per-TR bars
    const cv = el.querySelector("canvas");
    const ctx = cv.getContext("2d");
    const vals = c.parcel_r_per_tr;
    const bw = cv.width / vals.length;
    vals.forEach((r, i) => {
      const h = Math.max(2, r * cv.height);
      const grad = ctx.createLinearGradient(0, cv.height - h, 0, cv.height);
      grad.addColorStop(0, "#ffb13d"); grad.addColorStop(1, "#c22e08");
      ctx.fillStyle = grad;
      ctx.fillRect(i * bw + 3, cv.height - h, bw - 6, h);
    });
    el.addEventListener("click", () => playDatasetClip(c));
    list.appendChild(el);
  }
}

/* ── In-Silico tab ────────────────────────────────────────────────────── */
function renderInSilicoChips() {
  const row = $("#is-clips");
  S.manifest.templates.forEach((t, i) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (i === 0 ? " on" : "");
    chip.textContent = t.title;
    chip.dataset.id = t.id;
    chip.addEventListener("click", () => {
      $$("#is-clips .chip").forEach((c) => c.classList.toggle("on", c === chip));
    });
    row.appendChild(chip);
  });
  $$("#is-mods .chip").forEach((c) => c.addEventListener("click", () => c.classList.toggle("on")));

  $("#is-run").addEventListener("click", runInSilico);
}

async function runInSilico() {
  const clipId = $("#is-clips .chip.on")?.dataset.id;
  const entry = S.manifest.templates.find((t) => t.id === clipId);
  if (!entry) return;
  const mods = Object.fromEntries(
    $$("#is-mods .chip").map((c) => [c.dataset.mod, c.classList.contains("on")])
  );
  const status = $("#is-status");
  const runBtn = $("#is-run");
  runBtn.disabled = true;
  try {
    status.textContent = "Loading features…";
    const buf = await fetchBin(`${BASE}${entry.feats}`);
    const feats = f16ToF32(buf); // (T,1408) = text|audio|video
    const T = entry.T;
    const text = new Float32Array(T * 384);
    const audio = new Float32Array(T * 384);
    const video = new Float32Array(T * 640);
    for (let t = 0; t < T; t++) {
      const base = t * 1408;
      if (mods.text) text.set(feats.subarray(base, base + 384), t * 384);
      if (mods.audio) audio.set(feats.subarray(base + 384, base + 768), t * 384);
      if (mods.video) video.set(feats.subarray(base + 768, base + 1408), t * 640);
    }

    status.textContent = "Running Flora fusion…";
    const [{ parcels }, baseParcels] = await Promise.all([
      runFusion(text, audio, video, T, (f, msg) => { status.textContent = msg; }),
      (async () => {
        const b = await fetchBin(`${BASE}${entry.pred}`);
        return f16ToF32(b);
      })(),
    ]);

    // delta metric
    let dsum = 0, bsum = 0;
    for (let i = 0; i < parcels.length; i++) {
      dsum += Math.abs(parcels[i] - baseParcels[i]);
      bsum += Math.abs(baseParcels[i]);
    }
    const delta = dsum / (bsum || 1);
    $("#is-result").classList.remove("hidden");
    $("#is-delta").textContent = (delta * 100).toFixed(1) + "%";

    // per-TR mean |activity| chart: full vs lesioned
    drawInSilicoChart(baseParcels, parcels, T);

    // show on brain (lesioned prediction), play the stimulus video
    stopPlayback();
    $("#stage-empty-hint").style.opacity = 0;
    const pred = normalizeSeries(parcelsToSeries(parcels, T));
    S.clip = {
      id: `${entry.id}-lesion`, title: `${entry.title} · in-silico`,
      T, predSeries: pred, trueSeries: null, videoURL: `${BASE}${entry.video}`, dur: 8,
    };
    setModeButtons({ hasTrue: false });
    showVideoFloat(`${entry.title} — lesioned (${Object.entries(mods).filter(([, v]) => v).map(([k]) => k).join("+") || "none"})`, `${BASE}${entry.video}`);
    S.playing = true;
    status.textContent = `Done — fusion ran on ${getBackend() === "webgpu" ? "WebGPU (FP16)" : "WASM"}.`;
  } catch (e) {
    console.error(e);
    status.textContent = `Error: ${e.message || e}`;
  } finally {
    runBtn.disabled = false;
  }
}

function drawInSilicoChart(base, lesioned, T) {
  const cv = $("#is-chart");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const mean = (arr, t) => {
    let s = 0;
    for (let i = 0; i < 400; i++) s += Math.abs(arr[t * 400 + i]);
    return s / 400;
  };
  const valsB = [...Array(T)].map((_, t) => mean(base, t));
  const valsL = [...Array(T)].map((_, t) => mean(lesioned, t));
  const mx = Math.max(...valsB, ...valsL) || 1;
  const bw = cv.width / T;
  for (let t = 0; t < T; t++) {
    const hB = valsB[t] / mx * (cv.height - 24);
    const hL = valsL[t] / mx * (cv.height - 24);
    ctx.fillStyle = "#3d3d45";
    ctx.fillRect(t * bw + 8, cv.height - 14 - hB, bw / 2 - 10, hB);
    ctx.fillStyle = "#ff8a3d";
    ctx.fillRect(t * bw + bw / 2, cv.height - 14 - hL, bw / 2 - 10, hL);
    ctx.fillStyle = "#8b8b93";
    ctx.font = "10px sans-serif";
    ctx.fillText(`TR${t + 1}`, t * bw + 8, cv.height - 2);
  }
  ctx.fillStyle = "#cfcfd6";
  ctx.fillText("■ full input", 8, 12);
  ctx.fillStyle = "#ff8a3d";
  ctx.fillText("■ lesioned", 80, 12);
}

/* ── Upload modal & live pipeline ─────────────────────────────────────── */
let umFile = null;

function openUploadModal() {
  $("#upload-modal").classList.remove("hidden");
  $("#um-drop").classList.remove("hidden");
  $("#um-pipeline").classList.add("hidden");
  $("#um-title").textContent = "Process your video";
  $$(".pipe-step").forEach((s) => s.classList.remove("active", "done", "failed"));
}

function wireUploadModal() {
  $("#um-close").addEventListener("click", () => $("#upload-modal").classList.add("hidden"));
  $("#upload-modal").addEventListener("click", (e) => {
    if (e.target === $("#upload-modal")) $("#upload-modal").classList.add("hidden");
  });
  $("#um-input").addEventListener("change", (e) => {
    if (e.target.files[0]) startPipeline(e.target.files[0]);
    e.target.value = ""; // allow picking the same file again
  });
  const drop = $("#um-drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) startPipeline(f);
  });
}

function setStep(name, state, info, barFrac) {
  const el = $(`.pipe-step[data-step="${name}"]`);
  if (!el) return;
  el.classList.remove("active", "done");
  if (state) el.classList.add(state);
  if (info != null) el.querySelector(".ps-info") && (el.querySelector(".ps-info").textContent = info);
  if (barFrac != null) {
    const bar = el.querySelector(".ps-bar > div");
    if (bar) bar.style.width = `${Math.round(barFrac * 100)}%`;
  }
}

async function startPipeline(file) {
  umFile = file;
  $("#um-drop").classList.add("hidden");
  $("#um-pipeline").classList.remove("hidden");
  $("#um-view").classList.add("hidden");
  $("#um-title").textContent = `Process: ${file.name.slice(0, 40)}`;
  $$(".pipe-step").forEach((s) => s.classList.remove("active", "done", "failed"));
  $$(".pipe-step .ps-bar > div").forEach((b) => (b.style.width = "0"));
  let currentStep = "decode";
  const T = 5;

  try {
    /* 1 · decode */
    currentStep = "decode";
    setStep("decode", "active");
    const video = await loadVideoElement(file);
    setStep("decode", "done",
      `${video.videoWidth}×${video.videoHeight} · ${video.duration.toFixed(1)} s · ${(file.size / 1e6).toFixed(1)} MB`);

    /* 2 · frames */
    currentStep = "frames";
    setStep("frames", "active");
    const framesBox = $("#ps-frames");
    framesBox.innerHTML = "";
    const { batch, thumbs } = await extractFrames(video, T, (i, n, thumb) => {
      const img = document.createElement("img");
      img.src = thumb;
      img.alt = `Sampled frame ${i}`;
      framesBox.appendChild(img);
      setStep("frames", "active", `${i}/${n} frames`);
    });
    setStep("frames", "done", `${T} frames @ 256×256, BGR [0,1]`);

    /* 3 · video encoder */
    currentStep = "video-enc";
    setStep("video-enc", "active");
    const vEnc = await runVideoEncoder(batch, T, (f, msg) => setStep("video-enc", "active", msg, f));
    setStep("video-enc", "done", `(T,640) features · ${fmtMs(vEnc.ms)}`);

    /* 4 · audio decode */
    currentStep = "audio";
    setStep("audio", "active");
    let pcm;
    try {
      pcm = await decodeAudio16k(file);
      drawWaveform($("#ps-wave"), pcm);
      setStep("audio", "done", `${(pcm.length / 16000).toFixed(1)} s @ 16 kHz`);
    } catch {
      pcm = new Float32Array(16000); // silent track fallback
      setStep("audio", "done", "no audio track — using silence");
    }

    /* 5 · mel + audio encoder */
    currentStep = "audio-enc";
    setStep("audio-enc", "active");
    const mel = await computeWhisperMel(pcm, (f) => setStep("audio-enc", "active", "log-mel spectrogram…", f * 0.5));
    drawMel($("#ps-mel"), mel);
    const aEnc = await runAudioEncoder(mel, T, (f, msg) => setStep("audio-enc", "active", msg, 0.5 + f * 0.5));
    setStep("audio-enc", "done", `(T,384) features · ${fmtMs(aEnc.ms)}`);

    /* 6 · fusion */
    currentStep = "fusion";
    setStep("fusion", "active");
    const text = new Float32Array(T * 384); // clips have no transcript (matches training)
    const { parcels, ms } = await runFusion(text, aEnc.feats, vEnc.feats, T,
      (f, msg) => setStep("fusion", "active", msg, f));
    setStep("fusion", "done", `(T,400) parcels · ${fmtMs(ms)}`);

    /* 7 · project to cortex */
    setStep("done", "done", "400 parcels → 20,484 vertices · ready");
    $("#um-view").classList.remove("hidden");
    $("#um-view").onclick = () => {
      $("#upload-modal").classList.add("hidden");
      addUploadClip(file, thumbs[0], parcels, T, video);
    };
  } catch (e) {
    console.error(e);
    setStep(currentStep, "failed", `Error: ${e.message || e}`);
    $("#um-title").textContent = "Processing failed";
  }
}

function addUploadClip(file, thumb, parcels, T, videoEl) {
  const title = file.name.replace(/\.[^.]+$/, "").slice(0, 26);
  const blobURL = videoEl.src; // keep — the hidden element holds the blob alive
  const entry = { id: `up-${Date.now()}`, title, T, thumb };

  const pred = normalizeSeries(parcelsToSeries(parcels, T));
  const startUploaded = () => {
    stopPlayback();
    $("#stage-empty-hint").style.opacity = 0;
    S.clip = {
      id: entry.id, title, T,
      predSeries: pred, trueSeries: null,
      videoURL: blobURL, dur: videoEl.duration,
    };
    setModeButtons({ hasTrue: false });
    showVideoFloat(`${title} (your upload)`, blobURL);
    $("#vf-video").loop = true;
    S.playing = true;
  };

  startUploaded();

  // card in "My uploads"
  const row = $("#uploads-row");
  row.classList.remove("hidden");
  const card = videoCard({ title, thumb, dur: "you", onClick: startUploaded });
  $("#uploads-grid").appendChild(card);
  $$(".video-card").forEach((c) => c.classList.remove("playing"));
  card.classList.add("playing");
}
