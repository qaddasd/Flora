/**
 * Flora main page — nav, hero, scroll reveals and the sticky live-brain panel.
 * The panel reuses the demo's BrainViewer with precomputed cortical maps.
 */

import { BASE, fetchBin, fetchJSON, f16ToF32, robustRange } from "./util.js";
import { applyI18n, wireLang, wirePaperToast, getLang } from "./i18n.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

/* ── Nav bar: solid after the hero ────────────────────────────────────── */
function wireNav() {
  const nav = $("#nav");
  const onScroll = () => {
    const h = $("#hero")?.clientHeight ?? 0;
    nav.classList.toggle("solid", window.scrollY > h * 0.6);
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const drawer = $("#drawer");
  $("#nav-burger").addEventListener("click", () => drawer.classList.remove("hidden"));
  drawer.addEventListener("click", (e) => {
    if (e.target.closest("[data-close]")) drawer.classList.add("hidden");
  });
}

/* ── Hero video: fade in when ready, portrait source on tall screens ──── */
function wireHero() {
  const v = $("#hero-video");
  const pick = () => {
    const portrait = window.innerWidth <= window.innerHeight;
    const src = portrait ? `${BASE}videos/video-banner-mobile.mp4` : `${BASE}videos/video-banner.mp4`;
    if (!v.src.endsWith(src.split("/").pop())) v.src = src;
  };
  pick();
  window.addEventListener("resize", pick);
  v.addEventListener("loadeddata", () => v.classList.add("ready"));
}

/* ── Scroll reveals + section arrows ──────────────────────────────────── */
function wireArticle() {
  const sections = $$(".a-section");
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("revealed"); io.unobserve(e.target); }
    }),
    { threshold: 0.12 },
  );
  sections.forEach((s) => io.observe(s));

  sections.forEach((sec, i) => {
    sec.querySelectorAll(".sec-btn").forEach((btn) => {
      const j = i + Number(btn.dataset.dir);
      if (j < 0 || j >= sections.length) btn.disabled = true;
      btn.addEventListener("click", () =>
        sections[j].scrollIntoView({ behavior: "smooth", block: "start" }));
    });
  });
}

/* ── Mobile banner ────────────────────────────────────────────────────── */
function wireMobileBanner() {
  const banner = $("#mobile-banner");
  if (window.matchMedia("(max-width: 1023px)").matches) banner.classList.remove("hidden");
  $("#mb-continue").addEventListener("click", () => banner.classList.add("hidden"));
}

/* ── Sticky live-brain panel ──────────────────────────────────────────── */
const LEGENDS = {
  activity: `${BASE}assets/legend-activity-4d3959ff.svg`,
  encoding: `${BASE}assets/legend-encoding-score-fd4aa7c1.svg`,
};

const PANEL = {
  intro:        { kind: "template", id: "ocean_waves",  mode: "predicted", legend: "activity", caption: "Predicted cortical activity while the model watches “Ocean Waves”.", captionRu: "Предсказанная активность коры, пока модель смотрит «Ocean Waves»." },
  architecture: { kind: "template", id: "night_drive",  mode: "predicted", legend: "activity", caption: "Three frozen encoders, one fusion stack — “Night Drive” mapped onto cortex.", captionRu: "Три замороженных энкодера, один фьюжн-стек — «Night Drive» на коре." },
  performance:  { kind: "dataset",  id: "clip_000001",  mode: "compare",   legend: "encoding", caption: "True (left) vs predicted (right) responses to a held-out clip.", captionRu: "Истина (слева) и предсказание (справа) на отложенном клипе." },
  scaling:      { kind: "template", id: "nebula",       mode: "predicted", legend: "activity", caption: "“Nebula Drift” — predicted BOLD responses unfolding over time.", captionRu: "«Nebula Drift» — предсказанный BOLD-ответ во времени." },
  vision:       { kind: "dataset",  id: "clip_000002",  mode: "compare",   legend: "encoding", caption: "Parcel-wise agreement on another held-out clip.", captionRu: "Согласование по парцеллам на ещё одном отложенном клипе." },
  insilico:     { kind: "template", id: "city_rain",    mode: "predicted", legend: "activity", caption: "“City Rain” — lesion its modalities live in the full demo.", captionRu: "«City Rain» — отключайте модальности вживую в полном демо." },
  explore:      { kind: "template", id: "forest_birds", mode: "predicted", legend: "activity", caption: "“Forest Birds” — open the demo and explore it yourself.", captionRu: "«Forest Birds» — откройте демо и исследуйте сами." },
};

const captionFor = (cfg) => (getLang() === "ru" ? cfg.captionRu : cfg.caption);

const P = {
  viewer: null,
  manifest: null,
  cache: new Map(),     // clip id → { predSeries, trueSeries|null }
  current: null,
  phase: 0,
  lastTs: 0,
};

function sampleEvery(arr, k) {
  const out = [];
  for (let i = 0; i < arr.length; i += k) out.push(arr[i]);
  return out;
}

function normalizeSeries(series) {
  const all = [];
  for (const s of series) all.push(...sampleEvery(s, 4));
  const [lo, hi] = robustRange(Float32Array.from(all), 1, 99);
  const range = hi - lo || 1;
  return series.map((s) => {
    const out = new Float32Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = Math.max(0, Math.min(1, (s[i] - lo) / range));
    return out;
  });
}

function parcelsToSeries(parcels, T) {
  const labels = P.viewer.parcelLabels;
  const n = P.viewer.nVertices;
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
  const n = P.viewer.nVertices;
  const series = [];
  for (let t = 0; t < T; t++) series.push(data.subarray(t * n, (t + 1) * n));
  return series;
}

async function clipSeries(cfg) {
  if (P.cache.has(cfg.id)) return P.cache.get(cfg.id);
  const promise = (async () => {
    if (cfg.kind === "template") {
      const t = P.manifest.templates.find((x) => x.id === cfg.id);
      const data = f16ToF32(await fetchBin(`${BASE}${t.pred}`));
      return { predSeries: normalizeSeries(parcelsToSeries(data, t.T)), trueSeries: null, T: t.T };
    }
    const c = P.manifest.dataset_clips.find((x) => x.id === cfg.id);
    const [predRaw, trueRaw] = await Promise.all([
      fetchBin(`${BASE}${c.pred}`).then(f16ToF32),
      fetchBin(`${BASE}${c.true}`).then(f16ToF32),
    ]);
    const joined = normalizeSeries([
      ...verticesToSeries(predRaw, c.T),
      ...verticesToSeries(trueRaw, c.T),
    ]);
    return { predSeries: joined.slice(0, c.T), trueSeries: joined.slice(c.T), T: c.T };
  })();
  P.cache.set(cfg.id, promise);
  return promise;
}

async function applyPanel(sectionId) {
  const cfg = PANEL[sectionId];
  if (!cfg || !P.viewer || P.current === sectionId) return;
  P.current = sectionId;

  $("#panel-legend").src = LEGENDS[cfg.legend];
  $("#panel-caption").textContent = captionFor(cfg);
  $("#panel-load").classList.remove("hidden");

  try {
    const { predSeries, trueSeries, T } = await clipSeries(cfg);
    if (P.current !== sectionId) return; // user scrolled on
    P.clip = { predSeries, trueSeries, T };
    P.phase = 0;
    P.viewer.setMode(cfg.mode);
  } finally {
    if (P.current === sectionId) $("#panel-load").classList.add("hidden");
  }
}

let scratch = null;
function panelLoop(ts) {
  requestAnimationFrame(panelLoop);
  if (!P.viewer || !P.clip) return;
  const dt = Math.min(0.1, (ts - P.lastTs) / 1000 || 0.016);
  P.lastTs = ts;
  P.phase = (P.phase + dt / 8) % 1; // 8 s loop, like the demo

  const { predSeries, trueSeries, T } = P.clip;
  const f = Math.min(T - 1e-4, P.phase * T);
  const i0 = Math.floor(f);
  const i1 = Math.min(T - 1, i0 + 1);
  const w = f - i0;
  const lerp = (series) => {
    const a = series[i0], b = series[i1];
    if (!scratch || scratch.length !== a.length) scratch = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) scratch[i] = a[i] + (b[i] - a[i]) * w;
    return scratch;
  };
  P.viewer.setActivity(lerp(predSeries), "pred");
  if (trueSeries) P.viewer.setActivity(lerp(trueSeries), "true");
}

function currentSection() {
  const y = window.innerHeight * 0.4;
  let current = "intro";
  for (const el of $$(".a-section")) {
    if (el.getBoundingClientRect().top <= y) current = el.dataset.panel;
  }
  return current;
}

function wirePanel() {
  const stage = $("#panel-stage");
  if (!stage) return;

  const onScroll = () => { if (P.viewer) applyPanel(currentSection()); };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("flora:lang", () => {
    const cfg = PANEL[P.current];
    if (cfg) $("#panel-caption").textContent = captionFor(cfg);
  });

  const io = new IntersectionObserver(async (entries) => {
    if (!entries.some((e) => e.isIntersecting) || P.viewer) return;
    io.disconnect();
    $("#panel-load").classList.remove("hidden");
    const { BrainViewer } = await import("./brain.js");
    P.viewer = await BrainViewer.create($("#panel-canvas"));
    window.addEventListener("resize", () => P.viewer.resize());
    P.manifest = await fetchJSON(`${BASE}data/templates.json`);
    $("#panel-load").classList.add("hidden");
    P.current = null;
    applyPanel(currentSection());
    requestAnimationFrame(panelLoop);
  }, { threshold: 0.05 });
  io.observe(stage);
}

applyI18n();
wireLang();
wirePaperToast();
wireNav();
wireHero();
wireArticle();
wireMobileBanner();
wirePanel();
