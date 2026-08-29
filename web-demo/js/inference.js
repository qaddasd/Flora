/**
 * Flora inference for the browser — ONNX Runtime Web, WebGPU-first.
 *
 * Sessions (lazy, on demand):
 *   fusion : (1,T,384)+(1,T,384)+(1,T,640) → (1,T,400)
 *   video  : (N,3,256,256) → (N,640)
 *   audio  : (1,80,3000)   → (1,1500,384)
 *
 * Precision strategy: FP16 weights are preferred on WebGPU (half the download,
 * faster kernels), but only when the adapter really supports shader-f16 —
 * otherwise fp16 graphs silently produce zeros/NaN. Even with the feature
 * present, every fresh fp16 session must pass a numerical smoke test
 * (finite + non-degenerate output) before it is trusted; on any failure we
 * fall back to the FP32 graph, then to WASM. Backend reported via onBackend.
 */

import { BASE, fetchWithProgress, resampleLinear, fmtMB, f32ToF16, f16ToF32 } from "./util.js";

const MODELS = {
  fusion: { fp16: BASE + "models/fusion_web_fp16.onnx", fp32: BASE + "models/fusion_web.onnx" },
  video:  { fp16: BASE + "models/video_encoder_web_fp16.onnx", fp32: BASE + "models/video_encoder_web.onnx" },
  audio:  { fp16: BASE + "models/audio_encoder_web_fp16.onnx", fp32: BASE + "models/audio_encoder_web.onnx" },
};

const sessions = {};
let backend = null; // "webgpu" | "wasm"
let backendListeners = [];

export function onBackend(cb) { backendListeners.push(cb); if (backend) cb(backend); }
function setBackend(b) {
  backend = b;
  backendListeners.forEach((cb) => cb(b));
}

export function getBackend() { return backend; }

export function webgpuAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

/** True only if the WebGPU adapter advertises real fp16 shader support. */
let _f16Support = null;
export async function webgpuF16Supported() {
  if (!webgpuAvailable()) return false;
  if (_f16Support === null) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      _f16Support = !!adapter && adapter.features.has("shader-f16");
    } catch {
      _f16Support = false;
    }
  }
  return _f16Support;
}

/** Deterministic non-degenerate test pattern (never all-zero / all-const). */
function testPattern(n, seed = 1) {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 4294967296) - 0.5;
  }
  return out;
}

/**
 * Run one forward pass on synthetic input and require finite, non-constant
 * output. Catches broken fp16 paths (silent zeros / NaN) before the model
 * is trusted with real data.
 */
async function smokeTest(name, sess, io) {
  const mk = (data, dims) =>
    io === "float16" ? new ort.Tensor("float16", f32ToF16(data), dims)
                     : new ort.Tensor("float32", data, dims);
  let out;
  if (name === "fusion") {
    out = await sess.run({
      text_features: mk(testPattern(2 * 384, 11), [1, 2, 384]),
      audio_features: mk(testPattern(2 * 384, 22), [1, 2, 384]),
      video_features: mk(testPattern(2 * 640, 33), [1, 2, 640]),
      subject_id: new ort.Tensor("int64", BigInt64Array.from([0n]), [1]),
    });
    out = out.parcels;
  } else if (name === "video") {
    out = await sess.run({ pixel_values: mk(testPattern(3 * 256 * 256, 44), [1, 3, 256, 256]) });
    out = out.embeddings;
  } else {
    out = await sess.run({ input_features: mk(testPattern(80 * 3000, 55), [1, 80, 3000]) });
    out = out.embeddings;
  }
  const d = out.type === "float16" ? f16ToF32(out.data) : out.data;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    if (!Number.isFinite(v)) throw new Error(`${name} ${io}: non-finite output (broken fp16 path)`);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (mx - mn < 1e-4) throw new Error(`${name} ${io}: degenerate constant output (broken fp16 path)`);
}

async function createSession(name, onProgress) {
  // WebGPU first, WASM as a real fallback if the GPU path fails for any reason
  const providers = webgpuAvailable() ? ["webgpu", "wasm"] : ["wasm"];
  const f16ok = await webgpuF16Supported();
  let lastErr = null;

  for (const prov of providers) {
    // fp16 only on WebGPU with hardware shader-f16; WASM always fp32
    const variants = prov === "webgpu" && f16ok ? ["fp16", "fp32"] : ["fp32"];
    for (const variant of variants) {
      const url = MODELS[name][variant];
      try {
        onProgress?.(0, `downloading ${name} (${variant}, ${prov})…`);
        const buf = await fetchWithProgress(url, (recv, total) => {
          onProgress?.(total ? recv / total * 0.9 : 0.5,
            `${name}: ${fmtMB(recv)} / ${total ? fmtMB(total) : "?"}`);
        });
        onProgress?.(0.92, `${name}: compiling graph (${prov})…`);
        const sess = await ort.InferenceSession.create(buf, {
          executionProviders: [prov],
          graphOptimizationLevel: "all",
        });
        const io = variant === "fp16" ? "float16" : "float32";
        if (variant === "fp16") {
          onProgress?.(0.97, `${name}: validating fp16 numerics…`);
          await smokeTest(name, sess, io);
        }
        onProgress?.(1, `${name}: ready (${prov}, ${variant})`);
        console.log(`[flora] ${name} session ready: ${prov} / ${variant}`);
        setBackend(prov);
        return { sess, io };
      } catch (e) {
        console.warn(`[flora] ${name} ${variant} on ${prov} failed:`, e.message || e);
        lastErr = e;
      }
    }
  }
  throw lastErr || new Error(`cannot create session ${name}`);
}

export function getSession(name, onProgress) {
  // store the in-flight promise immediately so concurrent callers share
  // a single download + compile instead of racing into duplicates
  if (!sessions[name]) {
    sessions[name] = createSession(name, onProgress).catch((e) => {
      delete sessions[name]; // allow retry after a failure
      throw e;
    });
  }
  return sessions[name];
}

/** Wrap data as an ORT tensor matching the session's graph I/O type. */
function mkTensor(f32data, dims, io) {
  return io === "float16"
    ? new ort.Tensor("float16", f32ToF16(f32data), dims)
    : new ort.Tensor("float32", f32data, dims);
}

/** Read a tensor's data back as Float32Array regardless of its dtype. */
function tensorToF32(t) {
  return t.type === "float16" ? f16ToF32(t.data) : t.data;
}

/* ── Fusion: features → 400 parcels ─────────────────────────────────────── */

/**
 * text (T*384) f32, audio (T*384) f32, video (T*640) f32 → Float32Array (T*400)
 */
export async function runFusion(textF, audioF, videoF, T, onProgress) {
  const { sess, io } = await getSession("fusion", onProgress);
  const t0 = performance.now();
  const feeds = {
    text_features: mkTensor(textF, [1, T, 384], io),
    audio_features: mkTensor(audioF, [1, T, 384], io),
    video_features: mkTensor(videoF, [1, T, 640], io),
    subject_id: new ort.Tensor("int64", BigInt64Array.from([0n]), [1]),
  };
  const out = await sess.run(feeds);
  const ms = performance.now() - t0;
  return { parcels: tensorToF32(out.parcels), ms };
}

/* ── Video encoder: batched frames → (T,640) ────────────────────────────── */

export async function runVideoEncoder(chwBatch, T, onProgress) {
  const { sess, io } = await getSession("video", onProgress);
  const t0 = performance.now();
  const feeds = { pixel_values: mkTensor(chwBatch, [T, 3, 256, 256], io) };
  const out = await sess.run(feeds);
  const ms = performance.now() - t0;
  return { feats: tensorToF32(out.embeddings), ms }; // (T*640)
}

/* ── Audio encoder: mel → resampled (T,384) ─────────────────────────────── */

export async function runAudioEncoder(mel, T, onProgress) {
  const { sess, io } = await getSession("audio", onProgress);
  const t0 = performance.now();
  const feeds = { input_features: mkTensor(mel, [1, 80, 3000], io) };
  const out = await sess.run(feeds);
  const hidden = tensorToF32(out.embeddings); // (1500*384)
  const feats = resampleLinear(hidden, 1500, T, 384);
  const ms = performance.now() - t0;
  return { feats, ms };
}

/* ── Frame extraction (matches tools/extract_features_v3.py + processor) ── */

/**
 * Sample T frames uniformly; per frame: stretch to 288×288, center-crop 256,
 * rescale 1/255, flip RGB→BGR (MobileViT processor semantics).
 * Returns { batch: Float32Array(T*3*256*256), thumbs: string[] (dataURLs) }.
 */
export async function extractFrames(videoEl, T = 5, onFrame) {
  const dur = videoEl.duration;
  const cv = document.createElement("canvas");
  cv.width = 288; cv.height = 288;
  const cx = cv.getContext("2d", { willReadFrequently: true });
  const crop = document.createElement("canvas");
  crop.width = 256; crop.height = 256;
  const cc = crop.getContext("2d", { willReadFrequently: true });

  const batch = new Float32Array(T * 3 * 256 * 256);
  const thumbs = [];
  const plane = 256 * 256;

  for (let i = 0; i < T; i++) {
    const t = Math.min(dur - 0.05, (i * dur) / T);
    await new Promise((res) => {
      const onSeek = () => { videoEl.removeEventListener("seeked", onSeek); res(); };
      videoEl.addEventListener("seeked", onSeek);
      videoEl.currentTime = Math.max(0, t);
    });

    cx.drawImage(videoEl, 0, 0, 288, 288);            // stretch (square input)
    cc.drawImage(cv, 16, 16, 256, 256, 0, 0, 256, 256); // center crop
    const img = cc.getImageData(0, 0, 256, 256).data;

    const base = i * 3 * plane;
    const INV = 1 / 255;
    for (let p = 0, q = 0; p < plane; p++, q += 4) {
      batch[base + p] = img[q + 2] * INV;      // channel 0 = B (flip)
      batch[base + plane + p] = img[q + 1] * INV;
      batch[base + 2 * plane + p] = img[q] * INV;
    }
    thumbs.push(crop.toDataURL("image/jpeg", 0.8));
    onFrame?.(i + 1, T, thumbs[thumbs.length - 1]);
  }
  return { batch, thumbs };
}

/** Load a File into a <video> element (muted, paused at 0). */
export function loadVideoElement(file) {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.preload = "auto";
    v.muted = true;
    v.playsInline = true;
    v.onloadedmetadata = () => resolve(v);
    v.onerror = () => reject(new Error("cannot decode this video file"));
    v.src = URL.createObjectURL(file);
  });
}
