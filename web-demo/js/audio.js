/**
 * Whisper-exact log-mel feature extraction in JS.
 *
 * Mirrors transformers.WhisperFeatureExtractor:
 *   - pad/trim waveform to 480,000 samples (30 s @ 16 kHz)
 *   - reflect-pad 200 samples each side (center=True)
 *   - frames of 400 @ hop 160 → 3,001 frames, drop the last → 3,000
 *   - periodic Hann window folded into precomputed rDFT matrices (201×400)
 *   - power spectrum → 80-bin mel filterbank → log10 → clamp(max−8) → (x+4)/4
 *
 * Verified against transformers to max|diff| ≈ 1.5e-06 (see tools/export_audio_assets.py).
 */

import { BASE, fetchBin, f32, tick } from "./util.js";

const N_SAMPLES = 480000;
const FFT = 400;
const HOP = 160;
const PAD = FFT / 2;
const N_BINS = 201;
const N_MELS = 80;
const N_FRAMES_OUT = 3000; // whisper window

let dft = null; // { Wr, Wi }
let mel = null; // Float32Array (80*201)

export async function loadAudioAssets() {
  if (dft && mel) return;
  const [dftBuf, melBuf] = await Promise.all([
    fetchBin(BASE + "assets/whisper_dft.bin"),
    fetchBin(BASE + "assets/whisper_mel.bin"),
  ]);
  const all = f32(dftBuf);
  const half = N_BINS * FFT;
  dft = { Wr: all.subarray(0, half), Wi: all.subarray(half, half * 2) };
  mel = f32(melBuf);
}

/**
 * pcm16k: Float32Array mono @ 16 kHz (any length ≤ 30 s used).
 * onProgress(frac) optional. Returns Float32Array (80 * 3000), mel-bin major.
 */
export async function computeWhisperMel(pcm16k, onProgress) {
  await loadAudioAssets();

  // pad/trim to 30 s
  const x = new Float32Array(N_SAMPLES);
  x.set(pcm16k.subarray(0, Math.min(pcm16k.length, N_SAMPLES)));

  // reflect padding (like torch.stft center=True, pad_mode=reflect)
  const xp = new Float32Array(N_SAMPLES + 2 * PAD);
  for (let i = 0; i < PAD; i++) xp[i] = x[PAD - i];
  xp.set(x, PAD);
  for (let i = 0; i < PAD; i++) xp[PAD + N_SAMPLES + i] = x[N_SAMPLES - 2 - i];

  const nFrames = N_FRAMES_OUT + 1; // 3001, drop last later
  const { Wr, Wi } = dft;

  // power spectrum per frame
  const power = new Float32Array(nFrames * N_BINS);
  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    const pOff = f * N_BINS;
    for (let j = 0; j < N_BINS; j++) {
      const row = j * FFT;
      let sr = 0, si = 0;
      for (let n = 0; n < FFT; n++) {
        const v = xp[off + n];
        sr += Wr[row + n] * v;
        si += Wi[row + n] * v;
      }
      power[pOff + j] = sr * sr + si * si;
    }
    if (f % 200 === 199) { onProgress?.(f / nFrames * 0.7); await tick(); }
  }

  // mel projection + log normalization (per whole clip, whisper style)
  const out = new Float32Array(N_MELS * N_FRAMES_OUT);
  const logSpec = new Float32Array(N_FRAMES_OUT * N_MELS);
  let maxLog = -Infinity;
  for (let f = 0; f < N_FRAMES_OUT; f++) {
    const pOff = f * N_BINS;
    for (let m = 0; m < N_MELS; m++) {
      const mRow = m * N_BINS;
      let e = 0;
      for (let j = 0; j < N_BINS; j++) e += mel[mRow + j] * power[pOff + j];
      const l = Math.log10(Math.max(e, 1e-10));
      logSpec[f * N_MELS + m] = l;
      if (l > maxLog) maxLog = l;
    }
    if (f % 600 === 599) { onProgress?.(0.7 + f / N_FRAMES_OUT * 0.3); await tick(); }
  }
  const floor = maxLog - 8.0;
  for (let f = 0; f < N_FRAMES_OUT; f++) {
    for (let m = 0; m < N_MELS; m++) {
      let l = logSpec[f * N_MELS + m];
      if (l < floor) l = floor;
      out[m * N_FRAMES_OUT + f] = (l + 4.0) / 4.0;
    }
  }
  onProgress?.(1);
  return out; // (80, 3000)
}

/** Draw a mel spectrogram (80×3000) onto a canvas — used in the pipeline view. */
export function drawMel(canvas, melData) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const img = ctx.createImageData(W, H);
  for (let py = 0; py < H; py++) {
    const m = N_MELS - 1 - Math.floor(py / H * N_MELS);
    for (let px = 0; px < W; px++) {
      const f = Math.floor(px / W * N_FRAMES_OUT);
      const v = Math.max(0, Math.min(1, (melData[m * N_FRAMES_OUT + f] + 1) / 1.6));
      const idx = (py * W + px) * 4;
      img.data[idx] = 30 + 225 * v;
      img.data[idx + 1] = 20 + 130 * v * v;
      img.data[idx + 2] = 40 + 60 * v;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Draw a waveform overview. */
export function drawWaveform(canvas, pcm) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0a0a0c";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#ff8a3d";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const per = Math.max(1, Math.floor(pcm.length / W));
  for (let px = 0; px < W; px++) {
    let mn = 1, mx = -1;
    const off = px * per;
    for (let i = 0; i < per; i += 4) {
      const v = pcm[off + i] || 0;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const y0 = (1 - (mx * 0.9 + 1) / 2) * H;
    const y1 = (1 - (mn * 0.9 + 1) / 2) * H;
    ctx.moveTo(px + 0.5, y0);
    ctx.lineTo(px + 0.5, Math.max(y1, y0 + 1));
  }
  ctx.stroke();
}

/** Decode a video file's audio track to 16 kHz mono Float32Array. */
export async function decodeAudio16k(file) {
  const buf = await file.arrayBuffer();
  const ctx = new AudioContext({ sampleRate: 16000 });
  try {
    const audio = await ctx.decodeAudioData(buf);
    return audio.getChannelData(0);
  } finally {
    ctx.close();
  }
}
