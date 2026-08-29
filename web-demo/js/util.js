/** Shared helpers: binary IO, float16 decode, progress fetch, misc math. */

/** Absolute URL of the site root, resolved from this module's location (js/). */
export const BASE = new URL("../", import.meta.url).href;

export async function fetchBin(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  return resp.arrayBuffer();
}

export async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  return resp.json();
}

/** Fetch with download progress callback(receivedBytes, totalBytes). */
export async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  const total = +resp.headers.get("Content-Length") || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

/** float16 (Uint16Array view of buffer) → Float32Array. */
export function f16ToF32(buffer) {
  const u16 = buffer instanceof Uint16Array ? buffer : new Uint16Array(buffer);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    let v;
    if (e === 0) v = f * 2 ** -24;
    else if (e === 31) v = f ? NaN : Infinity;
    else v = (1 + f / 1024) * 2 ** (e - 15);
    out[i] = s ? -v : v;
  }
  return out;
}

/** Float32Array → float16 bits (Uint16Array), round-to-nearest-even. */
export function f32ToF16(f32arr) {
  const out = new Uint16Array(f32arr.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < f32arr.length; i++) {
    f32[0] = f32arr[i];
    const x = u32[0];
    const sign = (x >> 16) & 0x8000;
    const exp = ((x >> 23) & 0xff) - 112; // rebias 127 → 15
    let mant = x & 0x7fffff;
    let h;
    if (exp >= 31) h = sign | 0x7c00;                     // overflow → Inf
    else if (exp <= 0) {                                   // subnormal / zero
      if (exp < -10) h = sign;
      else {
        mant |= 0x800000;
        h = sign | (mant >> (14 - exp));
      }
    } else {
      h = sign | (exp << 10) | (mant >> 13);
      if (mant & 0x1000) h += 1;                           // round-to-nearest
    }
    out[i] = h;
  }
  return out;
}

export function f32(buffer) { return new Float32Array(buffer); }
export function u32(buffer) { return new Uint32Array(buffer); }
export function u16(buffer) { return new Uint16Array(buffer); }

/** Robust min/max via percentiles (for activation normalization). */
export function robustRange(arr, lo = 2, hi = 98) {
  const sorted = Float32Array.from(arr).sort();
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p / 100))];
  return [q(lo), q(hi)];
}

/** align_corners=False linear resample of (T_in, D) → (T_out, D). */
export function resampleLinear(x, tIn, tOut, D) {
  const out = new Float32Array(tOut * D);
  if (tOut === 1) { out.set(x.subarray(0, D)); return out; }
  for (let i = 0; i < tOut; i++) {
    const src = i * (tIn - 1) / (tOut - 1);
    const i0 = Math.floor(src), i1 = Math.min(tIn - 1, i0 + 1);
    const w = src - i0;
    for (let d = 0; d < D; d++) {
      out[i * D + d] = x[i0 * D + d] * (1 - w) + x[i1 * D + d] * w;
    }
  }
  return out;
}

/** Simple tween helper: each frame value += (target-value)*k. */
export function damp(current, target, k = 0.12) {
  return current + (target - current) * k;
}

export function fmtMB(bytes) { return (bytes / 1e6).toFixed(1) + " MB"; }
export function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms.toFixed(0) + " ms"; }

/** Yield to the event loop so long loops don't freeze the UI. */
export function tick() { return new Promise((r) => setTimeout(r, 0)); }
