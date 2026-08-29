/**
 * Dev self-test — runs only with ?autotest=1 in the URL.
 * Verifies the riskiest pieces end-to-end and reports into the DOM + title:
 *   1. manifest + binary data formats
 *   2. brain geometry buffers
 *   3. a real fusion-model forward pass (WebGPU, WASM fallback)
 */

import { BASE, fetchBin, fetchJSON, f16ToF32, u16, f32 } from "./util.js";
import { runFusion, getBackend } from "./inference.js";

export async function runSelfTest() {
  fetch("/report", { method: "POST", body: "AUTOTEST:STARTED" }).catch(() => {});
  const lines = [];
  const ok = (name, cond, extra = "") => {
    lines.push(`${cond ? "PASS" : "FAIL"} ${name}${extra ? " — " + extra : ""}`);
    return cond;
  };
  let allOk = true;

  try {
    // 1 · manifest + one template prediction file
    const man = await fetchJSON(BASE + "data/templates.json");
    allOk &= ok("manifest", man.templates.length >= 1, `${man.templates.length} templates`);
    const t0 = man.templates[0];
    const pred = f16ToF32(await fetchBin(`${BASE}${t0.pred}`));
    allOk &= ok("template pred shape", pred.length === t0.T * 400, `${pred.length} floats`);
    allOk &= ok("template pred finite", pred.every(Number.isFinite));
    const feats = f16ToF32(await fetchBin(`${BASE}${t0.feats}`));
    allOk &= ok("template feats shape", feats.length === t0.T * 1408, `${feats.length} floats`);

    // 2 · brain assets
    const meta = await fetchJSON(BASE + "assets/brain_meta.json");
    const pial = f32(await fetchBin(BASE + "assets/brain_pial.bin"));
    allOk &= ok("pial verts", pial.length === meta.n_vertices * 3);
    const parcels = u16(await fetchBin(BASE + "assets/brain_parcels.bin"));
    allOk &= ok("parcel labels", parcels.length === meta.n_vertices);
    let mx = 0;
    for (const p of parcels) if (p > mx) mx = p;
    allOk &= ok("parcel range", mx === 400, `max label ${mx}`);

    // 3 · live fusion forward pass with real template features
    const T = t0.T;
    const text = new Float32Array(T * 384);
    const audio = new Float32Array(T * 384);
    const video = new Float32Array(T * 640);
    for (let t = 0; t < T; t++) {
      const b = t * 1408;
      text.set(feats.subarray(b, b + 384), t * 384);
      audio.set(feats.subarray(b + 384, b + 768), t * 384);
      video.set(feats.subarray(b + 768, b + 1408), t * 640);
    }
    const { parcels: outP, ms } = await runFusion(text, audio, video, T);
    allOk &= ok("fusion output shape", outP.length === T * 400, `${outP.length} floats`);
    let finite = true;
    for (const v of outP) if (!Number.isFinite(v)) { finite = false; break; }
    allOk &= ok("fusion output finite", finite);
    lines.push(`INFO backend=${getBackend()} fusion_ms=${ms.toFixed(0)}`);

    // sanity: fusion(output) on template feats ≈ precomputed pred (same model)
    let maxD = 0;
    for (let i = 0; i < outP.length; i++) maxD = Math.max(maxD, Math.abs(outP[i] - pred[i]));
    allOk &= ok("fusion matches precomputed", maxD < 0.15, `max|diff|=${maxD.toFixed(4)}`);
  } catch (e) {
    lines.push(`FAIL exception: ${e.message || e}`);
    allOk = false;
  }

  const verdict = allOk ? "ALL-PASS" : "HAS-FAIL";
  const report = `AUTOTEST:${verdict}\n` + lines.join("\n");
  document.title = `AUTOTEST:${verdict}`;
  // persist for headless runs (dev_server.py saves it to autotest-report.txt)
  fetch("/report", { method: "POST", body: report }).catch(() => {});
  const div = document.createElement("pre");
  div.id = "autotest-result";
  div.textContent = report;
  div.style.cssText =
    "position:fixed;left:8px;bottom:8px;z-index:999;background:#000d;color:#7f7;padding:8px 12px;font:11px monospace;border-radius:8px;white-space:pre-wrap;max-width:90vw";
  document.body.appendChild(div);
  console.log(div.textContent);
}
