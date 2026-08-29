/**
 * 3D cortical surface viewer (fsaverage5, 20,484 vertices).
 *
 * - White matter-style shading driven by sulcal depth (like the reference UI)
 * - Hot activity colormap overlaid per-vertex, updated per animation frame
 * - GPU morph between pial and inflated surfaces (Normal / Inflated)
 * - Hemispheres swing apart around the vertical axis (Open / Close)
 * - Compare mode: two surfaces side-by-side via scissored viewports
 */

import * as THREE from "three";
import { BASE, fetchBin, fetchJSON, f32, u32, u16, damp } from "./util.js";

const VERT_SRC = /* glsl */ `
attribute vec3 aInflated;
attribute float aSulc;
attribute float aActivity;
uniform float uInflation;
uniform float uOpen;
varying float vSulc;
varying float vAct;
varying vec3 vNormal;
varying vec3 vViewDir;

void main() {
  vec3 pos = mix(position, aInflated, uInflation);
  vec3 nrm = normal;
  float side = position.x < 0.0 ? -1.0 : 1.0;
  float ang = uOpen * side * 1.1;
  float ca = cos(ang), sa = sin(ang);
  pos = vec3(pos.x * ca - pos.y * sa, pos.x * sa + pos.y * ca, pos.z);
  nrm = vec3(nrm.x * ca - nrm.y * sa, nrm.x * sa + nrm.y * ca, nrm.z);
  pos.x += uOpen * side * 0.7;
  vSulc = aSulc;
  vAct = aActivity;
  vNormal = normalize(normalMatrix * nrm);
  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  vViewDir = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}
`;

const FRAG_SRC = /* glsl */ `
precision highp float;
varying float vSulc;
varying float vAct;
varying vec3 vNormal;
varying vec3 vViewDir;

vec3 hot(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = vec3(0.22, 0.0, 0.0);
  c = mix(c, vec3(0.78, 0.10, 0.02), smoothstep(0.00, 0.38, t));
  c = mix(c, vec3(1.00, 0.45, 0.00), smoothstep(0.38, 0.72, t));
  c = mix(c, vec3(1.00, 0.93, 0.50), smoothstep(0.72, 1.00, t));
  return c;
}

void main() {
  vec3 n = normalize(vNormal);
  float base = 1.0 - 0.44 * vSulc;
  vec3 l1 = normalize(vec3(-0.55, 0.25, 0.80));
  vec3 l2 = normalize(vec3(0.65, -0.45, 0.35));
  float diff = 0.60 + 0.40 * max(dot(n, l1), 0.0) + 0.16 * max(dot(n, l2), 0.0);
  vec3 col = vec3(base) * diff;
  float rim = pow(1.0 - max(dot(n, normalize(vViewDir)), 0.0), 2.5);
  col += rim * 0.075;
  float a = clamp(vAct, 0.0, 1.0);
  float mask = smoothstep(0.52, 0.86, a); // only the top of the range lights up — focal hot spots
  vec3 hc = hot(pow(a, 1.55));
  col = mix(col, hc * (0.72 + 0.45 * diff), mask * 0.96);
  gl_FragColor = vec4(col, 1.0);
}
`;

export class BrainViewer {
  static async create(canvas) {
    const v = new BrainViewer(canvas);
    await v._load();
    return v;
  }

  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camDist = 3.55; // slightly wider than 1:1 so the head silhouette stays visible
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.mode = "predicted";
    this.uInflation = { pred: 0, true: 0, target: 0 };
    this.uOpen = { pred: 0, true: 0, target: 0 };
    this.yaw = 0;         // around vertical z
    this.pitch = 0;       // tilt
    this.targetYaw = 0;
    this.targetPitch = 0;
    this.autoRotate = true;
    this.lastInteract = 0;

    this._setupInteraction();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  async _load() {
    const [pial, infl, faces, sulc, parcels, meta] = await Promise.all([
      fetchBin(BASE + "assets/brain_pial.bin"),
      fetchBin(BASE + "assets/brain_inflated.bin"),
      fetchBin(BASE + "assets/brain_faces.bin"),
      fetchBin(BASE + "assets/brain_sulc.bin"),
      fetchBin(BASE + "assets/brain_parcels.bin"),
      fetchJSON(BASE + "assets/brain_meta.json"),
    ]);
    this.meta = meta;
    this.parcelLabels = u16(parcels);
    this.nVertices = meta.n_vertices;

    const posArr = f32(pial);
    const inflArr = f32(infl);
    const faceRaw = u32(faces);
    const sulcArr = f32(sulc);

    // Drop triangles straddling the midline: when the hemispheres swing
    // apart they stretch into a visible band connecting the two halves.
    const kept = [];
    for (let i = 0; i < faceRaw.length; i += 3) {
      const a = faceRaw[i], b = faceRaw[i + 1], c = faceRaw[i + 2];
      const sa = posArr[a * 3] < 0, sb = posArr[b * 3] < 0, sc = posArr[c * 3] < 0;
      if (sa === sb && sb === sc) kept.push(a, b, c);
    }
    const faceArr = new Uint32Array(kept);

    // normalize sulc → 0 crown .. 1 groove (robust)
    let mn = Infinity, mx = -Infinity;
    for (const s of sulcArr) { if (s < mn) mn = s; if (s > mx) mx = s; }
    const sulcN = new Float32Array(sulcArr.length);
    for (let i = 0; i < sulcArr.length; i++) {
      sulcN[i] = Math.max(0, Math.min(1, (sulcArr[i] - mn) / (mx - mn || 1)));
    }

    const build = () => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
      g.setAttribute("aInflated", new THREE.BufferAttribute(inflArr, 3));
      g.setAttribute("aSulc", new THREE.BufferAttribute(sulcN, 1));
      const act = new Float32Array(this.nVertices);
      g.setAttribute("aActivity", new THREE.BufferAttribute(act, 1));
      g.setIndex(new THREE.BufferAttribute(faceArr, 1));
      g.computeVertexNormals();
      return g;
    };

    this.geomPred = build();
    this.geomTrue = build();

    const mkMesh = (geom) => {
      const mat = new THREE.ShaderMaterial({
        vertexShader: VERT_SRC,
        fragmentShader: FRAG_SRC,
        uniforms: {
          uInflation: { value: 0 },
          uOpen: { value: 0 },
        },
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geom, mat);
      this.group.add(mesh);
      return mesh;
    };

    this.meshPred = mkMesh(this.geomPred);
    this.meshTrue = mkMesh(this.geomTrue);
    this.meshTrue.visible = false;

    this._resize();
  }

  /* ── public API ──────────────────────────────────────────────────────── */

  setSurface(surface) { this.uInflation.target = surface === "inflated" ? 1 : 0; }
  setOpen(open) { this.uOpen.target = open ? 1 : 0; }

  setMode(mode) {
    this.mode = mode;
    this.meshTrue.visible = mode === "true" || mode === "compare";
    this._resize();
  }

  /** verts: Float32Array(nVertices) already normalized 0..1, or null to clear. */
  setActivity(verts, which = "pred") {
    const geom = which === "pred" ? this.geomPred : this.geomTrue;
    if (!geom) return;
    const attr = geom.getAttribute("aActivity");
    if (verts) attr.array.set(verts.subarray(0, this.nVertices));
    else attr.array.fill(0);
    attr.needsUpdate = true;
  }

  clearActivity() { this.setActivity(null, "pred"); this.setActivity(null, "true"); }

  resize() { this._resize(); }

  /* ── internals ───────────────────────────────────────────────────────── */

  _resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _setupInteraction() {
    const c = this.canvas;
    let dragging = false, px = 0, py = 0;

    c.addEventListener("pointerdown", (e) => {
      dragging = true; px = e.clientX; py = e.clientY;
      this.lastInteract = performance.now();
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointerup", () => { dragging = false; });
    c.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      this.targetYaw += (e.clientX - px) * 0.006;
      this.targetPitch += (e.clientY - py) * 0.004;
      this.targetPitch = Math.max(-0.9, Math.min(0.9, this.targetPitch));
      px = e.clientX; py = e.clientY;
      this.lastInteract = performance.now();
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.camDist = Math.max(1.7, Math.min(5.5, this.camDist + e.deltaY * 0.0022));
      this.lastInteract = performance.now();
    }, { passive: false });
  }

  _animate(t) {
    requestAnimationFrame(this._animate);
    if (!this.meshPred) return;

    // tween uniforms
    this.uInflation.pred = damp(this.uInflation.pred, this.uInflation.target, 0.10);
    this.uInflation.true = damp(this.uInflation.true, this.uInflation.target, 0.10);
    this.uOpen.pred = damp(this.uOpen.pred, this.uOpen.target, 0.10);
    this.uOpen.true = damp(this.uOpen.true, this.uOpen.target, 0.10);
    this.meshPred.material.uniforms.uInflation.value = this.uInflation.pred;
    this.meshPred.material.uniforms.uOpen.value = this.uOpen.pred;
    this.meshTrue.material.uniforms.uInflation.value = this.uInflation.true;
    this.meshTrue.material.uniforms.uOpen.value = this.uOpen.true;

    // idle auto-rotation
    if (performance.now() - this.lastInteract > 3500 && this.autoRotate) {
      this.targetYaw += 0.0016;
    }
    this.yaw = damp(this.yaw, this.targetYaw, 0.14);
    this.pitch = damp(this.pitch, this.targetPitch, 0.14);
    this.group.rotation.set(0, this.pitch, this.yaw);

    // camera: left-lateral default, frontal to the left;
    // compare mode pulls back so the two brains don't touch at the seam
    const dist = this.camDist * (this.mode === "compare" ? 1.3 : 1.0);
    this.camera.position.set(-dist, 0.35, 0.45);
    this.camera.lookAt(0, 0, 0);

    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const r = this.renderer;

    if (this.mode === "compare" && this.meshTrue.visible) {
      r.setScissorTest(true);
      const hw = Math.floor(w / 2);
      this.camera.aspect = hw / h;
      this.camera.updateProjectionMatrix();
      // left: true, right: predicted
      this.meshPred.visible = false;
      this.meshTrue.visible = true;
      r.setViewport(0, 0, hw, h);
      r.setScissor(0, 0, hw, h);
      r.render(this.scene, this.camera);
      this.meshPred.visible = true;
      this.meshTrue.visible = false;
      r.setViewport(hw, 0, w - hw, h);
      r.setScissor(hw, 0, w - hw, h);
      r.render(this.scene, this.camera);
      this.meshTrue.visible = true;
      r.setScissorTest(false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    } else {
      const showTrueOnly = this.mode === "true";
      this.meshPred.visible = !showTrueOnly;
      this.meshTrue.visible = showTrueOnly;
      r.setViewport(0, 0, w, h);
      r.render(this.scene, this.camera);
    }
  }

  /** pause/resume idle rotation (during playback we hold the view steady) */
  setAutoRotate(on) { this.autoRotate = on; }
}
