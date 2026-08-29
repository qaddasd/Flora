import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ColorMode, SurfaceMode } from "@/lib/datasets";
import { loadBrainColors, type BrainColorSet, type HemiColors } from "@/lib/load-colors";

type Props = {
  colorsUrl: string;
  colorMode: ColorMode;
  surface: SurfaceMode;
  open: boolean;
  progress: number;
  onLoading?: (loading: boolean) => void;
};

type HemiMesh = {
  mesh: THREE.Mesh;
  colorAttr: THREE.BufferAttribute;
  faceCount: number;
};

type SceneApi = {
  left: HemiMesh;
  right: HemiMesh;
  trueLeft: HemiMesh;
  trueRight: HemiMesh;
  predLeft: HemiMesh;
  predRight: HemiMesh;
  compareGroup: THREE.Group;
  singleGroup: THREE.Group;
  colors?: BrainColorSet;
};

function toNonIndexed(geo: THREE.BufferGeometry) {
  return geo.index ? geo.toNonIndexed() : geo.clone();
}

function applyFaceColors(hemi: HemiMesh, colors: HemiColors | undefined, frame: number) {
  const attr = hemi.colorAttr;
  const arr = attr.array as Float32Array;
  const faces = hemi.faceCount;
  if (!colors || colors.frames.length === 0) {
    for (let i = 0; i < arr.length; i++) arr[i] = 0.16;
    attr.needsUpdate = true;
    return;
  }
  const f = Math.max(0, Math.min(colors.numFrames - 1, frame));
  const src = colors.frames[f]!;
  const n = Math.min(faces, colors.numFaces);
  for (let i = 0; i < n; i++) {
    const r = src[i * 3]! / 255;
    const g = src[i * 3 + 1]! / 255;
    const b = src[i * 3 + 2]! / 255;
    const o = i * 9;
    for (let v = 0; v < 3; v++) {
      arr[o + v * 3] = r;
      arr[o + v * 3 + 1] = g;
      arr[o + v * 3 + 2] = b;
    }
  }
  attr.needsUpdate = true;
}

function makeMaterial() {
  return new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.55,
    metalness: 0.04,
    side: THREE.DoubleSide,
    flatShading: true,
  });
}

function firstMesh(root: THREE.Object3D): THREE.Mesh {
  const meshes: THREE.Mesh[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
  });
  const mesh = meshes[0];
  if (!mesh) throw new Error("Missing hemisphere mesh");
  return mesh;
}

function buildHemi(normalRoot: THREE.Object3D, inflatedRoot: THREE.Object3D): HemiMesh {
  const nMesh = firstMesh(normalRoot);
  const iMesh = firstMesh(inflatedRoot);
  const geo = toNonIndexed(nMesh.geometry);
  const inf = toNonIndexed(iMesh.geometry);
  if (inf.attributes.position && inf.attributes.position.count === geo.attributes.position.count) {
    geo.morphAttributes.position = [inf.attributes.position.clone()];
    geo.morphTargetsRelative = false;
  }
  const colors = new Float32Array(geo.attributes.position.count * 3);
  colors.fill(0.16);
  const colorAttr = new THREE.BufferAttribute(colors, 3);
  geo.setAttribute("color", colorAttr);
  const mesh = new THREE.Mesh(geo, makeMaterial());
  mesh.morphTargetInfluences = [0];
  return { mesh, colorAttr, faceCount: geo.attributes.position.count / 3 };
}

function paintPair(
  left: HemiMesh,
  right: HemiMesh,
  leftSrc: HemiColors | undefined,
  rightSrc: HemiColors | undefined,
  frame: number,
  infl: number,
  split: number,
) {
  applyFaceColors(left, leftSrc, frame);
  applyFaceColors(right, rightSrc, frame);
  if (left.mesh.morphTargetInfluences) left.mesh.morphTargetInfluences[0] = infl;
  if (right.mesh.morphTargetInfluences) right.mesh.morphTargetInfluences[0] = infl;
  left.mesh.position.x = -split;
  right.mesh.position.x = split;
}

export function BrainViewer({
  colorsUrl,
  colorMode,
  surface,
  open,
  progress,
  onLoading,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<SceneApi | null>(null);
  const [ready, setReady] = useState(0);
  const onLoadingRef = useRef(onLoading);
  onLoadingRef.current = onLoading;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let disposed = false;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x080808, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080808);
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
    camera.position.set(0, 6, 44);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.minDistance = 18;
    controls.maxDistance = 90;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.58));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(14, 18, 22);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9bb7ff, 0.32);
    fill.position.set(-18, 8, -12);
    scene.add(fill);

    const singleGroup = new THREE.Group();
    const compareGroup = new THREE.Group();
    compareGroup.visible = false;
    scene.add(singleGroup);
    scene.add(compareGroup);

    const resize = () => {
      const w = el.clientWidth || 1;
      const h = el.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    onLoadingRef.current?.(true);
    const loader = new GLTFLoader();

    Promise.all([
      loader.loadAsync("/models/brain-left-hemisphere.glb"),
      loader.loadAsync("/models/brain-right-hemisphere.glb"),
      loader.loadAsync("/models/brain-left-hemisphere-inflated.glb"),
      loader.loadAsync("/models/brain-right-hemisphere-inflated.glb"),
    ])
      .then(([lN, rN, lI, rI]) => {
        if (disposed) return;
        const left = buildHemi(lN.scene, lI.scene);
        const right = buildHemi(rN.scene, rI.scene);
        singleGroup.add(left.mesh, right.mesh);

        const trueLeft = buildHemi(lN.scene, lI.scene);
        const trueRight = buildHemi(rN.scene, rI.scene);
        const predLeft = buildHemi(lN.scene, lI.scene);
        const predRight = buildHemi(rN.scene, rI.scene);
        const trueG = new THREE.Group();
        const predG = new THREE.Group();
        trueG.position.x = -15;
        predG.position.x = 15;
        trueG.add(trueLeft.mesh, trueRight.mesh);
        predG.add(predLeft.mesh, predRight.mesh);
        compareGroup.add(trueG, predG);

        const box = new THREE.Box3().setFromObject(singleGroup);
        const center = box.getCenter(new THREE.Vector3());
        singleGroup.position.sub(center);
        trueG.position.y -= center.y;
        predG.position.y -= center.y;

        apiRef.current = {
          left,
          right,
          trueLeft,
          trueRight,
          predLeft,
          predRight,
          compareGroup,
          singleGroup,
        };
        setReady((n) => n + 1);
      })
      .catch((err) => {
        console.error(err);
        onLoadingRef.current?.(false);
      });

    let raf = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      apiRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    onLoadingRef.current?.(true);
    loadBrainColors(colorsUrl)
      .then((colors) => {
        if (cancelled || !apiRef.current) return;
        apiRef.current.colors = colors;
        setReady((n) => n + 1);
        onLoadingRef.current?.(false);
      })
      .catch((err) => {
        console.warn(err);
        onLoadingRef.current?.(false);
      });
    return () => {
      cancelled = true;
    };
  }, [colorsUrl, ready === 0 ? 0 : 1]);

  useEffect(() => {
    const a = apiRef.current;
    if (!a) return;
    const colors = a.colors;
    const frames = Math.max(1, colors?.numFrames ?? 1);
    const frame = Math.min(frames - 1, Math.floor(Math.max(0, progress) * 0.999 * frames));
    const compare = colorMode === "compare";
    a.compareGroup.visible = compare;
    a.singleGroup.visible = !compare;
    const infl = surface === "inflated" ? 1 : 0;
    const split = open && !compare ? 8 : 0;

    if (compare) {
      paintPair(a.trueLeft, a.trueRight, colors?.leftTrue, colors?.rightTrue, frame, infl, 0);
      paintPair(a.predLeft, a.predRight, colors?.leftPred, colors?.rightPred, frame, infl, 0);
    } else {
      const useTrue = colorMode === "true";
      paintPair(
        a.left,
        a.right,
        useTrue ? colors?.leftTrue : colors?.leftPred,
        useTrue ? colors?.rightTrue : colors?.rightPred,
        frame,
        infl,
        split,
      );
    }
  }, [ready, colorMode, surface, open, progress, colorsUrl]);

  return <div ref={wrapRef} className="absolute inset-0 h-full w-full touch-none" />;
}
