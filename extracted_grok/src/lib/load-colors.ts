import JSZip from "jszip";

export type HemiColors = {
  numFaces: number;
  numFrames: number;
  frames: Uint8Array[];
};

export type BrainColorSet = {
  leftTrue?: HemiColors;
  rightTrue?: HemiColors;
  leftPred?: HemiColors;
  rightPred?: HemiColors;
  numFrames: number;
  title?: string;
};

async function readHemi(zip: JSZip, hemi: "left" | "right", kind: "true" | "prediction") {
  const jsonName = `${hemi}-hemisphere-face-colors-binary-${kind}.json`;
  const jsonFile = zip.file(jsonName);
  if (!jsonFile) return undefined;
  const meta = JSON.parse(await jsonFile.async("string")) as {
    numFaces: number;
    numFrames: number;
    colorsBin: string;
    frameByteSize: number;
  };
  const binFile = zip.file(meta.colorsBin);
  if (!binFile) return undefined;
  const buf = await binFile.async("uint8array");
  const frames: Uint8Array[] = [];
  const frameSize = meta.frameByteSize || meta.numFaces * 3;
  for (let i = 0; i < meta.numFrames; i++) {
    frames.push(buf.subarray(i * frameSize, (i + 1) * frameSize));
  }
  return { numFaces: meta.numFaces, numFrames: meta.numFrames, frames };
}

const cache = new Map<string, Promise<BrainColorSet>>();

export function loadBrainColors(url: string): Promise<BrainColorSet> {
  const hit = cache.get(url);
  if (hit) return hit;
  const p = (async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load ${url}`);
    const zip = await JSZip.loadAsync(await res.arrayBuffer());
    const [leftTrue, rightTrue, leftPred, rightPred] = await Promise.all([
      readHemi(zip, "left", "true"),
      readHemi(zip, "right", "true"),
      readHemi(zip, "left", "prediction"),
      readHemi(zip, "right", "prediction"),
    ]);
    let title: string | undefined;
    const metaFile = zip.file("metadata.json");
    if (metaFile) {
      try {
        const meta = JSON.parse(await metaFile.async("string")) as { name?: string; title?: string };
        title = meta.name ?? meta.title;
      } catch {
        /* ignore */
      }
    }
    const numFrames = Math.max(
      leftTrue?.numFrames ?? 0,
      rightTrue?.numFrames ?? 0,
      leftPred?.numFrames ?? 0,
      rightPred?.numFrames ?? 0,
      1,
    );
    return { leftTrue, rightTrue, leftPred, rightPred, numFrames, title };
  })();
  cache.set(url, p);
  return p;
}
