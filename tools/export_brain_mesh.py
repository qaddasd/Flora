"""Export fsaverage5 brain surfaces + Schaefer-400 labels to compact binary files
for the Flora web demo (Three.js).

Outputs (little-endian) in web-demo/assets/:
  brain_pial.bin      float32[20484*3]  pial surface vertex coords (L then R hemi)
  brain_inflated.bin  float32[20484*3]  inflated surface vertex coords
  brain_faces.bin     uint32[N*3]       triangle indices (L then R, R offset by 10242)
  brain_sulc.bin      float32[20484]    sulcal depth (for shading)
  brain_parcels.bin   uint16[20484]     Schaefer-400 parcel id per vertex (0 = medial wall)
  brain_meta.json     counts and bounding boxes
"""

import json
from pathlib import Path

import numpy as np
from nilearn import datasets, surface

OUT = Path(__file__).resolve().parent.parent / "web-demo" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

print("Fetching fsaverage5 ...")
fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

print("Fetching Schaefer-400 surface labels (fsaverage5 .annot from CBIG) ...")
import urllib.request
import tempfile

CBIG_URL = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            "Parcellations/FreeSurfer5.3/fsaverage5/label/"
            "{hemi}.Schaefer2018_400Parcels_7Networks_order.annot")

import nibabel as nib

def load_schaefer_annot(hemi):
    """Returns uint16 parcel ids per vertex: 0 = medial wall, 1..200 (L) / 201..400 (R)."""
    url = CBIG_URL.format(hemi="lh" if hemi == "left" else "rh")
    with tempfile.NamedTemporaryFile(suffix=".annot", delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        labels, ctab, names = nib.freesurfer.read_annot(f.name)
    # nibabel returns per-vertex ids 0..200 (0 = medial wall, 1..200 = LH parcels)
    ids = np.asarray(labels).astype(np.int64)
    if ids.max() > 200:  # encoded RGB form -> map via ctab row index
        row_of_label = {int(v): i for i, v in enumerate(ctab[:, 4])}
        ids = np.array([row_of_label.get(int(v), 0) for v in ids])
    ids = ids.astype(np.uint16)
    if hemi == "right":
        ids = np.where(ids > 0, ids + 200, 0).astype(np.uint16)
    return ids

hemi_pos = []
hemi_infl = []
hemi_faces = []
hemi_sulc = []
hemi_labels = []
offset = 0

for hemi in ["left", "right"]:
    pial = surface.load_surf_mesh(fs[f"pial_{hemi}"])
    infl = surface.load_surf_mesh(fs[f"infl_{hemi}"])
    sulc = surface.load_surf_data(fs[f"sulc_{hemi}"]).astype(np.float32)

    lab_ids = load_schaefer_annot(hemi)

    print(f"  {hemi}: {pial.coordinates.shape[0]} vertices, "
          f"{pial.faces.shape[0]} faces, parcels={len(np.unique(lab_ids[lab_ids>0]))}")

    hemi_pos.append(pial.coordinates.astype(np.float32))
    hemi_infl.append(infl.coordinates.astype(np.float32))
    hemi_faces.append((pial.faces.astype(np.uint32) + offset))
    hemi_sulc.append(sulc)
    hemi_labels.append(lab_ids)
    offset += pial.coordinates.shape[0]

pos = np.concatenate(hemi_pos, axis=0)
infl = np.concatenate(hemi_infl, axis=0)
faces = np.concatenate(hemi_faces, axis=0)
sulc = np.concatenate(hemi_sulc, axis=0)
labels = np.concatenate(hemi_labels, axis=0)

# Normalize coordinates: center at origin, scale so max abs = 1 (Three.js friendly)
center = (pos.min(axis=0) + pos.max(axis=0)) / 2
pos_c = pos - center
scale = np.abs(pos_c).max()
pos_n = pos_c / scale
infl_n = (infl - center) / scale

pos_n.astype("<f4").tofile(OUT / "brain_pial.bin")
infl_n.astype("<f4").tofile(OUT / "brain_inflated.bin")
faces.astype("<u4").tofile(OUT / "brain_faces.bin")
sulc.astype("<f4").tofile(OUT / "brain_sulc.bin")
labels.astype("<u2").tofile(OUT / "brain_parcels.bin")

# Parcel centroids (for heat-spot glow placement) in normalized coords
centroids = {}
for p in range(1, 401):
    idx = np.where(labels == p)[0]
    if len(idx):
        centroids[p] = pos_n[idx].mean(axis=0).tolist()

meta = {
    "n_vertices": int(pos.shape[0]),
    "n_vertices_per_hemi": int(offset // 2),
    "n_faces": int(faces.shape[0]),
    "n_parcels": int(labels.max()),
    "center": center.tolist(),
    "scale": float(scale),
    "parcel_centroids": centroids,
}
with open(OUT / "brain_meta.json", "w") as f:
    json.dump(meta, f)

print("Saved:",
      f"pial={pos_n.nbytes/1e6:.2f}MB inflated={infl_n.nbytes/1e6:.2f}MB",
      f"faces={faces.nbytes/1e6:.2f}MB sulc={sulc.nbytes/1e3:.0f}KB",
      f"parcels={labels.nbytes/1e3:.0f}KB")
print("Parcels found:", labels.max(), "unique non-zero:", len(np.unique(labels[labels>0])))
