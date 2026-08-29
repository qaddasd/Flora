"""Precompute Flora predictions for the web demo.

A) Synthetic template clips (web-demo/videos/*.mp4):
     run the REAL extraction pipeline (MobileViT-S + Whisper-Tiny encoder)
     -> FloraV3 -> parcels (T,400) -> saved as float16 .bin

B) Dataset clips (data/features/*.pt, real teacher fMRI):
     FloraV3 on stored features -> parcels -> mapped to 20484 vertices
     via Schaefer-400 labels; saved together with the teacher map.
     Also reports vertex-level Pearson r (validation of the mapping).

Outputs in web-demo/data/ + templates.json manifest.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flora.v3_model import FloraV3
from flora.extract_features_v3 import (
    load_video_frames, load_audio_waveform,
    extract_video_features, extract_audio_features,
)

VID_DIR = ROOT / "web-demo" / "videos"
DATA_DIR = ROOT / "web-demo" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "checkpoints" / "best-epoch=052-val" / "pearson_r=0.7278.ckpt"
T = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPLATES = [
    ("ocean_waves",  "Ocean Waves",  "Slow swell and foam over deep water."),
    ("night_drive",  "Night Drive",  "Headlights and neon streaking past."),
    ("forest_birds", "Forest Birds", "Dappled light and birdsong."),
    ("fireplace",    "Fireplace",    "Flickering flames with crackle."),
    ("city_rain",    "City Rain",    "Rain streaks over a night city."),
    ("nebula",       "Nebula Drift", "Slow cosmic clouds, ambient pad."),
]

DATASET_CLIPS = ["000000", "000001", "000002"]  # real teacher fMRI clips


def save_f16(path, arr):
    arr.astype("<f2").tofile(path)


def parcels_to_vertices(parcels_tx400, labels):
    """parcels (T,400) -> (T, n_vertices) using the SAME block mapping that
    training used (flora/v3_dataset._map_to_schaefer400):
    parcel i -> vertices [i*chunk, (i+1)*chunk), chunk = V//400."""
    T, P = parcels_tx400.shape
    V = labels.shape[0]
    chunk = V // P
    out = np.zeros((T, V), dtype=np.float32)
    for i in range(P):
        out[:, i * chunk:(i + 1) * chunk] = parcels_tx400[:, i:i + 1]
    return out


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")
    model = FloraV3.load_from_checkpoint(str(CKPT), map_location="cpu")
    model.eval().to(DEVICE)

    labels = np.fromfile(ROOT / "web-demo/assets/brain_parcels.bin", dtype="<u2")

    manifest = {"templates": [], "dataset_clips": []}

    # ── A) synthetic template clips ──────────────────────────────────────
    for stem, title, desc in TEMPLATES:
        mp4 = VID_DIR / f"{stem}.mp4"
        if not mp4.exists():
            print(f"  !! missing {mp4.name}, skipping")
            continue
        frames = load_video_frames(mp4, n_frames=T)
        video_feat = extract_video_features(frames, DEVICE, T)
        waveform = load_audio_waveform(mp4)
        audio_feat = extract_audio_features(waveform, DEVICE, T)
        text_feat = torch.zeros(T, 384)

        out = model(
            text_feat[None].to(DEVICE), audio_feat[None].to(DEVICE),
            video_feat[None].to(DEVICE), torch.zeros(1, dtype=torch.long, device=DEVICE),
        )
        pred = out["prediction"].float().cpu().numpy()  # (T,400)
        assert pred.shape == (T, 400), pred.shape
        save_f16(DATA_DIR / f"{stem}_pred.bin", pred)

        # Save features (T,1408) = [text 384 | audio 384 | video 640] for In-Silico mode
        feats = np.concatenate(
            [text_feat.numpy(), audio_feat.numpy(), video_feat.numpy()], axis=1
        ).astype(np.float32)
        save_f16(DATA_DIR / f"{stem}_feats.bin", feats)

        manifest["templates"].append({
            "id": stem, "title": title, "desc": desc,
            "video": f"videos/{stem}.mp4", "thumb": f"videos/{stem}.jpg",
            "pred": f"data/{stem}_pred.bin", "feats": f"data/{stem}_feats.bin",
            "T": T, "kind": "parcels",
        })
        print(f"  {stem}: pred std={pred.std():.3f} range=[{pred.min():.2f},{pred.max():.2f}]")

    # ── B) dataset clips with real teacher fMRI ──────────────────────────
    for stem in DATASET_CLIPS:
        f = ROOT / "data" / "features" / f"{stem}.pt"
        if not f.exists():
            print(f"  !! missing {f.name}, skipping")
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        text = d["text"].float()[None].to(DEVICE)
        audio = d["audio"].float()[None].to(DEVICE)
        video = d["video"].float()[None].to(DEVICE)
        teacher = d["teacher"].float().numpy()           # (T, 20484)

        out = model(text, audio, video, torch.zeros(1, dtype=torch.long, device=DEVICE))
        pred400 = out["prediction"].float().cpu().numpy()  # (T,400)
        pred_v = parcels_to_vertices(pred400, labels)      # (T,20484)

        # vertex-wise Pearson r (validation of parcel mapping)
        rs = []
        for t in range(T):
            a, b = pred_v[t], teacher[t]
            if a.std() > 1e-8 and b.std() > 1e-8:
                rs.append(np.corrcoef(a, b)[0, 1])
        # parcel-level r (aggregate teacher to 400 blocks — the training view)
        V = teacher.shape[1]
        chunk = V // 400
        t400 = np.stack([teacher[:, i*chunk:(i+1)*chunk].mean(1)
                         for i in range(400)], axis=1)
        r_parcel = [float(np.corrcoef(pred400[t], t400[t])[0, 1]) for t in range(T)]
        print(f"  clip {stem}: vertex r = "
              + ", ".join(f"{r:.3f}" for r in rs)
              + f" | parcel r = {np.mean(r_parcel):.3f}")

        save_f16(DATA_DIR / f"clip_{stem}_pred.bin", pred_v)
        save_f16(DATA_DIR / f"clip_{stem}_true.bin", teacher)
        manifest["dataset_clips"].append({
            "id": f"clip_{stem}", "title": f"Dataset clip #{int(stem)+1}",
            "desc": "Held-out clip with measured fMRI (teacher).",
            "pred": f"data/clip_{stem}_pred.bin", "true": f"data/clip_{stem}_true.bin",
            "T": T, "kind": "vertices",
            "mean_r": float(np.mean(rs)) if rs else None,
            "parcel_r": float(np.mean(r_parcel)),
            "parcel_r_per_tr": r_parcel,
        })

    with open(DATA_DIR / "templates.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved manifest -> data/templates.json")


if __name__ == "__main__":
    main()
