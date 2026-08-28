# -*- coding: utf-8 -*-
"""Flora benchmark pipeline.

1. Reproduces the exact 160/40 split of train_lightning.py (seed 42).
2. Runs the released checkpoint (pearson_r=0.7278.ckpt) on the 40 val clips.
3. Computes per-parcel Pearson r (prediction vs reference), per-network stats.
4. Trains a ridge linear baseline on the same features (no-MoE linear probe).
5. Saves benchmark_results.json + per_parcel_r.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flora.v3_model import FloraV3  # noqa: E402

FEAT_DIR = ROOT / "data" / "features"
CKPT = ROOT / "checkpoints" / "best-epoch=052-val" / "pearson_r=0.7278.ckpt"
OUT_DIR = ROOT / "benchmarks"
OUT_DIR.mkdir(exist_ok=True)

N_PARCELS = 400
SEED = 42


def parcellate(t: torch.Tensor, n_parcels: int = N_PARCELS) -> torch.Tensor:
    """(T, 20484) -> (T, n_parcels) chunk averaging — identical to train_lightning.py."""
    T, V = t.shape
    if V == n_parcels:
        return t
    chunk = V // n_parcels
    return torch.stack([t[:, i * chunk:(i + 1) * chunk].mean(dim=1)
                        for i in range(n_parcels)], dim=1)


def load_split():
    files = sorted(FEAT_DIR.glob("*.pt"))
    n_val = max(1, int(len(files) * 0.20))
    gen = torch.Generator().manual_seed(SEED)
    train_f, val_f = torch.utils.data.random_split(
        files, [len(files) - n_val, n_val], generator=gen)
    return list(train_f), list(val_f)


def load_clip(p: Path):
    d = torch.load(p, map_location="cpu", weights_only=True)
    ref = parcellate(d["reference"].float())
    return (d["text"].float(), d["audio"].float(), d["video"].float(),
            ref)


@torch.no_grad()
def run_model(model, files, device):
    preds, targets = [], []
    for f in files:
        t, a, v, ref = load_clip(f)
        out = model(t.unsqueeze(0).to(device), a.unsqueeze(0).to(device),
                    v.unsqueeze(0).to(device))
        pred = out["prediction"]
        pred = pred.squeeze(0)
        if pred.shape[0] == N_PARCELS and pred.shape[1] != N_PARCELS:
            pred = pred.transpose(0, 1)   # (n_v, T) -> (T, n_v)
        pred = pred.cpu()
        T = min(pred.shape[0], ref.shape[0])
        preds.append(pred[:T])
        targets.append(ref[:T])
    return torch.cat(preds, 0), torch.cat(targets, 0)


def per_parcel_r(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Pearson r per parcel across the flattened (N*T) axis — same as PearsonRMetric."""
    p = pred.transpose(0, 1)   # (V, N*T)
    t = target.transpose(0, 1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1) + 1e-8
    return (num / den).numpy()


def ridge_baseline(train_files, val_files, lam=1e-2):
    """Linear encoder: concat(text,audio,video) per TR -> ridge -> parcels."""
    Xtr, Ytr = [], []
    for f in train_files:
        t, a, v, ref = load_clip(f)
        x = torch.cat([t, a, v], dim=1)  # (T, 1408)
        Xtr.append(x)
        Ytr.append(ref)
    Xtr = torch.cat(Xtr, 0).double()
    Ytr = torch.cat(Ytr, 0).double()
    mu_x, mu_y = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc, Yc = Xtr - mu_x, Ytr - mu_y
    XtX = Xc.T @ Xc
    W = torch.linalg.solve(XtX + lam * torch.eye(XtX.shape[0], dtype=torch.double),
                           Xc.T @ Yc)

    Xv, Yv = [], []
    for f in val_files:
        t, a, v, ref = load_clip(f)
        Xv.append(torch.cat([t, a, v], dim=1))
        Yv.append(ref)
    Xv = torch.cat(Xv, 0).double() - mu_x
    Yv = torch.cat(Yv, 0).double()
    pred = (Xv @ W).float()
    return pred, Yv.float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_files, val_files = load_split()
    print(f"Split reproduced: {len(train_files)} train / {len(val_files)} val "
          f"(first val: {val_files[0].name})")

    # --- Flora checkpoint ---
    model = FloraV3.load_from_checkpoint(str(CKPT), map_location=device)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FloraV3 loaded: {n_params/1e6:.2f}M params")

    pred_f, tgt = run_model(model, val_files, device)
    r_flora = per_parcel_r(pred_f, tgt)

    # --- Ridge linear baseline ---
    pred_lin, tgt_lin = ridge_baseline(train_files, val_files)
    r_linear = per_parcel_r(pred_lin, tgt_lin)

    # --- Null baseline: train-set mean per parcel ---
    mean_pred = torch.stack([load_clip(f)[3] for f in train_files]).mean(dim=(0, 1))
    r_null = per_parcel_r(mean_pred.unsqueeze(0).expand(tgt.shape[0], -1), tgt)

    def stats(r):
        return {"mean": float(np.mean(r)), "std": float(np.std(r)),
                "p90": float(np.percentile(r, 90)),
                "p50": float(np.percentile(r, 50)),
                "max": float(np.max(r)), "min": float(np.min(r))}

    results = {
        "protocol": "per-parcel Pearson r, prediction vs reference, "
                    "40 held-out clips, split seed=42 (train_lightning.py)",
        "n_train_clips": len(train_files), "n_val_clips": len(val_files),
        "n_parcels": N_PARCELS, "T_per_clip": 5,
        "flora_checkpoint_params": int(n_params),
        "flora": stats(r_flora),
        "linear_ridge_baseline": stats(r_linear),
        "null_mean_baseline": stats(r_null),
        "val_files": [f.name for f in val_files],
    }
    with open(OUT_DIR / "benchmark_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    import csv
    with open(OUT_DIR / "per_parcel_r.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["parcel", "r_flora", "r_linear_ridge"])
        for i in range(N_PARCELS):
            w.writerow([i, f"{r_flora[i]:.6f}", f"{r_linear[i]:.6f}"])

    # Persist tensors for the plotting stage
    torch.save({"r_flora": r_flora, "r_linear": r_linear, "r_null": r_null,
                "pred_flora": pred_f, "target": tgt,
                "pred_linear": pred_lin},
               OUT_DIR / "benchmark_tensors.pt")

    print("\n===== RESULTS (per-parcel Pearson r) =====")
    print(f"Flora  : mean={results['flora']['mean']:.4f} "
          f"std={results['flora']['std']:.4f} p90={results['flora']['p90']:.4f}")
    print(f"Linear : mean={results['linear_ridge_baseline']['mean']:.4f} "
          f"std={results['linear_ridge_baseline']['std']:.4f}")
    print(f"Null   : mean={results['null_mean_baseline']['mean']:.4f}")
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
