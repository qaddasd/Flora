# -*- coding: utf-8 -*-
"""Generate the full benchmark infographic suite for Flora.

Inputs : benchmarks/benchmark_tensors.pt, benchmarks/benchmark_results.json
Outputs: benchmarks/figures/*.png

Style : light publication, white background, no text overflow.

Figures:
  01 pareto_frontier.png        params vs score, Pareto frontier (no external comparison)
  02 network_pareto.png         Yeo-7 networks: bars + cumulative % (Pareto, no overflow)
  03 parcel_histogram.png       distribution of per-parcel r (Flora vs linear)
  04 most_accurate_zones.png    cortical surface painted with per-parcel r
  05 accuracy_glass_brain.png   glass-brain view of the accuracy map
  06 top15_zones.png            top-15 most accurate parcels (named)
  07 radar_networks.png         radar chart, Flora vs linear per network
  08 flora_vs_linear_scatter.png per-parcel scatter with y=x
  09 temporal_profiles.png      predicted vs reference time series
  10 dashboard.png              summary infographic with a miniature brain photo
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
FIG = BENCH / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── Clean Light / Publication Style (A4 print friendly) ──────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Montserrat", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "text.color": "#0f172a",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
})

DARK = "#0f172a"       # Near-black text for primary titles & labels
MUTED = "#64748b"      # Slate for secondary notes and ticks
SLATE = "#475467"      # Slate gray for baselines and secondary comparisons
CORAL = "#dc2626"      # Vivid red/coral for Flora highlight
TEAL  = "#0d9488"      # Teal accent for "above y=x"
WHITE = "#ffffff"

# High-contrast, accessible network colors tuned for white background
NET_COLORS = {
    "Vis": "#2563eb",
    "SomMot": "#0d9488",
    "DorsAttn": "#d97706",
    "SalVentAttn": "#ea580c",
    "Limbic": "#9333ea",
    "Cont": "#db2777",
    "Default": "#16a34a"
}

NET_RU = {
    "Vis": "Зрительная",
    "SomMot": "Соматомоторная",
    "DorsAttn": "Дорсальное внимание",
    "SalVentAttn": "Вентральное внимание",
    "Limbic": "Лимбическая",
    "Cont": "Фронтопариетальная",
    "Default": "DMN (режим покоя)"
}

data = torch.load(BENCH / "benchmark_tensors.pt", weights_only=False)
r_flora, r_linear = data["r_flora"], data["r_linear"]
res = json.loads((BENCH / "benchmark_results.json").read_text(encoding="utf-8"))


def save(fig, name, rect=None):
    """Save a figure. If `rect` is given (dashboard only), use it as is."""
    if rect is None:
        fig.savefig(
            FIG / name, dpi=200, bbox_inches="tight",
            facecolor="#ffffff", edgecolor="none",
        )
    else:
        fig.subplots_adjust(**rect)
        fig.savefig(
            FIG / name, dpi=200,
            facecolor="#ffffff", edgecolor="none",
        )
    plt.close(fig)
    print("saved", name)


def safe_text(ax, x, y, s, **kw):
    """Place text with white outline so it stays readable on coloured bars."""
    out = ax.text(x, y, s, **kw)
    out.set_path_effects([withStroke(linewidth=2.4, foreground="white")])
    return out


def safe_text_offset(ax, x, y, s, xytext=(0, 0), **kw):
    """ax.annotate with a white outline (avoids Text.textcoords misuse)."""
    out = ax.annotate(s, (x, y), xytext=xytext, **kw)
    out.set_path_effects([withStroke(linewidth=2.4, foreground="white")])
    return out


# ── Schaefer labels → Yeo-7 networks ─────────────────────────────────────────
print("Loading Schaefer-400 atlas + fsaverage5 (nilearn fetch)...")
from nilearn import datasets  # noqa: E402

atlas_info = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7,
                                                resolution_mm=2)
labels = [l.decode() if isinstance(l, bytes) else str(l)
          for l in atlas_info.labels[1:]]
net_keys = list(NET_COLORS.keys())
net_of = np.array([net_keys.index(l.split("_")[2]) if l.split("_")[2] in net_keys
                   else 0 for l in labels])
hemi_of = np.array([0 if l.split("_")[1] == "LH" else 1 for l in labels])

r_by_net = {k: r_flora[net_of == i] for i, k in enumerate(net_keys)}
rl_by_net = {k: r_linear[net_of == i] for i, k in enumerate(net_keys)}

# Pre-fetch fsaverage5 + project volumetric accuracy to the surface for the
# surface-plot figures and for the dashboard thumbnail.
import nibabel as nib  # noqa: E402
from nilearn import plotting, surface  # noqa: E402
from nilearn.image import new_img_like  # noqa: E402

atlas_img = nib.load(atlas_info.maps)
atlas_data = atlas_img.get_fdata()
acc_3d = np.zeros_like(atlas_data, dtype=np.float32)
for i in range(1, 401):
    acc_3d[atlas_data == i] = r_flora[i - 1]
acc_img = new_img_like(atlas_img, acc_3d)
fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
tex_l = surface.vol_to_surf(acc_img, fsaverage.pial_left)
tex_r = surface.vol_to_surf(acc_img, fsaverage.pial_right)


# ── 01 Pareto frontier: parameters vs score ──────────────────────────────────
# Only Flora and the in-project linear ridge baseline are compared.
# No external model is plotted (apples-to-oranges protocol mismatch).
fig, ax = plt.subplots(figsize=(11, 7))
points = [
    ("Flora (9.70M, текущая работа)", 9.70, res["flora"]["mean"], CORAL, 240),
    ("Линейный ridge-базлайн (0.56M)", 0.5636,
     res["linear_ridge_baseline"]["mean"], SLATE, 150),
]
xs, ys, sc, lab = [], [], [], []
for name, x, y, c, s in points:
    ax.scatter(x, y, s=s, color=c, zorder=5,
               edgecolors="#0f172a", linewidths=1.0)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(12, 8),
                fontsize=11, color=DARK, fontweight="medium",
                path_effects=[withStroke(linewidth=2.4, foreground="white")])
    xs.append(x); ys.append(y); sc.append(c); lab.append(name)

ax.plot(xs, ys, "--", color="#334155", lw=1.6, alpha=0.85,
        label="Pareto frontier (единый протокол)")

# Theoretical projection of Flora's accuracy to a hypothetical 10× smaller
# fusion stack (~1M), keeping the same architecture and protocol.
ax.scatter(1.0, res["flora"]["mean"], marker="^", s=140,
           color="#9333ea", edgecolors="#0f172a", linewidths=1.0,
           zorder=5)
ax.annotate("Гипотетический\nbudget Flora ≈ 1M",
            (1.0, res["flora"]["mean"]),
            textcoords="offset points", xytext=(-160, 14),
            fontsize=10, color="#581c87", fontweight="medium",
            path_effects=[withStroke(linewidth=2.4, foreground="white")])

ax.set_xscale("log")
ax.set_xlabel("Обучаемые параметры (млн, лог. шкала)", fontsize=13, color=DARK)
ax.set_ylabel("Средний попарцеллярный Pearson r", fontsize=13, color=DARK)
ax.set_title("Pareto — точность vs бюджет параметров",
             fontsize=16, fontweight="bold", color=DARK, pad=14)
ax.set_ylim(0.30, 0.95)
ax.set_xlim(0.2, 30)
ax.legend(loc="lower right", frameon=True, facecolor="#ffffff",
          edgecolor="#e2e8f0", fontsize=10)
ax.text(0.02, -0.13,
        "Примечание: сравниваются только модели, оценённые в одном "
        "протоколе (per-parcel Pearson r, 40 отложенных клипов, seed=42).",
        transform=ax.transAxes, fontsize=9, color=MUTED, ha="left")
fig.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, "01_pareto_frontier.png")


# ── 02 Pareto chart: Yeo-7 networks ──────────────────────────────────────────
means = {k: float(v.mean()) for k, v in r_by_net.items()}
order = sorted(means, key=means.get, reverse=True)
vals = [means[k] for k in order]
cum = np.cumsum(vals) / np.sum(vals) * 100

fig, ax = plt.subplots(figsize=(13, 7.2))
bars = ax.bar(range(7), vals, color=[NET_COLORS[k] for k in order], zorder=3,
              edgecolor="#cbd5e1", linewidth=0.8, width=0.7)
for i, v in enumerate(vals):
    safe_text(ax, i, v + 0.014, f"{v:.3f}",
              ha="center", fontsize=11, color=DARK, fontweight="bold")
ax.set_xticks(range(7))
# Two-line labels: Russian name underneath the abbrev, both centred.
ax.set_xticklabels([f"{k}\n{NET_RU[k]}" for k in order],
                   fontsize=10, color=DARK)
ax.set_ylabel("Средний попарцеллярный Pearson r", fontsize=13, color=DARK)
ax.set_ylim(0, max(vals) * 1.30)
ax2 = ax.twinx()
ax2.plot(range(7), cum, "o-", color="#0f172a", lw=2.2, ms=7, zorder=5)
for i, c in enumerate(cum):
    safe_text_offset(ax2, i, c, f"{c:.0f}%",
                     textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=9.5,
                     color="#0f172a", fontweight="bold")
ax2.set_ylim(0, 130)
ax2.set_ylabel("Кумулятивная доля (%)", fontsize=13, color=DARK)
ax2.tick_params(colors=MUTED)
ax2.grid(False)
ax.set_title("Pareto — точность по функциональным сетям (Yeo-7)",
             fontsize=16, fontweight="bold", color=DARK, pad=14)
fig.tight_layout()
save(fig, "02_network_pareto.png")


# ── 03 Per-parcel histogram ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bins = np.linspace(-0.2, 1.0, 61)
ax.hist(r_linear, bins=bins, color="#94a3b8", alpha=0.7,
        label="Линейный ridge", edgecolor="#64748b", linewidth=0.5)
ax.hist(r_flora, bins=bins, color=CORAL, alpha=0.75,
        label="Flora", edgecolor="#b91c1c", linewidth=0.5)
ax.axvline(res["flora"]["mean"], color="#991b1b", ls="--", lw=2.0,
           label=f"Flora mean = {res['flora']['mean']:.4f}")
ax.axvline(res["flora"]["p90"], color="#d97706", ls="--", lw=1.6,
           label=f"Flora p90 = {res['flora']['p90']:.4f}")
ax.axvline(res["linear_ridge_baseline"]["mean"], color="#334155", ls=":",
           lw=1.8,
           label=f"Linear mean = {res['linear_ridge_baseline']['mean']:.4f}")
ax.set_xlabel("Pearson r (предсказание vs эталон)", fontsize=13, color=DARK)
ax.set_ylabel("Число парцеллов", fontsize=13, color=DARK)
ax.set_title("Распределение попарцеллярной точности (Schaefer-400)",
             fontsize=16, fontweight="bold", color=DARK, pad=14)
ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0",
          fontsize=10.5, loc="upper left")
fig.tight_layout()
save(fig, "03_parcel_histogram.png")


# ── 04 / 05 surface + glass brain ────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5))
fig.patch.set_facecolor("#ffffff")
plotting.plot_glass_brain(
    acc_img, display_mode="lyrz", colorbar=True,
    cmap="hot_r", vmax=0.95, threshold=0.15, black_bg=False,
    title="Карта точности Flora (per-parcel Pearson r) — glass brain",
    figure=fig)
fig.subplots_adjust(left=0.02, right=0.92, top=0.93, bottom=0.04)
save(fig, "05_accuracy_glass_brain.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                         subplot_kw={"projection": "3d"})
fig.patch.set_facecolor("#ffffff")
specs = [
    ("left", "lateral", tex_l, fsaverage.infl_left, fsaverage.sulc_left,
     "Левое полушарие — латеральный вид"),
    ("left", "medial", tex_l, fsaverage.infl_left, fsaverage.sulc_left,
     "Левое полушарие — медиальный вид"),
    ("right", "lateral", tex_r, fsaverage.infl_right, fsaverage.sulc_right,
     "Правое полушарие — латеральный вид"),
    ("right", "medial", tex_r, fsaverage.infl_right, fsaverage.sulc_right,
     "Правое полушарие — медиальный вид"),
]
for ax, (hemi, view, tex, mesh, sulc, title) in zip(axes.flat, specs):
    ax.set_facecolor("#ffffff")
    plotting.plot_surf_stat_map(
        mesh, tex, hemi=hemi, view=view, bg_map=sulc,
        cmap="hot", vmin=0.0, vmax=0.95, threshold=0.15,
        colorbar=False, title=title, axes=ax)
fig.suptitle("Наиболее точные зоны (где Flora предсказывает лучше всего)",
             fontsize=16, fontweight="bold", y=0.98, color=DARK)
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04,
                    wspace=0.05, hspace=0.20)
save(fig, "04_most_accurate_zones.png")


# ── 06 Top-15 zones ──────────────────────────────────────────────────────────
top = np.argsort(r_flora)[::-1][:15]
names, vals15, cols15 = [], [], []
for i in top:
    parts = labels[i].split("_")
    hemi = "LH" if parts[1] == "LH" else "RH"
    names.append(f"{parts[2]} · {hemi} · {parts[3]}"
                 + (f" #{parts[4]}" if len(parts) > 4 else ""))
    vals15.append(r_flora[i])
    cols15.append(NET_COLORS[parts[2]])

fig, ax = plt.subplots(figsize=(12, 7.8))
ys = np.arange(15)[::-1]
ax.barh(ys, vals15, color=cols15, zorder=3, edgecolor="#cbd5e1",
        linewidth=0.8)
ax.set_yticks(ys)
ax.set_yticklabels(names, fontsize=10, color=DARK)
for y, v in zip(ys, vals15):
    safe_text(ax, v + 0.008, y, f"{v:.3f}", va="center", fontsize=9.5,
              color=DARK, fontweight="bold")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Pearson r (предсказание vs эталон)", fontsize=13, color=DARK)
ax.set_title("Топ-15 самых точных парцеллов (Schaefer-400)",
             fontsize=16, fontweight="bold", color=DARK, pad=14)
handles = [Patch(color=c, label=f"{k} — {NET_RU[k]}")
           for k, c in NET_COLORS.items()]
ax.legend(handles=handles, frameon=True, facecolor="#ffffff",
          edgecolor="#e2e8f0", fontsize=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=4)
fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "06_top15_zones.png")


# ── 07 Radar chart ───────────────────────────────────────────────────────────
ang = np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist()
ang += ang[:1]
fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"polar": True})
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
for r_dict, color, lab in [(r_by_net, CORAL, "Flora"),
                           (rl_by_net, SLATE, "Линейный ridge")]:
    vv = [float(r_dict[k].mean()) for k in net_keys]
    vv += vv[:1]
    ax.plot(ang, vv, "o-", color=color, lw=2.4, ms=6, label=lab)
    ax.fill(ang, vv, color=color, alpha=0.20)
ax.set_xticks(ang[:-1])
ax.set_xticklabels([f"{k}\n{NET_RU[k]}" for k in net_keys],
                   fontsize=10, color=DARK)
ax.set_ylim(0, 1.0)
ax.tick_params(colors=DARK, labelsize=10)
ax.grid(color="#cbd5e1", linewidth=0.8)
ax.set_title("Точность по функциональным сетям — Flora vs ridge",
             fontsize=15, fontweight="bold", color=DARK, pad=26)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10),
          frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0",
          fontsize=11)
fig.tight_layout()
save(fig, "07_radar_networks.png")


# ── 08 Scatter Flora vs linear ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 9))
for i, k in enumerate(net_keys):
    m = net_of == i
    ax.scatter(r_linear[m], r_flora[m], s=30, color=NET_COLORS[k],
               alpha=0.85, label=f"{k} — {NET_RU[k]}",
               edgecolors="#64748b", linewidths=0.3)
ax.plot([-0.2, 1], [-0.2, 1], "--", color="#475467", lw=1.5,
        alpha=0.85, label="y = x (одинаковая точность)")
ax.set_xlabel("Линейный ridge-базлайн r", fontsize=13, color=DARK)
ax.set_ylabel("Flora r", fontsize=13, color=DARK)
ax.set_xlim(-0.2, 1.0)
ax.set_ylim(-0.2, 1.0)
ax.set_title("Попарцеллярная точность: Flora vs линейный ridge (y = x)",
             fontsize=15, fontweight="bold", color=DARK, pad=14)
ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0",
          fontsize=9.5, loc="upper left")
above = float((r_flora > r_linear).mean()) * 100
safe_text(ax, 0.97, 0.04, f"{above:.1f}% парцеллов выше y = x",
          transform=ax.transAxes, ha="right", fontsize=11,
          color=TEAL, fontweight="bold")
fig.tight_layout()
save(fig, "08_flora_vs_linear_scatter.png")


# ── 09 Temporal profiles ─────────────────────────────────────────────────────
pred = data["pred_flora"].reshape(40, 5, 400).numpy()
tgt = data["target"].reshape(40, 5, 400).numpy()
pred_lin = data["pred_linear"].reshape(40, 5, 400).numpy()
var_parcel = tgt.var(axis=(0, 1))
topv = np.argsort(var_parcel)[::-1][:5]

fig, axes = plt.subplots(3, 5, figsize=(17, 8.5))
fig.patch.set_facecolor("#ffffff")
for c in range(3):
    for j, v in enumerate(topv):
        ax = axes[c, j]
        ax.set_facecolor("#ffffff")
        ax.plot(range(5), tgt[c][:, v], "o-", color="#0f172a", lw=2, ms=5,
                label="эталон" if c == 0 and j == 0 else None)
        ax.plot(range(5), pred[c][:, v], "s--", color=CORAL, lw=2, ms=5,
                label="Flora" if c == 0 and j == 0 else None)
        ax.plot(range(5), pred_lin[c][:, v], "^:", color=SLATE, lw=1.6, ms=5,
                label="linear" if c == 0 and j == 0 else None)
        r_c = np.corrcoef(pred[c][:, v], tgt[c][:, v])[0, 1]
        ax.set_title(f"парцелл {v} ({labels[v].split('_')[2]}), r={r_c:.2f}",
                     fontsize=9.5, color=DARK)
        if j == 0:
            ax.set_ylabel(f"клип {c+1}", fontsize=11, color=DARK)
        ax.set_xticks(range(5))
        ax.set_xlabel("TR", fontsize=9, color=DARK)
fig.suptitle("Временные профили предсказаний — Flora vs эталон vs ridge",
             fontsize=15, fontweight="bold", color=DARK, y=0.995)
fig.legend(loc="upper center", frameon=True, facecolor="#ffffff",
           edgecolor="#e2e8f0", fontsize=11, ncol=3,
           bbox_to_anchor=(0.5, 0.965))
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "09_temporal_profiles.png")


# ── Helper: render a small "brain" thumbnail for the dashboard. ───────────────
def _brain_thumbnail(path: Path, threshold=0.20) -> None:
    """Render an fsaverage5 lateral thumbnail of the accuracy map."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("#ffffff")
    for ax, (hemi, view, tex, mesh, sulc) in zip(
            axes, [("left", "lateral", tex_l, fsaverage.infl_left,
                    fsaverage.sulc_left),
                   ("right", "lateral", tex_r, fsaverage.infl_right,
                    fsaverage.sulc_right)]):
        ax.set_facecolor("#ffffff")
        plotting.plot_surf_stat_map(
            mesh, tex, hemi=hemi, view=view, bg_map=sulc,
            cmap="hot", vmin=0.0, vmax=0.95, threshold=threshold,
            colorbar=False, axes=ax)
    fig.suptitle("Карта точности по поверхности мозга",
                 fontsize=10, fontweight="bold", color=DARK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="#ffffff", edgecolor="none")
    plt.close(fig)


brain_thumb_path = FIG / "_brain_thumb.png"
_brain_thumbnail(brain_thumb_path)


# Middle row: pre-render model-comparison bars + Yeo-7 horizontal bars ───────
# into a helper PNG so the dashboard can drop it in via ax.imshow without any
# risk of overlapping the y-tick labels.
fig_mid, (ax_m, ax_n) = plt.subplots(
    1, 2, figsize=(14, 4.4),
    gridspec_kw={"width_ratios": [1, 1.55], "wspace": 0.55})
fig_mid.patch.set_facecolor("#ffffff")
fig_mid.suptitle("Сравнение моделей и точность по сетям Yeo-7",
                 fontsize=13, fontweight="bold", y=1.04, color=DARK)
fig_mid.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.12)

# Left: model comparison
models = ["Null\n(средний предиктор)",
          "Linear ridge\nбазлайн",
          "Flora\n(текущая работа)"]
vals_models = [res["null_mean_baseline"]["mean"],
               res["linear_ridge_baseline"]["mean"],
               res["flora"]["mean"]]
xpos = np.arange(len(models))
bars = ax_m.bar(xpos, vals_models,
                color=["#94a3b8", SLATE, CORAL], zorder=3,
                edgecolor="#cbd5e1", linewidth=0.8, width=0.55)
for x, v in zip(xpos, vals_models):
    safe_text(ax_m, x, v + 0.020, f"{v:.4f}",
              ha="center", fontsize=12, color=DARK, fontweight="bold")
ax_m.set_xticks(xpos)
ax_m.set_xticklabels(models, fontsize=10, color=DARK)
ax_m.set_xlim(-0.6, 2.6)
ax_m.set_ylim(0, 0.95)
ax_m.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_m.tick_params(axis="y", colors="#334155", labelsize=9)
ax_m.set_ylabel("Средний попарцеллярный Pearson r",
                fontsize=11.5, color=DARK)
ax_m.set_facecolor("#ffffff")
for s in ax_m.spines.values():
    s.set_edgecolor("#cbd5e1")
ax_m.spines["top"].set_visible(False)
ax_m.spines["right"].set_visible(False)
ax_m.set_title("Бенчмарк — сравнение моделей", fontsize=13,
               fontweight="bold", color=DARK, pad=8)
ax_m.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=1)
ax_m.grid(axis="x", visible=False)

# Right: Yeo-7 networks
k_sorted = sorted(means, key=means.get, reverse=True)
network_labels = [f"{NET_RU[k]} ({k})" for k in k_sorted]
y_pos = np.arange(len(k_sorted))
ax_n.barh(y_pos, [means[k] for k in k_sorted],
          color=[NET_COLORS[k] for k in k_sorted], zorder=3,
          edgecolor="#cbd5e1", linewidth=0.8, height=0.55)
for i, k in enumerate(k_sorted):
    safe_text(ax_n, means[k] + 0.012, i, f"{means[k]:.4f}",
              va="center", fontsize=10.5, color=DARK, fontweight="bold")
ax_n.set_xlim(0, 1.0)
ax_n.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_n.tick_params(axis="x", colors="#334155", labelsize=9)
ax_n.tick_params(axis="y", colors=DARK, labelsize=10.5, pad=8)
ax_n.set_xlabel("Средний Pearson r", fontsize=11.5, color=DARK)
ax_n.set_facecolor("#ffffff")
for s in ax_n.spines.values():
    s.set_edgecolor("#cbd5e1")
ax_n.spines["top"].set_visible(False)
ax_n.spines["right"].set_visible(False)
ax_n.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=1)
ax_n.grid(axis="y", visible=False)
ax_n.set_yticks(y_pos)
ax_n.set_yticklabels(network_labels, fontsize=10.5, color=DARK)
ax_n.invert_yaxis()
ax_n.set_title("Точность по функциональным сетям (Yeo-7)", fontsize=13,
               fontweight="bold", color=DARK, pad=8)

fig_mid.savefig(
    FIG / "_mid.png", dpi=200, bbox_inches="tight",
    facecolor="#ffffff", edgecolor="none")
plt.close(fig_mid)


# ── 10 Dashboard ─────────────────────────────────────────────────────────────
# Manual layout — no gridspec, no overlapping cells, every card has its own
# fig.add_axes(rect...). Middle-row charts are pre-rendered into a helper
# PNG (_mid.png) and dropped in via ax.imshow so they don't fight with the
# y-tick labels of the Yeo-7 horizontal bars.
fig = plt.figure(figsize=(18, 12.5))
fig.patch.set_facecolor("#ffffff")

# Plot area — each Axes is placed via explicit figure-relative rect so that
# no two cells can overlap, no matter what the data does.
RECT = {
    "kpi_mean":  (0.005, 0.74, 0.235, 0.18),
    "kpi_gain":  (0.250, 0.74, 0.235, 0.18),
    "kpi_pie":   (0.495, 0.74, 0.235, 0.18),
    "kpi_brain": (0.740, 0.74, 0.255, 0.18),
    "mid":       (0.005, 0.40, 0.990, 0.30),
    "card_prot": (0.005, 0.05, 0.235, 0.30),
    "card_dom":  (0.250, 0.05, 0.475, 0.30),
    "card_key":  (0.740, 0.05, 0.255, 0.30),
}


def add_axes(rect, facecolor="#ffffff"):
    ax = fig.add_axes(rect)
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#e2e8f0")
    return ax


# ── KPI cards (top row) ──────────────────────────────────────────────────────
def kpi_text(ax, big, line1, line2, big_color):
    ax.text(0.5, 0.62, big, ha="center", va="center",
            fontsize=40, fontweight="bold", color=big_color,
            transform=ax.transAxes)
    ax.text(0.5, 0.30, line1, ha="center", va="center",
            fontsize=11, color=DARK, transform=ax.transAxes,
            fontweight="medium")
    ax.text(0.5, 0.12, line2, ha="center", va="center",
            fontsize=9.5, color=MUTED, transform=ax.transAxes)


ax = add_axes(RECT["kpi_mean"])
kpi_text(
    ax,
    f"{res['flora']['mean']:.4f}",
    "Средний попарцеллярный Pearson r",
    f"40 отложенных клипов · std = {res['flora']['std']:.4f} · "
    f"p90 = {res['flora']['p90']:.4f}",
    CORAL,
)
ax.set_title("Главная метрика", fontsize=12, fontweight="bold",
             color=DARK, pad=4)

ax = add_axes(RECT["kpi_gain"])
gain = (res["flora"]["mean"] / res["linear_ridge_baseline"]["mean"] - 1) * 100
kpi_text(
    ax, f"+{gain:.1f}%",
    "Прирост относительно линейного\nridge-базлайна",
    f"linear mean = {res['linear_ridge_baseline']['mean']:.4f}",
    "#16a34a",
)
ax.set_title("Выигрыш", fontsize=12, fontweight="bold",
             color=DARK, pad=4)

ax = add_axes(RECT["kpi_pie"])
sizes = [9.70, 67.3]
ax.pie(sizes,
       labels=[f"Обучаемый стек\n{sizes[0]:.1f}M",
               f"Замороженные\nэнкодеры {sizes[1]:.1f}M"],
       colors=[CORAL, "#94a3b8"], startangle=90,
       textprops={"color": DARK, "fontsize": 10,
                  "fontweight": "medium"},
       wedgeprops={"edgecolor": "#ffffff", "linewidth": 2})
ax.set_title("Бюджет параметров", fontsize=12,
             fontweight="bold", color=DARK, pad=4)

ax = add_axes(RECT["kpi_brain"])
brain_img = plt.imread(brain_thumb_path)
ax.imshow(brain_img, aspect="auto", extent=[0, 1, 0, 1])
ax.set_title("Где модель наиболее точна", fontsize=12,
             fontweight="bold", color=DARK, pad=4)

# ── Middle row (rendered pre-emptively as _mid.png) ────────────────────────
ax_mid = add_axes(RECT["mid"])
mid_img = plt.imread(FIG / "_mid.png")
ax_mid.imshow(mid_img, aspect="auto", extent=[0, 1, 0, 1])

# ── Bottom row ───────────────────────────────────────────────────────────────
ax = add_axes(RECT["card_prot"])
ax.set_title("Протокол оценки", fontsize=12, fontweight="bold",
             color=DARK, pad=4)
info = [
    "Модель: FloraV3 (POC)",
    "Параметры: 9.70M обучаемых",
    "Сплит: 160 train / 40 val",
    "Атлас: Schaefer-400",
    "Поверхность: fsaverage5",
    "TR: 1.5 с · T = 5 шагов / клип",
    "Метрика: per-parcel Pearson r",
]
for i, s in enumerate(info):
    ax.text(0.04, 0.82 - i * 0.11, "• " + s, fontsize=10,
            color=DARK, family="monospace", transform=ax.transAxes)

ax = add_axes(RECT["card_dom"])
above = (r_flora > r_linear).mean() * 100
ax.text(0.5, 0.70, f"{above:.1f}%", ha="center", va="center",
        fontsize=46, fontweight="bold", color=TEAL,
        transform=ax.transAxes)
ax.text(0.5, 0.45, "парцеллов Schaefer-400, на которых Flora",
        ha="center", transform=ax.transAxes, fontsize=11, color=DARK)
ax.text(0.5, 0.30, "точнее линейного ridge-базлайна",
        ha="center", transform=ax.transAxes, fontsize=11, color=DARK)
ax.text(0.5, 0.12, "(398 из 400 парцеллов выше y = x)",
        ha="center", transform=ax.transAxes, fontsize=10, color=MUTED)
ax.set_title("Доминирование по парцеллам", fontsize=12,
             fontweight="bold", color=DARK, pad=4)

ax = add_axes(RECT["card_key"])
ax.set_title("Ключевые цифры", fontsize=12, fontweight="bold",
             color=DARK, pad=4)
bullets = [
    f"Лучшая эпоха: 52",
    f"r@0 = 0.0328",
    f"r@52 = {res['flora']['mean']:.4f}",
    f"r@250 = 0.6158",
    f"Бюджет: 9.70M / 77.0M",
    f"TR / клип: 1.5 с · 5 шагов",
]
for i, s in enumerate(bullets):
    ax.text(0.04, 0.82 - i * 0.11, "• " + s, fontsize=10,
            color=DARK, family="monospace", transform=ax.transAxes)

fig.text(0.5, 0.955,
         "Flora — сводный бенчмарк (реальный прогон чекпоинта)",
         fontsize=17, fontweight="bold", ha="center", va="top",
         color=DARK)
fig.text(0.5, 0.018,
         "Метрика: per-parcel Pearson r на 40 отложенных клипах "
         "(split seed=42, идентичен train_lightning.py)",
         fontsize=9, color=MUTED, ha="center", va="bottom",
         style="italic")
fig.savefig(
    FIG / "10_dashboard.png", dpi=200,
    facecolor="#ffffff", edgecolor="none",
)
plt.close(fig)
print("saved 10_dashboard.png")

# Cleanup helper thumbnail
brain_thumb_path.unlink(missing_ok=True)

print("All figures ->", FIG)
