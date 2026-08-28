# -*- coding: utf-8 -*-
"""Publication-quality figures for main.tex — replaces all TikZ diagrams.

Style follows the TRIBE v2 papers (see reference screenshots): white
background, flat rounded panels, colour-coded modalities
(video = blue, audio = green, text = red), dark slate fusion blocks and
FreeSurfer-style cortical surface renders (nilearn, fsaverage5, hot cmap).

Outputs (assets/):
  fig_overview.png     pipeline overview (encoders -> MoE-Transformer -> brain)
  fig_hierarchy.png    cortex -> surface -> Schaefer-400 -> Yeo-7
  fig_hrf.png          canonical double-gamma HRF
  fig_projection.png   voxels -> fsaverage5 -> Schaefer-400 -> Yeo-7 renders
  fig_motion.png       TemporalMotionModule (frame differencing)
  fig_moe_block.png    MoE-transformer block with HRF attention bias
  fig_moe_router.png   top-2 expert routing
  fig_gating.png       modality gating + gate dynamics
  fig_film.png         FiLM subject conditioning

Run:  venv/Scripts/python.exe scripts/generate_paper_figures.py [name ...]
"""
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle, FancyArrowPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Montserrat', 'Inter', 'Segoe UI', 'DejaVu Sans', 'Arial'],
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
    'text.color': '#0f172a',
    'axes.edgecolor': '#cbd5e1',
    'axes.labelcolor': '#0f172a',
    'xtick.color': '#334155',
    'ytick.color': '#334155',
})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'assets'
OUT.mkdir(parents=True, exist_ok=True)

# ── TRIBE-style palette ──────────────────────────────────────────────────────
WHITE = '#ffffff'
PANEL = '#f8fafc'
DARK = '#0f172a'
SLATE = '#475467'
MUTED = '#64748b'
BORDER = '#cbd5e1'
GRID = '#e2e8f0'
CORAL = '#dc2626'
TEAL = '#0d9488'

VIDEO_C = '#4f83f7'   # blue   (cf. V-JEPA2 in TRIBE v2)
AUDIO_C = '#7ac143'   # green  (cf. W2vec-Bert)
TEXT_C = '#e2574c'    # red    (cf. Llama 3.2)
FUSION_C = '#3d4a5c'  # dark slate (Transformer block)
SUBJ_C = '#5a6b7f'    # lighter slate (Subject block)

YEO7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
YEO7_COLORS = ['#2563eb', '#0d9488', '#d97706', '#ea580c',
               '#9333ea', '#db2777', '#16a34a']
YEO7_RU = ['Зрительная', 'Соматомоторная', 'Дорсальное внимание',
           'Вентральное внимание', 'Лимбическая', 'Фронтопариетальная',
           'DMN (режим покоя)']


# ════════════════════════════════════════════════════════════════════════════
# generic drawing helpers
# ════════════════════════════════════════════════════════════════════════════

def rbox(ax, x, y, w, h, fill=PANEL, edge=BORDER, lw=1.2, r=0.09, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color='#334155', lw=1.8, style='-|>', ms=14,
          connectionstyle=None, z=4, ls='-'):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, zorder=z, linestyle=ls,
                        connectionstyle=connectionstyle,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    return a


def panel_tag(ax, tag, x=0.015, y=0.955):
    """TRIBE-style bold panel letter (A, B, C ...)."""
    text_fn = ax.text2D if hasattr(ax, 'text2D') else ax.text
    text_fn(x, y, tag, transform=ax.transAxes, fontsize=17,
            fontweight='bold', color=DARK, va='top', ha='left', zorder=10)


def hot_colorbar(fig, rect, lo='Low', hi='High', label=None):
    """Horizontal 'Low —— High' hot gradient bar like in TRIBE figures."""
    cax = fig.add_axes(rect)
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(grad, aspect='auto', cmap='hot')
    cax.set_xticks([0, 255])
    cax.set_xticklabels([lo, hi], fontsize=10, color=DARK)
    cax.xaxis.set_ticks_position('top')
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    cax.tick_params(length=0, pad=2)
    if label:
        cax.set_xlabel(label, fontsize=9.5, color=MUTED, labelpad=3)
    return cax


def icon_video(ax, cx, cy, s=1.0, color=DARK):
    w, h = 0.86 * s, 0.58 * s
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.045', linewidth=1.6,
                                edgecolor=color, facecolor=WHITE, zorder=5))
    t = 0.19 * s
    ax.add_patch(Polygon([(cx - t * 0.55, cy - t), (cx - t * 0.55, cy + t),
                          (cx + t * 0.85, cy)], closed=True,
                         facecolor=color, edgecolor='none', zorder=6))


def icon_waveform(ax, cx, cy, s=1.0, color=DARK):
    heights = [0.16, 0.30, 0.46, 0.34, 0.52, 0.28, 0.42, 0.20, 0.34]
    bw, gap = 0.055 * s, 0.088 * s
    x0 = cx - gap * (len(heights) - 1) / 2
    for i, hh in enumerate(heights):
        h = hh * s
        ax.add_patch(Rectangle((x0 + i * gap - bw / 2, cy - h / 2), bw, h,
                               facecolor=color, edgecolor='none', zorder=5))


def icon_text(ax, cx, cy, s=1.0, color=DARK):
    ax.text(cx, cy + 0.06 * s, 'Abc', ha='center', va='center',
            fontsize=15 * s, color=color, fontweight='bold', zorder=5)
    for dy, wfrac in ((-0.22, 0.42), (-0.34, 0.28)):
        ax.plot([cx - wfrac * s, cx + wfrac * s],
                [cy + dy * s, cy + dy * s],
                color=MUTED, lw=1.4, solid_capstyle='round', zorder=5)


def icon_gear(ax, cx, cy, r=0.42, color=WHITE, hub=FUSION_C):
    n_teeth = 8
    for k in range(n_teeth):
        ang = 2 * np.pi * k / n_teeth
        tx, ty = np.cos(ang), np.sin(ang)
        w, L = r * 0.34, r * 0.42
        verts = []
        for al, rr in ((-w, r * 0.78), (w, r * 0.78),
                       (w * 0.8, r * 0.78 + L), (-w * 0.8, r * 0.78 + L)):
            px_, py_ = -ty, tx
            verts.append((cx + tx * rr + px_ * al, cy + ty * rr + py_ * al))
        ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                             edgecolor='none', zorder=5))
    ax.add_patch(Circle((cx, cy), r * 0.82, facecolor=color,
                        edgecolor='none', zorder=5))
    ax.add_patch(Circle((cx, cy), r * 0.34, facecolor=hub,
                        edgecolor='none', zorder=6))


def icon_person(ax, cx, cy, s=1.0, color=WHITE):
    ax.add_patch(Circle((cx, cy + 0.22 * s), 0.16 * s, facecolor=color,
                        edgecolor='none', zorder=5))
    th = np.linspace(np.pi, 0, 40)
    sx = cx + 0.30 * s * np.cos(th)
    sy = cy + 0.34 * s * np.sin(th) - 0.32 * s
    ax.add_patch(Polygon(np.column_stack([sx, sy]), closed=True,
                         facecolor=color, edgecolor='none', zorder=5))


# ════════════════════════════════════════════════════════════════════════════
# nilearn brain data (fsaverage5 + Schaefer-400), lazy and cached
# ════════════════════════════════════════════════════════════════════════════
_BRAIN = {}


def brain_data():
    """fsaverage5 meshes + synthetic TRIBE-like activation / parcellation."""
    if _BRAIN:
        return _BRAIN
    import nibabel as nib
    from nilearn import datasets, surface
    from nilearn.image import new_img_like

    print('  [nilearn] fetching fsaverage5 + Schaefer-400 (cached) ...')
    fs = datasets.fetch_surf_fsaverage('fsaverage5')
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7,
                                               resolution_mm=2)
    atlas_img = nib.load(atlas.maps)
    labels = [l.decode() if isinstance(l, bytes) else str(l)
              for l in atlas.labels[1:]]
    net_of = np.array([YEO7.index(l.split('_')[2]) if l.split('_')[2] in YEO7
                       else 0 for l in labels])

    # -- synthetic "predicted brain activity": audio-visual narrative ------
    # focal TRIBE-like pattern: strong visual, moderate auditory/attention
    rng = np.random.default_rng(11)
    base = {0: 0.92, 1: 0.48, 2: 0.38, 3: 0.20, 4: 0.15, 5: 0.28, 6: 0.22}
    act = np.array([base[n] for n in net_of])
    act += rng.normal(0, 0.04, act.size)
    act = np.clip(act, 0.02, 1.0)

    adata = atlas_img.get_fdata()
    act_3d = np.zeros_like(adata, dtype=np.float32)
    net_3d = np.zeros_like(adata, dtype=np.float32)
    for i in range(1, 401):
        m = adata == i
        act_3d[m] = act[i - 1]
        net_3d[m] = net_of[i - 1] + 1
    act_img = new_img_like(atlas_img, act_3d)
    net_img = new_img_like(atlas_img, net_3d)

    tex_l = surface.vol_to_surf(act_img, fs.pial_left, radius=6.0)
    tex_r = surface.vol_to_surf(act_img, fs.pial_right, radius=6.0)
    roi_l = surface.vol_to_surf(atlas_img, fs.pial_left,
                                interpolation='nearest_most_frequent')
    roi_r = surface.vol_to_surf(atlas_img, fs.pial_right,
                                interpolation='nearest_most_frequent')
    net_l = surface.vol_to_surf(net_img, fs.pial_left,
                                interpolation='nearest_most_frequent')
    net_r = surface.vol_to_surf(net_img, fs.pial_right,
                                interpolation='nearest_most_frequent')

    _BRAIN.update(fs=fs, act_img=act_img, act=(tex_l, tex_r),
                  roi=(roi_l, roi_r), net=(net_l, net_r))
    return _BRAIN


def surf_ax(fig, rect, mesh, stat, hemi, view, sulc, cmap='hot', vmin=0.0,
            vmax=1.0, threshold=0.35, title=None, title_size=10.5):
    from nilearn import plotting
    ax = fig.add_axes(rect, projection='3d')
    ax.set_facecolor(WHITE)
    plotting.plot_surf_stat_map(mesh, stat, hemi=hemi, view=view,
                                bg_map=sulc, cmap=cmap, vmin=vmin, vmax=vmax,
                                threshold=threshold, colorbar=False,
                                axes=ax)
    if title:
        ax.set_title(title, fontsize=title_size, color=DARK, pad=2)
    return ax


def roi_ax(fig, rect, mesh, roi, hemi, view, sulc, cmap, title=None,
           title_size=10.5):
    from nilearn import plotting
    ax = fig.add_axes(rect, projection='3d')
    ax.set_facecolor(WHITE)
    plotting.plot_surf_roi(mesh, roi, hemi=hemi, view=view, bg_map=sulc,
                           cmap=cmap, colorbar=False, axes=ax, threshold=0.5)
    if title:
        ax.set_title(title, fontsize=title_size, color=DARK, pad=2)
    return ax


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, facecolor=WHITE, edgecolor='none')
    plt.close(fig)
    print('saved', path)


# ════════════════════════════════════════════════════════════════════════════
# 1. fig_overview.png — full pipeline (TRIBE v2 Fig.1B style)
# ════════════════════════════════════════════════════════════════════════════

def fig_overview():
    B = brain_data()
    fs = B['fs']

    fig = plt.figure(figsize=(16.2, 6.4))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0.005, 0.02, 0.66, 0.96])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis('off')
    panel_tag(ax, 'A')

    # ── left: modality input chips ───────────────────────────────────────
    IN_X, IN_W, IN_H = 0.35, 2.55, 1.75
    inputs = [('Видео', icon_video, VIDEO_C, 6.85),
              ('Аудио', icon_waveform, AUDIO_C, 4.45),
              ('Текст', icon_text, TEXT_C, 2.05)]
    in_cy = {}
    for label, icon, col, y in inputs:
        rbox(ax, IN_X, y, IN_W, IN_H, fill=WHITE, edge=col, lw=1.6)
        icon(ax, IN_X + IN_W / 2, y + IN_H * 0.62, s=1.0, color=col)
        ax.text(IN_X + IN_W / 2, y + IN_H * 0.18, label, ha='center',
                va='center', fontsize=12, color=DARK, fontweight='bold', zorder=6)
        in_cy[label] = y + IN_H / 2

    # ── frozen encoders (colour coded) ───────────────────────────────────
    ENC_X, ENC_W, ENC_H = 3.85, 3.15, 1.55
    encs = [('Видео', 'MobileViT-S', 'заморожен · 5.6M', VIDEO_C),
            ('Аудио', 'Whisper-Tiny', 'заморожен · 39M', AUDIO_C),
            ('Текст', 'MiniLM-L6-v2', 'заморожен · 22.7M', TEXT_C)]
    for mod, name, sub, col in encs:
        cy = in_cy[mod]
        rbox(ax, ENC_X, cy - ENC_H / 2, ENC_W, ENC_H, fill=col, edge='none')
        ax.text(ENC_X + ENC_W / 2, cy + 0.22, name, ha='center', va='center',
                fontsize=12.5, color=WHITE, fontweight='bold', zorder=6)
        ax.text(ENC_X + ENC_W / 2, cy - 0.38, sub, ha='center', va='center',
                fontsize=9, color='#f1f5f9', zorder=6)
        arrow(ax, (IN_X + IN_W + 0.08, cy), (ENC_X - 0.10, cy))

    # ── fusion transformer (dark) ────────────────────────────────────────
    TR_X, TR_W, TR_Y, TR_H = 8.05, 3.35, 1.55, 7.05
    rbox(ax, TR_X, TR_Y, TR_W, TR_H, fill=FUSION_C, edge='none', r=0.14)
    icon_gear(ax, TR_X + TR_W / 2, TR_Y + TR_H - 1.55, r=0.52,
              color=WHITE, hub=FUSION_C)
    ax.text(TR_X + TR_W / 2, TR_Y + TR_H - 2.75, 'FloraV3', ha='center',
            va='center', fontsize=14.5, color=WHITE, fontweight='bold', zorder=6)
    ax.text(TR_X + TR_W / 2, TR_Y + TR_H - 3.35,
            'MoE-Трансформер', ha='center', va='center',
            fontsize=11.5, color=WHITE, zorder=6)
    ax.text(TR_X + TR_W / 2, TR_Y + 1.85,
            'HRF-свёртка · гейтинг\nFiLM · MoE (top-2/8)',
            ha='center', va='center', fontsize=9.5, color='#cbd5e1', zorder=6)
    rbox(ax, TR_X + 0.35, TR_Y + 0.35, TR_W - 0.7, 0.85, fill='#2c3644',
         edge='none', r=0.10)
    ax.text(TR_X + TR_W / 2, TR_Y + 0.78, '9.7M обучаемых параметров',
            ha='center', va='center', fontsize=9.5, color=WHITE,
            fontweight='bold', zorder=6)

    tr_cy = TR_Y + TR_H / 2
    for mod, dy in [('Видео', 1.9), ('Аудио', 0.0), ('Текст', -1.9)]:
        cy = in_cy[mod]
        bus = ENC_X + ENC_W + 0.55
        ax.plot([ENC_X + ENC_W + 0.08, bus, bus],
                [cy, cy, tr_cy + dy], color='#334155', lw=1.8, zorder=3)
        arrow(ax, (bus, tr_cy + dy), (TR_X - 0.10, tr_cy + dy))

    # ── subject block ────────────────────────────────────────────────────
    SB_X, SB_W, SB_Y, SB_H = 12.35, 3.30, 2.60, 4.95
    rbox(ax, SB_X, SB_Y, SB_W, SB_H, fill=SUBJ_C, edge='none', r=0.14)
    icon_person(ax, SB_X + SB_W / 2, SB_Y + SB_H - 1.55, s=1.15, color=WHITE)
    ax.text(SB_X + SB_W / 2, SB_Y + SB_H - 2.75, 'Блок субъекта',
            ha='center', va='center', fontsize=12.5, color=WHITE,
            fontweight='bold', zorder=6)
    ax.text(SB_X + SB_W / 2, SB_Y + SB_H - 3.45, 'FiLM-кондиционирование',
            ha='center', va='center', fontsize=9.5, color='#e2e8f0', zorder=6)
    rbox(ax, SB_X + 0.35, SB_Y + 0.35, SB_W - 0.7, 1.15, fill='#475467',
         edge='none', r=0.10)
    ax.text(SB_X + SB_W / 2, SB_Y + 0.92, '400 парцеллов\nSchaefer-400',
            ha='center', va='center', fontsize=9.5, color=WHITE,
            fontweight='bold', zorder=6)
    sb_cy = SB_Y + SB_H / 2
    arrow(ax, (TR_X + TR_W + 0.10, tr_cy), (SB_X - 0.12, sb_cy))

    # ── dashed loss path ─────────────────────────────────────────────────
    rbox(ax, 8.05, 0.15, 3.35, 0.95, fill=WHITE, edge=MUTED, lw=1.2)
    ax.text(9.725, 0.625, 'Эталон (reference)', ha='center', va='center',
            fontsize=9.5, color=SLATE, zorder=6)
    rbox(ax, 12.35, 0.15, 3.30, 0.95, fill=WHITE, edge=CORAL, lw=1.4)
    ax.text(14.0, 0.625, r'Целевая функция $\mathcal{L}$', ha='center',
            va='center', fontsize=9.5, color=CORAL, zorder=6)
    arrow(ax, (9.725, 1.10), (9.725, TR_Y - 0.05), color=MUTED, lw=1.3, ls='--')
    arrow(ax, (14.0, SB_Y - 0.05), (14.0, 1.10), color=CORAL, lw=1.3, ls='--')
    ax.text(7.9, 9.15, '3 замороженных энкодера · 67.3M', fontsize=10,
            color=MUTED, ha='left')

    # ── right: predicted brain activity ──────────────────────────────────
    axb = fig.add_axes([0.675, 0.06, 0.315, 0.82], projection='3d')
    from nilearn import plotting
    axb.set_facecolor(WHITE)
    plotting.plot_surf_stat_map(fs.infl_left, B['act'][0], hemi='left',
                                view='lateral', bg_map=fs.sulc_left,
                                cmap='hot', vmin=0, vmax=1.0, threshold=0.35,
                                colorbar=False, axes=axb)
    axb.set_title('Предсказанная кортикальная активность',
                  fontsize=11.5, color=DARK, pad=6)
    panel_tag(axb, 'B', x=-0.06, y=1.02)
    hot_colorbar(fig, [0.745, 0.085, 0.17, 0.028], lo='Low', hi='High',
                 label='средняя активация (a.u.)')
    fig.text(0.664, 0.50, '$\\rightarrow$', fontsize=26, color=SLATE,
             ha='center', va='center')

    save(fig, 'fig_overview.png')


# ════════════════════════════════════════════════════════════════════════════
# 2. fig_hierarchy.png — brain -> surface -> Schaefer-400 -> Yeo-7
# ════════════════════════════════════════════════════════════════════════════

def fig_hierarchy():
    B = brain_data()
    fs = B['fs']
    from nilearn import plotting

    fig = plt.figure(figsize=(16.5, 5.1))
    fig.patch.set_facecolor(WHITE)

    # A: plain inflated surface
    ax1 = fig.add_axes([0.015, 0.06, 0.22, 0.80], projection='3d')
    ax1.set_facecolor(WHITE)
    plotting.plot_surf(fs.infl_left, bg_map=fs.sulc_left, hemi='left',
                       view='lateral', axes=ax1, cmap='Greys')
    ax1.set_title('Поверхность коры\nfsaverage5 · 20 484 вершины',
                  fontsize=10.5, color=DARK, pad=4)
    panel_tag(ax1, 'A', x=-0.02, y=1.04)

    # B: Schaefer-400 parcellation
    roi_ax(fig, [0.265, 0.06, 0.22, 0.80], fs.infl_left, B['roi'][0], 'left',
           'lateral', fs.sulc_left, cmap=ListedColormap(
               plt.cm.gist_ncar(np.linspace(0, 1, 400))),
           title='Парцелляция\nSchaefer-400 · 400 парцеллов')
    panel_tag(fig.axes[-1], 'B', x=-0.06, y=1.04)

    # C: Yeo-7 networks
    roi_ax(fig, [0.515, 0.06, 0.22, 0.80], fs.infl_left, B['net'][0], 'left',
           'lateral', fs.sulc_left, cmap=ListedColormap(YEO7_COLORS),
           title='Функциональные сети\nYeo-7')
    panel_tag(fig.axes[-1], 'C', x=-0.06, y=1.04)

    # arrows between panels
    axa = fig.add_axes([0, 0, 1, 1]); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    axa.axis('off'); axa.set_zorder(-1)
    for x0, x1 in [(0.236, 0.262), (0.486, 0.512)]:
        arrow(axa, (x0, 0.5), (x1, 0.5), lw=2.2, ms=18)

    # D: legend
    axl = fig.add_axes([0.755, 0.06, 0.235, 0.84])
    axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.axis('off')
    panel_tag(axl, 'D', x=0.0, y=1.02)
    axl.text(0.02, 0.92, '7 сетей Yeo', fontsize=12, fontweight='bold',
             color=DARK, va='top')
    for i, (name, ru, col) in enumerate(zip(YEO7, YEO7_RU, YEO7_COLORS)):
        y = 0.78 - i * 0.105
        rbox(axl, 0.02, y - 0.032, 0.16, 0.064, fill=col, edge='none', r=0.02)
        axl.text(0.23, y, f'{name}', fontsize=10, color=DARK,
                 fontweight='bold', va='center')
        axl.text(0.60, y, ru, fontsize=9.5, color=MUTED, va='center')
    axl.text(0.02, 0.02, 'Визуальная кора → видео,\nвисочная → речь и т.д.',
             fontsize=9, color=MUTED, va='bottom', style='italic')

    save(fig, 'fig_hierarchy.png')


# ════════════════════════════════════════════════════════════════════════════
# 3. fig_hrf.png — canonical double-gamma HRF
# ════════════════════════════════════════════════════════════════════════════

def fig_hrf():
    from scipy.special import gamma as gamma_fn

    t = np.linspace(0, 32, 800)

    def g(t, a, b):
        return t ** (a - 1) * np.exp(-t / b) / (b ** a * gamma_fn(a))

    g1 = g(t, 6, 1)
    g2 = g(t, 16, 1) / 6
    h = g1 - g2

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.grid(color=GRID, lw=0.8, zorder=0)

    ax.plot(t, g1, color=CORAL, lw=2.0, ls='--', zorder=3,
            label=r'$\gamma_1$: главный пик ($a_1{=}6, b_1{=}1$)')
    ax.plot(t, -g2, color='#2563eb', lw=2.0, ls='--', zorder=3,
            label=r'$-\frac{1}{6}\gamma_2$: спад ($a_2{=}16, b_2{=}1$)')
    ax.plot(t, h, color=DARK, lw=3.0, zorder=4,
            label=r'$h(t)=\gamma_1-\frac{1}{6}\gamma_2$ — каноническая HRF')
    ax.axhline(0, color=MUTED, lw=1.0)

    t_peak = t[np.argmax(h)]
    ax.axvline(t_peak, color=CORAL, lw=1.2, ls=':', zorder=2)
    ax.annotate('пик 5–6 с', xy=(t_peak, h.max()), xytext=(t_peak + 2.2, h.max() * 0.92),
                fontsize=11, color=CORAL, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=CORAL, lw=1.3))
    t_under = t[np.argmin(h)]
    ax.axvline(t_under, color='#2563eb', lw=1.2, ls=':', zorder=2)
    ax.annotate('постстимульное падение\n~12–16 с', xy=(t_under, h.min()),
                xytext=(t_under + 3.2, -0.026),
                fontsize=10.5, color='#2563eb', fontweight='bold',
                va='center',
                arrowprops=dict(arrowstyle='-|>', color='#2563eb', lw=1.3))
    ax.fill_between(t, 0, h, where=(t < t_peak), color=CORAL, alpha=0.06,
                    zorder=1)
    ax.fill_between(t, 0, h, where=(h < 0), color='#2563eb', alpha=0.08,
                    zorder=1)

    ax.set_xlim(0, 32)
    ax.set_ylim(-0.034, 0.19)
    ax.set_xlabel('Время после стимула, с', fontsize=12.5, color=DARK)
    ax.set_ylabel('BOLD-отклик (a.u.)', fontsize=12.5, color=DARK)
    ax.set_title('Каноническая двойная гамма-HRF (Glover 1999)',
                 fontsize=14.5, fontweight='bold', color=DARK, pad=12)
    leg = ax.legend(loc='upper right', fontsize=10, frameon=True,
                    facecolor=WHITE, edgecolor=BORDER)
    for s in ax.spines.values():
        s.set_color(BORDER)
    fig.tight_layout()
    save(fig, 'fig_hrf.png')


# ════════════════════════════════════════════════════════════════════════════
# 4. fig_projection.png — voxels -> fsaverage5 -> Schaefer-400 -> Yeo-7
# ════════════════════════════════════════════════════════════════════════════

def fig_projection():
    B = brain_data()
    fs = B['fs']
    from nilearn import plotting

    fig = plt.figure(figsize=(17.0, 4.9))
    fig.patch.set_facecolor(WHITE)

    # A: volumetric glass brain
    axg = fig.add_axes([0.015, 0.06, 0.20, 0.74])
    plotting.plot_glass_brain(B['act_img'], display_mode='xz', cmap='hot',
                              colorbar=False, threshold=0.30, black_bg=False,
                              figure=fig, axes=axg, title=None)
    fig.text(0.115, 0.88, 'Объёмный фМРТ-сигнал\n(воксели, MNI)',
             fontsize=10.5, color=DARK, ha='center', va='top',
             fontweight='bold')
    fig.text(0.012, 0.92, 'A', fontsize=17, fontweight='bold', color=DARK,
             ha='left', va='top')

    # B: surface render
    surf_ax(fig, [0.265, 0.06, 0.20, 0.82], fs.infl_left, B['act'][0],
            'left', 'lateral', fs.sulc_left, title='fsaverage5\n20 484 вершины')
    panel_tag(fig.axes[-1], 'B', x=-0.10, y=1.04)

    # C: parcellation
    roi_ax(fig, [0.515, 0.06, 0.20, 0.82], fs.infl_left, B['roi'][0], 'left',
           'lateral', fs.sulc_left,
           cmap=ListedColormap(plt.cm.gist_ncar(np.linspace(0, 1, 400))),
           title='Schaefer-400\n400 парцеллов')
    panel_tag(fig.axes[-1], 'C', x=-0.10, y=1.04)

    # D: Yeo-7
    roi_ax(fig, [0.765, 0.06, 0.20, 0.82], fs.infl_left, B['net'][0], 'left',
           'lateral', fs.sulc_left, cmap=ListedColormap(YEO7_COLORS),
           title='Yeo-7\n7 сетей')
    panel_tag(fig.axes[-1], 'D', x=-0.10, y=1.04)

    # arrows with labels
    axa = fig.add_axes([0, 0, 1, 1]); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    axa.axis('off'); axa.set_zorder(-1)
    for x0, x1, lab in [(0.218, 0.262, 'vol_to_surf'),
                        (0.468, 0.512, 'parcellate'),
                        (0.718, 0.762, 'группировка')]:
        arrow(axa, (x0, 0.5), (x1, 0.5), lw=2.0, ms=16)
        axa.text((x0 + x1) / 2, 0.56, lab, fontsize=8.5, color=MUTED,
                 ha='center', style='italic')

    save(fig, 'fig_projection.png')


# ════════════════════════════════════════════════════════════════════════════
# 5. fig_motion.png — TemporalMotionModule
# ════════════════════════════════════════════════════════════════════════════

def fig_motion():
    rng = np.random.default_rng(3)

    def frame(cx):
        yy, xx = np.mgrid[0:64, 0:64]
        img = np.exp(-(((xx - cx) ** 2 + (yy - 32) ** 2) / 140.0))
        img += 0.35 * np.exp(-(((xx - 46) ** 2 + (yy - 14) ** 2) / 60.0))
        img += rng.normal(0, 0.03, img.shape)
        return np.clip(img, 0, 1)

    f1, f2 = frame(22), frame(30)
    diff = f2 - f1

    fig = plt.figure(figsize=(15.5, 4.1))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15.5); ax.set_ylim(0, 4.1)
    ax.axis('off')
    panel_tag(ax, 'A')

    def img_panel(x, img, cmap, title, vmin=None, vmax=None):
        iax = fig.add_axes([(x - 0.2) / 15.5, 0.24, 1.75 / 15.5, 0.56])
        iax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_color(BORDER)
        ax.text(x + 0.675, 3.42, title, ha='center', fontsize=10.5,
                color=DARK, fontweight='bold')

    img_panel(1.05, f1, 'gray', 'кадр $t-1$')
    img_panel(3.45, f2, 'gray', 'кадр $t$')
    img_panel(5.85, diff, 'RdBu_r', 'разность (движение)', vmin=-0.6, vmax=0.6)

    arrow(ax, (2.95, 2.05), (3.35, 2.05), lw=1.8)
    arrow(ax, (5.35, 2.05), (5.75, 2.05), lw=1.8)

    # kernel stem
    kax = fig.add_axes([8.15 / 15.5, 0.17, 1.9 / 15.5, 0.50])
    ml, sl, bl = kax.stem([-1, 0, 1], [-0.5, 0.0, 0.5])
    plt.setp(ml, color=CORAL, ms=7)
    plt.setp(sl, color=CORAL, lw=2.2)
    plt.setp(bl, color=MUTED, lw=0.9)
    kax.set_xticks([-1, 0, 1]); kax.set_xticklabels(['$t-1$', '$t$', '$t+1$'])
    kax.axhline(0, color=MUTED, lw=0.9)
    kax.set_ylim(-0.85, 0.85)
    kax.grid(color=GRID, lw=0.7)
    kax.set_title('ядро DWConv1D $k{=}3$:  $[-\\frac{1}{2},\\,0,\\,\\frac{1}{2}]$',
                  fontsize=10, color=DARK, fontweight='bold', pad=8)
    for s in kax.spines.values():
        s.set_color(BORDER)
    arrow(ax, (7.75, 2.05), (8.1, 2.05), lw=1.8)

    # formula block
    rbox(ax, 10.55, 1.15, 4.35, 1.85, fill=PANEL, edge=BORDER, lw=1.3)
    ax.text(12.725, 2.62, 'Резидуальное сложение', ha='center',
            fontsize=11, color=DARK, fontweight='bold')
    ax.text(12.725, 1.95, r'$\tilde{h}_v(t) = h_v(t) + '
                          r'\mathrm{DWConv1D}_{k=3}\,h_v(t)$',
            ha='center', va='center', fontsize=13.5, color=DARK)
    ax.text(12.725, 1.38, '~1.5 тыс. параметров · кора MT+/V5', ha='center',
            fontsize=9.5, color=MUTED)
    arrow(ax, (10.15, 2.05), (10.5, 2.05), lw=1.8)

    save(fig, 'fig_motion.png')


# ════════════════════════════════════════════════════════════════════════════
# 6. fig_moe_block.png — MoE-transformer block + HRF attention bias heatmap
# ════════════════════════════════════════════════════════════════════════════

def fig_moe_block():
    fig = plt.figure(figsize=(16.0, 6.7))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.7)
    ax.axis('off')
    panel_tag(ax, 'A', x=0.008, y=0.97)

    steps = [
        ('Токены\n$(B,\\,3T,\\,D)$', PANEL, DARK),
        ('LayerNorm', PANEL, DARK),
        ('Multi-Head\nSelf-Attention', '#dbeafe', DARK),
        ('+ HRF-смещение\n$B_{ij}=-\\alpha|\\Delta\\tau_{ij}|$', '#fee2e2', CORAL),
        ('Add & Norm', PANEL, DARK),
        ('MoE FFN\n8 экспертов · top-2', '#e0f2fe', DARK),
        ('Add & Norm\nвыход $(B,\\,3T,\\,D)$', PANEL, DARK),
    ]
    x = 0.50
    w, h, y = 1.80, 1.55, 4.55
    centers = []
    for i, (txt, fill, tc) in enumerate(steps):
        edge = CORAL if tc == CORAL else BORDER
        rbox(ax, x, y, w, h, fill=fill, edge=edge, lw=1.4)
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center',
                fontsize=9.2, color=tc, fontweight='bold', zorder=6)
        centers.append(x + w / 2)
        if i:
            arrow(ax, (x - 0.40, y + h / 2), (x - 0.04, y + h / 2), lw=1.7)
        x += w + 0.40

    # residual arcs above the row
    for x0, x1 in [(centers[1], centers[4]), (centers[4], centers[6])]:
        ax.annotate('', xy=(x1, y + h + 0.10), xytext=(x0, y + h + 0.10),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4,
                                    connectionstyle='arc3,rad=-0.22',
                                    mutation_scale=13))
        ax.text((x0 + x1) / 2, y + h + 0.78, 'резидуальная связь',
                fontsize=8.5, color=MUTED, ha='center')

    # B: HRF-decay bias heatmap
    hax = fig.add_axes([0.045, 0.065, 0.185, 0.335])
    n = 12
    Bm = -0.35 * np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    im = hax.imshow(Bm, cmap='magma_r', vmin=-3.0, vmax=0)
    hax.set_title('Матрица HRF-смещения $B_{ij}$\n(слои 1–2)',
                  fontsize=10, color=DARK, fontweight='bold', pad=5)
    hax.set_xlabel('ключ $j$', fontsize=8.5, color=MUTED)
    hax.set_ylabel('запрос $i$', fontsize=8.5, color=MUTED)
    hax.tick_params(labelsize=7.5)
    cb = fig.colorbar(im, ax=hax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    cb.outline.set_edgecolor(BORDER)
    panel_tag(hax, 'B', x=-0.26, y=1.09)

    # C: decay curve
    dax = fig.add_axes([0.285, 0.085, 0.145, 0.29])
    d = np.arange(0, 12)
    alpha = np.log(2) / 6
    dax.plot(d, np.exp(-alpha * d), color=CORAL, lw=2.4, marker='o', ms=4)
    dax.axhline(0.5, color=MUTED, ls=':', lw=1.2)
    dax.axvline(6, color=MUTED, ls=':', lw=1.2)
    dax.text(6.3, 0.55, 'вес 0.5\nна 6 TR', fontsize=8, color=MUTED)
    dax.set_xlabel('$|\\Delta\\tau|$ (TR)', fontsize=8.5, color=MUTED)
    dax.set_ylabel('вес внимания', fontsize=8.5, color=MUTED)
    dax.set_title('Затухание внимания\n$\\exp(-\\alpha|\\Delta\\tau|)$',
                  fontsize=10, color=DARK, fontweight='bold', pad=5)
    dax.grid(color=GRID, lw=0.7)
    dax.tick_params(labelsize=7.5)
    for s in dax.spines.values():
        s.set_color(BORDER)
    panel_tag(dax, 'C', x=-0.30, y=1.11)

    # D: expert bars
    eax = fig.add_axes([0.475, 0.085, 0.145, 0.29])
    probs = np.array([0.04, 0.31, 0.06, 0.09, 0.31, 0.05, 0.08, 0.06])
    colors = [CORAL if i in (1, 4) else '#94a3b8' for i in range(8)]
    eax.bar(range(8), probs, color=colors, edgecolor=BORDER, lw=0.6)
    eax.set_xticks(range(8))
    eax.set_xticklabels([f'$E_{i+1}$' for i in range(8)], fontsize=7.5)
    eax.set_ylabel('$g_e(x)$', fontsize=8.5, color=MUTED)
    eax.set_title('Роутер: top-2 из 8\n(красные — выбраны)',
                  fontsize=10, color=DARK, fontweight='bold', pad=5)
    eax.grid(axis='y', color=GRID, lw=0.7)
    eax.tick_params(labelsize=7.5)
    for s in eax.spines.values():
        s.set_color(BORDER)
    panel_tag(eax, 'D', x=-0.30, y=1.11)

    # E: notes card
    rbox(ax, 10.35, 0.55, 5.15, 2.55, fill=PANEL, edge=BORDER, lw=1.2)
    notes = [
        'Слои 1–2: локальное внимание + HRF-смещение',
        'Слои 3–4: полное внимание (нарративная интеграция)',
        'FFN: MoE, 8 экспертов, top-2 маршрутизация',
        'aux-loss балансировки + z-loss роутера',
        'Стохастическая глубина $0 \\to 0.2$',
    ]
    ax.text(10.65, 2.72, 'Конструкция блока', fontsize=10.5,
            fontweight='bold', color=DARK, va='center')
    for i, s in enumerate(notes):
        ax.text(10.65, 2.28 - i * 0.42, '• ' + s, fontsize=9.3,
                color=SLATE, va='center')
    panel_tag(ax, 'E', x=0.637, y=0.42)

    save(fig, 'fig_moe_block.png')


# ════════════════════════════════════════════════════════════════════════════
# 7. fig_moe_router.png — top-2 routing diagram
# ════════════════════════════════════════════════════════════════════════════

def fig_moe_router():
    fig = plt.figure(figsize=(14.5, 5.3))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14.5); ax.set_ylim(0, 5.3)
    ax.axis('off')
    panel_tag(ax, 'A')

    # input
    rbox(ax, 0.4, 2.35, 1.9, 1.1, fill=PANEL, edge=BORDER, lw=1.3)
    ax.text(1.35, 2.9, 'входной\nтокен $x$', ha='center', va='center',
            fontsize=10.5, color=DARK, fontweight='bold')

    # router
    rbox(ax, 3.3, 2.05, 2.5, 1.7, fill='#fff7ed', edge='#ea580c', lw=1.6)
    ax.text(4.55, 3.28, 'Роутер', ha='center', fontsize=12, color='#ea580c',
            fontweight='bold')
    ax.text(4.55, 2.72, '$g(x)=W_r x$\n$\\mathrm{Top2}(\\mathrm{softmax}\\,g)$',
            ha='center', va='center', fontsize=9.5, color=DARK)
    arrow(ax, (2.35, 2.9), (3.25, 2.9), lw=1.8)

    # experts 2x4
    EX_X0, EX_Y0 = 6.35, 1.35
    EW, EH, GX, GY = 1.35, 1.05, 0.20, 0.42
    chosen = {1: 0.62, 4: 0.38}   # E2 and E5 (0-based)
    for i in range(8):
        r, c = divmod(i, 4)
        x = EX_X0 + c * (EW + GX)
        y = EX_Y0 + (1 - r) * (EH + GY)
        on = i in chosen
        rbox(ax, x, y, EW, EH,
             fill=('#fee2e2' if on else PANEL),
             edge=(CORAL if on else BORDER), lw=(1.8 if on else 1.1))
        ax.text(x + EW / 2, y + EH * 0.62, f'$E_{i+1}$', ha='center',
                fontsize=11, color=(CORAL if on else MUTED),
                fontweight='bold')
        ax.text(x + EW / 2, y + EH * 0.24, 'FFN', ha='center',
                fontsize=8, color=MUTED)
        ty = y + EH / 2
        arrow(ax, (5.85, 2.9), (x - 0.05, ty),
              color=(CORAL if on else '#cbd5e1'),
              lw=(1.9 if on else 1.1),
              ls=('-' if on else '--'),
              connectionstyle='arc3,rad=0.12', ms=(13 if on else 9))
        if on:
            ly = y + EH + 0.14 if r == 0 else y - 0.16
            ax.text(x + EW / 2, ly, f'$w_{i+1}={chosen[i]:.2f}$',
                    fontsize=9.5, color=CORAL, fontweight='bold',
                    ha='center', va=('bottom' if r == 0 else 'top'))

    # output
    OUT_X = 12.85
    rbox(ax, OUT_X, 2.35, 1.55, 1.1, fill='#e0f2fe', edge='#0284c7', lw=1.4)
    ax.text(OUT_X + 0.775, 2.9, 'выход\n$y=\\sum_e w_e E_e(x)$', ha='center',
            va='center', fontsize=8.8, color=DARK, fontweight='bold')
    for i in chosen:
        r, c = divmod(i, 4)
        x = EX_X0 + c * (EW + GX)
        y = EX_Y0 + (1 - r) * (EH + GY)
        arrow(ax, (x + EW + 0.05, y + EH / 2), (OUT_X - 0.06, 2.9),
              color=CORAL, lw=1.9, connectionstyle='arc3,rad=-0.12', ms=13)

    # losses note
    rbox(ax, 3.3, 0.18, 8.6, 0.80, fill=WHITE, edge=BORDER, lw=1.1)
    ax.text(7.6, 0.58, 'балансировочный aux-loss  $\\mathcal{L}_{aux}=E\\sum_e f_e P_e$'
                       '   ·   z-loss роутера  $10^{-3}\\,\\mathbb{E}[g(x)^2]$',
            ha='center', va='center', fontsize=9.5, color=SLATE)

    save(fig, 'fig_moe_router.png')


# ════════════════════════════════════════════════════════════════════════════
# 8. fig_gating.png — modality gating + dynamics
# ════════════════════════════════════════════════════════════════════════════

def fig_gating():
    fig = plt.figure(figsize=(14.5, 5.2))
    fig.patch.set_facecolor(WHITE)

    # A: diagram
    ax = fig.add_axes([0.02, 0.02, 0.42, 0.90])
    ax.set_xlim(0, 7); ax.set_ylim(0, 6); ax.axis('off')
    panel_tag(ax, 'A', x=0.0, y=1.0)

    mods = [('$h_v$ · видео', VIDEO_C, 4.7),
            ('$h_a$ · аудио', AUDIO_C, 3.0),
            ('$h_\\ell$ · текст', TEXT_C, 1.3)]
    for txt, col, y in mods:
        rbox(ax, 0.25, y, 1.75, 0.95, fill=col, edge='none')
        ax.text(1.125, y + 0.475, txt, ha='center', va='center',
                fontsize=9.5, color=WHITE, fontweight='bold')
        arrow(ax, (2.05, y + 0.475), (3.35, 3.0), color=col, lw=1.8,
              connectionstyle='arc3,rad=0.08')

    rbox(ax, 3.4, 2.25, 1.85, 1.5, fill='#fff7ed', edge='#ea580c', lw=1.6)
    ax.text(4.325, 3.32, 'гейт', ha='center', fontsize=11, color='#ea580c',
            fontweight='bold')
    ax.text(4.325, 2.78, '$g=\\mathrm{softmax}$\n$(W_g\\bar h + b_g)$',
            ha='center', va='center', fontsize=9, color=DARK)

    rbox(ax, 5.6, 2.45, 1.25, 1.1, fill=FUSION_C, edge='none')
    ax.text(6.225, 3.0, '$\\hat h=\\sum_m g_m h_m$', ha='center', va='center',
            fontsize=9.5, color=WHITE, fontweight='bold')
    arrow(ax, (5.3, 3.0), (5.55, 3.0), lw=1.8)
    ax.text(0.25, 5.72, 'Мягкое взвешивание трёх\nмодальностей на каждом шаге',
            fontsize=10, color=MUTED, ha='left', va='top', style='italic')
    ax.text(0.25, 0.55, 'инициализация нулями\n→ стартовые веса $1/3$',
            fontsize=9, color=MUTED, ha='left', va='center')

    # B: gate dynamics stacked area
    bx = fig.add_axes([0.50, 0.13, 0.46, 0.72])
    T = np.linspace(0, 4, 60)
    v = 0.55 + 0.25 * np.exp(-((T - 1.0) ** 2) / 1.2)
    a = 0.30 + 0.28 * np.exp(-((T - 2.6) ** 2) / 0.9)
    l = 0.15 + 0.30 * np.exp(-((T - 3.8) ** 2) / 0.8)
    s = v + a + l
    v, a, l = v / s, a / s, l / s
    bx.stackplot(T, v, a, l, colors=[VIDEO_C, AUDIO_C, TEXT_C], alpha=0.88,
                 labels=['видео $g_v$', 'аудио $g_a$', 'текст $g_\\ell$'])
    bx.axhline(1 / 3, color=MUTED, ls=':', lw=1.2)
    bx.text(3.96, 1 / 3 + 0.015, '$1/3$', fontsize=8.5, color=MUTED,
            va='bottom', ha='right')
    bx.set_xlim(0, 4.0); bx.set_ylim(0, 1.0)
    bx.set_xlabel('время внутри клипа (TR)', fontsize=11.5, color=DARK)
    bx.set_ylabel('вес модальности $g_m(t)$', fontsize=11.5, color=DARK)
    bx.set_title('Динамика гейтов: зрительные сцены → видео,\nреплики → аудио и текст',
                 fontsize=12, fontweight='bold', color=DARK, pad=8)
    bx.legend(loc='upper left', fontsize=9.5, frameon=True, facecolor=WHITE,
              edgecolor=BORDER)
    bx.grid(color=GRID, lw=0.7)
    for sp in bx.spines.values():
        sp.set_color(BORDER)
    panel_tag(bx, 'B', x=-0.10, y=1.04)

    save(fig, 'fig_gating.png')


# ════════════════════════════════════════════════════════════════════════════
# 9. fig_film.png — FiLM conditioning
# ════════════════════════════════════════════════════════════════════════════

def fig_film():
    fig = plt.figure(figsize=(14.5, 5.2))
    fig.patch.set_facecolor(WHITE)

    # A: diagram
    ax = fig.add_axes([0.02, 0.02, 0.44, 0.90])
    ax.set_xlim(0, 7.2); ax.set_ylim(0, 6); ax.axis('off')
    panel_tag(ax, 'A', x=0.0, y=1.0)

    rbox(ax, 0.25, 2.5, 1.8, 1.0, fill=SUBJ_C, edge='none')
    ax.text(1.15, 3.0, 'субъект $s$\nэмбеддинг', ha='center', va='center',
            fontsize=9.5, color=WHITE, fontweight='bold')
    rbox(ax, 2.75, 4.0, 1.5, 0.85, fill='#fee2e2', edge=CORAL, lw=1.4)
    ax.text(3.5, 4.42, '$\\gamma_s$ (масштаб)', ha='center', va='center',
            fontsize=9, color=CORAL, fontweight='bold')
    rbox(ax, 2.75, 1.15, 1.5, 0.85, fill='#dbeafe', edge='#2563eb', lw=1.4)
    ax.text(3.5, 1.57, '$\\beta_s$ (сдвиг)', ha='center', va='center',
            fontsize=9, color='#2563eb', fontweight='bold')
    arrow(ax, (2.1, 3.25), (2.7, 4.3), color=CORAL, lw=1.6)
    arrow(ax, (2.1, 2.75), (2.7, 1.7), color='#2563eb', lw=1.6)

    rbox(ax, 4.85, 2.5, 1.0, 1.0, fill=PANEL, edge=BORDER, lw=1.3)
    ax.text(5.35, 3.0, '$\\odot$\n$+$', ha='center', va='center',
            fontsize=13, color=DARK, fontweight='bold')
    arrow(ax, (4.3, 4.42), (5.2, 3.55), color=CORAL, lw=1.6)
    arrow(ax, (4.3, 1.57), (5.2, 2.45), color='#2563eb', lw=1.6)

    rbox(ax, 0.25, 0.15, 1.8, 0.85, fill=PANEL, edge=BORDER, lw=1.3)
    ax.text(1.15, 0.57, 'общее $x$\n$(B,T,D)$', ha='center', va='center',
            fontsize=9, color=DARK, fontweight='bold')
    arrow(ax, (2.1, 0.57), (5.0, 2.45), color=MUTED, lw=1.5,
          connectionstyle='arc3,rad=-0.18')

    rbox(ax, 6.1, 2.5, 1.0, 1.0, fill=FUSION_C, edge='none')
    ax.text(6.6, 3.0, '$\\hat y_s$', ha='center', va='center', fontsize=11,
            color=WHITE, fontweight='bold')
    arrow(ax, (5.9, 3.0), (6.05, 3.0), lw=1.8)
    ax.text(3.6, 5.55, '$y_s = \\gamma_s \\odot x + \\beta_s$ — два вектора длины $D$ на субъекта',
            fontsize=10.5, color=DARK, ha='center', fontweight='bold')

    # B: effect plot
    bx = fig.add_axes([0.52, 0.14, 0.44, 0.70])
    t = np.linspace(0, 4 * np.pi, 200)
    x_sig = np.sin(t) * np.exp(-t / 14)
    bx.plot(t, x_sig, color=MUTED, lw=2.2, label='общее $x$ (до FiLM)')
    bx.plot(t, 1.25 * x_sig + 0.18, color=CORAL, lw=2.0, ls='--',
            label='субъект 1: $\\gamma{=}1.25,\\ \\beta{=}+0.18$')
    bx.plot(t, 0.78 * x_sig - 0.12, color=TEAL, lw=2.0, ls='--',
            label='субъект 2: $\\gamma{=}0.78,\\ \\beta{-}0.12$')
    bx.axhline(0, color=GRID, lw=1.0)
    bx.set_xlabel('скрытое измерение / время', fontsize=11, color=DARK)
    bx.set_ylabel('активация', fontsize=11, color=DARK)
    bx.set_title('FiLM: дешёвая персубъектная адаптация\n'
                 '$2D = 1\\ 024$ параметров вместо $32.8$M матриц',
                 fontsize=12, fontweight='bold', color=DARK, pad=8)
    bx.legend(fontsize=9, frameon=True, facecolor=WHITE, edgecolor=BORDER,
              loc='upper right')
    bx.grid(color=GRID, lw=0.7)
    for sp in bx.spines.values():
        sp.set_color(BORDER)
    bx.set_xticks([])
    panel_tag(bx, 'B', x=-0.10, y=1.04)

    save(fig, 'fig_film.png')


# ════════════════════════════════════════════════════════════════════════════
ALL = {
    'overview': fig_overview,
    'hierarchy': fig_hierarchy,
    'hrf': fig_hrf,
    'projection': fig_projection,
    'motion': fig_motion,
    'moe_block': fig_moe_block,
    'moe_router': fig_moe_router,
    'gating': fig_gating,
    'film': fig_film,
}

if __name__ == '__main__':
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(f'== {n} ==')
        ALL[n]()
    print('All paper figures ->', OUT)
