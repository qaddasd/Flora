# -*- coding: utf-8 -*-
"""Publication-quality pipeline figure for Flora (A4 print friendly).

Style: minimalist flat vector, white background, strict publication palette
(white / dark slate #0f172a / muted slate #475467 / coral #dc2626),
crisp outlines and right-angle elbow arrows.

Layout (left -> right):
    input panels (Video / Audio / Text) -> frozen encoders ->
    Transformer -> Subject Block.

Renders assets/flora_architecture.png
"""
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle

plt.rcParams['font.sans-serif'] = ['Montserrat', 'Inter', 'Segoe UI',
                                   'DejaVu Sans', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'

OUT = Path(__file__).resolve().parent.parent / 'assets' / 'flora_architecture.png'

# ── publication light palette ─────────────────────────────────────────
WHITE = '#ffffff'
BG_PANEL = '#f8fafc'
DARK = '#0f172a'
SLATE = '#475467'
SLATE_LIGHT = '#e2e8f0'
CORAL = '#dc2626'
CORAL_LIGHT = '#fee2e2'
OUTLINE = '#94a3b8'
ARROW_COLOR = '#334155'


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

def panel(ax, x, y, w, h, fill=BG_PANEL, edge=OUTLINE, lw=1.2):
    """Rounded rectangle."""
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.09',
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def elbow_arrow(ax, pts, color=ARROW_COLOR, lw=1.5, head=0.16):
    """Polyline with a small triangular head at the last point."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle='butt',
            zorder=3)
    # last segment -> triangle head
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = np.hypot(dx, dy)
    ux, uy = dx / n, dy / n            # unit along
    px, py = -uy, ux                   # unit perpendicular
    b1 = (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55)
    b2 = (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)
    ax.add_patch(Polygon([(x1, y1), b1, b2], closed=True,
                         facecolor=color, edgecolor='none', zorder=3))
    ax.plot([xs[-2], x1 - ux * head], [ys[-2], y1 - uy * head],
            color=color, lw=lw, solid_capstyle='butt', zorder=3)


# ── monochrome icons ─────────────────────────────────────────────────

def icon_video(ax, cx, cy, s=1.0):
    """Video player: rounded frame + play triangle."""
    w, h = 0.86 * s, 0.58 * s
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle='round,pad=0.045', linewidth=1.6,
                       edgecolor=DARK, facecolor=WHITE, zorder=5)
    ax.add_patch(p)
    t = 0.19 * s
    ax.add_patch(Polygon([(cx - t * 0.55, cy - t), (cx - t * 0.55, cy + t),
                          (cx + t * 0.85, cy)], closed=True,
                         facecolor=DARK, edgecolor='none', zorder=6))


def icon_waveform(ax, cx, cy, s=1.0):
    """Audio waveform: symmetric vertical bars."""
    heights = [0.16, 0.30, 0.46, 0.34, 0.52, 0.28, 0.42, 0.20, 0.34]
    bw = 0.055 * s
    gap = 0.088 * s
    x0 = cx - gap * (len(heights) - 1) / 2
    for i, hh in enumerate(heights):
        h = hh * s
        ax.add_patch(Rectangle((x0 + i * gap - bw / 2, cy - h / 2), bw, h,
                               facecolor=DARK, edgecolor='none', zorder=5))


def icon_text(ax, cx, cy, s=1.0):
    """Text snippet: quote lines."""
    ax.text(cx, cy + 0.10 * s, '“Hey, how you doin’?”',
            ha='center', va='center', fontsize=10.5, color=DARK,
            style='italic', zorder=5)
    for dy, wfrac in ((-0.16, 0.62), (-0.30, 0.40)):
        ax.plot([cx - wfrac * s, cx + wfrac * s], [cy + dy * s, cy + dy * s],
                color='#64748b', lw=1.4, solid_capstyle='round', zorder=5)


def icon_gear(ax, cx, cy, r=0.42):
    """Gear icon with hub hole."""
    n_teeth = 8
    for k in range(n_teeth):
        ang = 2 * np.pi * k / n_teeth
        tx, ty = np.cos(ang), np.sin(ang)
        w = r * 0.34
        L = r * 0.42
        verts = []
        for al, rr in ((-w, r * 0.78), (w, r * 0.78), (w * 0.8, r * 0.78 + L),
                       (-w * 0.8, r * 0.78 + L)):
            px_, py_ = -ty, tx
            verts.append((cx + tx * rr + px_ * al, cy + ty * rr + py_ * al))
        ax.add_patch(Polygon(verts, closed=True, facecolor=DARK,
                             edgecolor='none', zorder=5))
    ax.add_patch(Circle((cx, cy), r * 0.82, facecolor=DARK,
                        edgecolor='none', zorder=5))
    ax.add_patch(Circle((cx, cy), r * 0.34, facecolor=CORAL_LIGHT,
                        edgecolor='none', zorder=6))


def icon_people(ax, cx, cy, s=1.0):
    """People group: three silhouettes (head + shoulders)."""
    def person(px, py, sc, z):
        ax.add_patch(Circle((px, py + 0.26 * sc), 0.135 * sc,
                            facecolor=DARK, edgecolor='none', zorder=z))
        th = np.linspace(np.pi, 0, 40)
        shoulder_x = px + 0.24 * sc * np.cos(th)
        shoulder_y = py + 0.30 * sc * np.sin(th) - 0.28 * sc
        ax.add_patch(Polygon(np.column_stack([shoulder_x, shoulder_y]),
                             closed=True, facecolor=DARK,
                             edgecolor='none', zorder=z))
    person(cx - 0.36 * s, cy - 0.05 * s, 0.80 * s, 5)
    person(cx + 0.36 * s, cy - 0.05 * s, 0.80 * s, 5)
    person(cx, cy + 0.05 * s, 1.00 * s, 6)


# ══════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 7), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(2.70, 9.70)
ax.axis('off')
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# ── left column: input panels ────────────────────────────────────────
IN_X, IN_W = 0.9, 2.6
inputs = [
    ('Video', icon_video, 7.55),
    ('Audio', icon_waveform, 5.35),
    ('Text', icon_text, 3.15),
]
in_centers = {}
for label, icon, y in inputs:
    h = 1.70
    panel(ax, IN_X, y, IN_W, h, fill='#f8fafc', edge='#cbd5e1')
    icon(ax, IN_X + IN_W / 2, y + h * 0.60)
    ax.text(IN_X + IN_W / 2, y + h * 0.16, label, ha='center', va='center',
            fontsize=12.5, color=DARK, fontweight='bold', zorder=6)
    in_centers[label] = y + h / 2

# ── middle column: frozen encoders ───────────────────────────────────
ENC_X, ENC_W, ENC_H = 5.10, 2.90, 1.05
encoders = [
    ('MobileViT-S', 'Video'),
    ('Whisper-Tiny', 'Audio'),
    ('MiniLM-L6-v2', 'Text'),
]
enc_centers = {}
for name, mod in encoders:
    cy = in_centers[mod]
    panel(ax, ENC_X, cy - ENC_H / 2, ENC_W, ENC_H, fill=SLATE_LIGHT, edge='#94a3b8')
    ax.text(ENC_X + ENC_W / 2, cy, name, ha='center', va='center',
            fontsize=12.5, color=DARK, fontweight='bold', zorder=6)
    enc_centers[mod] = cy
    # input panel -> encoder
    elbow_arrow(ax, [(IN_X + IN_W + 0.10, cy), (ENC_X - 0.14, cy)])

# ── right: Transformer + Subject Block ───────────────────────────────
TR_X, TR_W = 9.70, 2.20
SB_X, SB_W = 13.10, 2.20
TALL_Y, TALL_H = 3.60, 5.20
TR_CX, SB_CX = TR_X + TR_W / 2, SB_X + SB_W / 2
TR_CY = TALL_Y + TALL_H / 2

panel(ax, TR_X, TALL_Y, TR_W, TALL_H, fill=CORAL_LIGHT, edge=CORAL)
icon_gear(ax, TR_CX, TR_CY + 0.85, r=0.46)
ax.text(TR_CX, TALL_Y + 0.75, 'Transformer', ha='center', va='center',
        fontsize=13.5, color=DARK, fontweight='bold', zorder=6)

panel(ax, SB_X, TALL_Y, SB_W, TALL_H, fill='#e0f2fe', edge='#0284c7')
icon_people(ax, SB_CX, TR_CY + 0.80, s=1.0)
ax.text(SB_CX, TALL_Y + 0.75, 'Subject Block', ha='center', va='center',
        fontsize=13.0, color=DARK, fontweight='bold', zorder=6)

# encoder -> Transformer : right-angle elbows converging on the tall box
targets = {'Video': TR_CY + 1.30, 'Audio': TR_CY, 'Text': TR_CY - 1.30}
BUS_X = ENC_X + ENC_W + 0.95
for mod, ty in targets.items():
    cy = enc_centers[mod]
    elbow_arrow(ax, [(ENC_X + ENC_W + 0.10, cy), (BUS_X, cy), (BUS_X, ty),
                     (TR_X - 0.14, ty)])

# Transformer -> Subject Block
elbow_arrow(ax, [(TR_X + TR_W + 0.10, TR_CY), (SB_X - 0.14, TR_CY)])

# ── save ─────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, facecolor=WHITE, edgecolor='none',
            bbox_inches='tight', pad_inches=0.35)
print(f'Saved: {OUT}')
