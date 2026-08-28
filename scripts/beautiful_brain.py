# -*- coding: utf-8 -*-
"""Красивые рендеры мозга в стиле: чёрный фон + hot-колормап.

Читает output/brain_activity.npy (результат run_flora.py) и строит:
  - beautiful_lateral.png   — латеральные виды обоих полушарий
  - beautiful_medial.png    — медиальные виды
  - beautiful_inflated.png  — инфлированная (развёрнутая) кора
  - beautiful_brain_3d.html — интерактивная 3D-модель (plotly, тёмная тема)
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output'


def load_stat_map():
    """Средняя за видео активация, спроецированная в объём MNI."""
    import nibabel as nib
    from nilearn import datasets
    from nilearn.image import new_img_like

    npy = OUT / 'brain_activity.npy'
    if not npy.exists():
        print('❌ Сначала запусти run_flora.py, чтобы получить brain_activity.npy')
        sys.exit(1)
    pred = np.load(npy)                      # (T, 400)
    mean_act = pred.mean(axis=0)

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    pred_3d = np.zeros_like(atlas_data, dtype=np.float32)
    for i in range(1, 401):
        pred_3d[atlas_data == i] = mean_act[i - 1]
    return new_img_like(atlas_img, pred_3d)


def render_matplotlib_views(stat_img, fsaverage):
    """Статичные PNG : чёрный фон, cmap hot."""
    from nilearn import surface, plotting

    tex_l = surface.vol_to_surf(stat_img, fsaverage.pial_left, radius=6.0)
    tex_r = surface.vol_to_surf(stat_img, fsaverage.pial_right, radius=6.0)
    # Контраст : тёмная база, яркие хотспоты
    vmax = float(np.percentile(np.concatenate([tex_l, tex_r]), 99.5))

    plt.style.use('dark_background')

    views = [
        ('lateral',  fsaverage.pial_left,  fsaverage.pial_right,  'beautiful_lateral.png',  'Латеральный вид'),
        ('medial',   fsaverage.pial_left,  fsaverage.pial_right,  'beautiful_medial.png',   'Медиальный вид'),
        ('lateral',  fsaverage.infl_left,  fsaverage.infl_right,  'beautiful_inflated.png', 'Инфлированная кора'),
    ]

    for view, surf_l, surf_r, fname, title in views:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': '3d'})
        fig.patch.set_facecolor('black')
        for ax in axes:
            ax.set_facecolor('black')

        plotting.plot_surf_stat_map(
            surf_l, tex_l, hemi='left', view=view,
            bg_map=fsaverage.sulc_left, cmap='hot', vmin=0, vmax=vmax,
            colorbar=False, axes=axes[0],
            title='Левое полушарие',
        )
        plotting.plot_surf_stat_map(
            surf_r, tex_r, hemi='right', view=view,
            bg_map=fsaverage.sulc_right, cmap='hot', vmin=0, vmax=vmax,
            colorbar=False, axes=axes[1],
            title='Правое полушарие',
        )

        sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, vmax))
        cax = fig.add_axes([0.93, 0.3, 0.012, 0.4])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Активность', color='white', fontsize=11)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.spines.values(), color='white')

        fig.suptitle(f'Flora — предсказанная активность мозга · {title}',
                     color='white', fontsize=15, fontweight='bold', y=0.95)
        fig.subplots_adjust(left=0.02, right=0.91, top=0.88, bottom=0.02, wspace=0.02)
        path = OUT / fname
        fig.savefig(str(path), dpi=250, facecolor='black')
        plt.close(fig)
        print(f'🎨 Сохранено: {path}')


def render_plotly_html(stat_img, fsaverage):
    """Интерактивный 3D-мозг: тёмные кнопки и открытие «как книга»."""
    import plotly.graph_objects as go
    from nilearn import surface
    from nilearn.surface import load_surf_mesh

    mesh_l = load_surf_mesh(fsaverage.pial_left)
    mesh_r = load_surf_mesh(fsaverage.pial_right)
    tex_l = surface.vol_to_surf(stat_img, fsaverage.pial_left, radius=6.0)
    tex_r = surface.vol_to_surf(stat_img, fsaverage.pial_right, radius=6.0)

    (xl, yl, zl), fl = mesh_l[0].T, mesh_l[1]
    (xr, yr, zr), fr = mesh_r[0].T, mesh_r[1]
    vmax = float(np.percentile(np.concatenate([tex_l, tex_r]), 99.5))

    def hemi_trace(x, y, z, faces, intensity, name):
        i, j, k = faces.T
        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=intensity, colorscale='hot', cmin=0, cmax=vmax,
            lighting=dict(ambient=0.45, diffuse=0.7, specular=0.25, roughness=0.6),
            lightposition=dict(x=100, y=-100, z=80),
            opacity=1.0, showscale=False, name=name,
        )

    fig = go.Figure(data=[
        hemi_trace(xl, yl, zl, fl, tex_l, 'Левое полушарие'),
        hemi_trace(xr, yr, zr, fr, tex_r, 'Правое полушарие'),
    ])
    fig.update_layout(
        paper_bgcolor='black', plot_bgcolor='black',
        scene=dict(
            bgcolor='black',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=0, z=0.2)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )

    div_html = fig.to_html(full_html=False, include_plotlyjs=False,
                           div_id='floraBrain',
                           config={'displayModeBar': False, 'responsive': True})

    page = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Flora — 3D-мозг</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; background: #000; overflow: hidden;
               font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }
  #floraBrain { width: 100vw; height: 100vh; }
  .hud { position: fixed; left: 50%; transform: translateX(-50%);
         display: flex; gap: 10px; z-index: 10; }
  .hud.top    { top: 18px; }
  .hud.bottom { bottom: 22px; }
  .title { color: #fff; font-size: 17px; font-weight: 700; letter-spacing: .3px;
           text-shadow: 0 1px 8px #000; }
  .btn { background: rgba(17,17,17,.85); color: #eee; border: 1px solid #555;
         border-radius: 999px; padding: 10px 26px; font-size: 14px; font-weight: 600;
         cursor: pointer; transition: all .2s; backdrop-filter: blur(4px); }
  .btn:hover { border-color: #fff; color: #fff; }
  .btn.active { background: #fff; color: #000; border-color: #fff; }
</style>
</head>
<body>
<div class="hud top"><span class="title">Flora — предсказанная активность мозга</span></div>
__DIV__
<div class="hud bottom">
  <button class="btn active" id="btnClose">Закрыто</button>
  <button class="btn" id="btnOpen">Открыто · как книга</button>
</div>
<script>
const gd = document.getElementById('floraBrain');
let base = null, angle = 0, raf = null;

function init() {
  if (base) return;
  // plotly может хранить координаты как binary-объекты — берём _inputArray
  base = gd.data.map(t => ({
    x: Array.from(t.x._inputArray || t.x),
    y: Array.from(t.y._inputArray || t.y),
    z: Array.from(t.z._inputArray || t.z),
  }));
}
init();  // встроенный Plotly.newPlot уже отработал к этому моменту

function apply(a) {
  // раскрытие «книжное»: каждое полушарие поворачивается на a вокруг
  // вертикальной оси и уезжает в свою сторону, медиальной поверхностью к камере
  const r = a * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
  const shift = 80 * (a / 90);
  const L = base[0], R = base[1];
  const xl = new Array(L.x.length), yl = new Array(L.x.length);
  for (let i = 0; i < L.x.length; i++) {          // левое: -a, влево
    xl[i] =  L.x[i] * c + L.y[i] * s - shift;
    yl[i] = -L.x[i] * s + L.y[i] * c;
  }
  const xr = new Array(R.x.length), yr = new Array(R.x.length);
  for (let i = 0; i < R.x.length; i++) {          // правое: +a, вправо
    xr[i] = R.x[i] * c - R.y[i] * s + shift;
    yr[i] = R.x[i] * s + R.y[i] * c;
  }
  Plotly.restyle(gd, { x: [xl, xr], y: [yl, yr] });
}

function animateTo(target) {
  if (raf) cancelAnimationFrame(raf);
  const from = angle, t0 = performance.now(), dur = 900;
  const ease = t => t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3)/2;
  function step(now) {
    const p = Math.min(1, (now - t0) / dur);
    angle = from + (target - from) * ease(p);
    apply(angle);
    if (p < 1) raf = requestAnimationFrame(step);
  }
  raf = requestAnimationFrame(step);
  // камера: при открытии — вид спереди, чтобы видеть обе медиальные поверхности
  const cam = target > 0 ? { eye: { x: 0, y: -2.8, z: 0.6 } }
                         : { eye: { x: 1.8, y: 0, z: 0.2 } };
  Plotly.relayout(gd, { 'scene.camera': cam });
}

document.getElementById('btnOpen').onclick = e => {
  e.target.classList.add('active');
  document.getElementById('btnClose').classList.remove('active');
  animateTo(90);   // полушария раскрываются в обе стороны, 
};
document.getElementById('btnClose').onclick = e => {
  e.target.classList.add('active');
  document.getElementById('btnOpen').classList.remove('active');
  animateTo(0);
};
</script>
</body>
</html>"""

    page = page.replace('__DIV__', div_html)
    path = OUT / 'beautiful_brain_3d.html'
    path.write_text(page, encoding='utf-8')
    print(f'🌐 Интерактивный 3D сохранен: {path} (открой в браузере, вращай мышью)')


def main():
    from nilearn import datasets
    stat_img = load_stat_map()
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    print('🎨 Рендер красивых видов мозга (стиль публикации)...')
    render_matplotlib_views(stat_img, fsaverage)
    try:
        render_plotly_html(stat_img, fsaverage)
    except Exception as e:
        print('  [!] Ошибка plotly-рендера:', e)
    print('✅ Готово!')


if __name__ == '__main__':
    main()

