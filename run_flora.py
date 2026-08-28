# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import numpy as np
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flora.v3_model import FloraV3


def get_video_duration(mp4_path: Path) -> float:
    """Длительность видео в секундах."""
    container = av.open(str(mp4_path))
    stream = container.streams.video[0]
    dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
    container.close()
    return dur


def time_ticks(T: int, duration: float):
    """Метки времени в секундах/мм:сс для оси из T шагов."""
    times = np.linspace(0, duration, T)
    labels = [f'{int(t // 60)}:{t % 60:04.1f}' for t in times]
    n = min(T, 9)  # не больше 9 меток, чтобы не сливались
    idx = np.linspace(0, T - 1, n, dtype=int)
    return idx, [labels[i] for i in idx]


def extract_video_frames(mp4_path: Path, n_frames: int = 5) -> np.ndarray:
    container = av.open(str(mp4_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='rgb24'))
    container.close()
    
    if not frames:
        raise ValueError(f'No frames in {mp4_path}')
    
    total = len(frames)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return np.stack([frames[i] for i in indices])


@torch.no_grad()
def extract_video_features(frames_np: np.ndarray, device: torch.device) -> torch.Tensor:
    from transformers import MobileViTModel, MobileViTImageProcessor
    print('   [1/3] Извлечение видео-фичей (apple/mobilevit-small)...')
    processor = MobileViTImageProcessor.from_pretrained('apple/mobilevit-small')
    vmodel = MobileViTModel.from_pretrained('apple/mobilevit-small').to(device).eval()

    feats = []
    for i in range(len(frames_np)):
        inp = processor(images=frames_np[i], return_tensors='pt')
        pv = inp['pixel_values'].to(device)
        out = vmodel(pixel_values=pv)
        feats.append(out.pooler_output.squeeze(0).cpu())
    return torch.stack(feats)


@torch.no_grad()
def extract_audio_features(mp4_path: Path, device: torch.device, target_T: int = 5) -> torch.Tensor:
    from transformers import WhisperModel, WhisperFeatureExtractor
    print('   [2/3] Извлечение аудио-фичей (openai/whisper-tiny)...')
    fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-tiny')
    whisper = WhisperModel.from_pretrained('openai/whisper-tiny').to(device)
    encoder = whisper.encoder.eval()

    container = av.open(str(mp4_path))
    if not container.streams.audio:
        container.close()
        return torch.zeros(target_T, 384)

    resampler = av.AudioResampler(format='flt', layout='mono', rate=16000)
    audio_frames = []
    for frame in container.decode(audio=0):
        for rframe in resampler.resample(frame):
            audio_frames.append(rframe.to_ndarray())
    container.close()

    if not audio_frames:
        return torch.zeros(target_T, 384)

    audio_np = np.concatenate(audio_frames, axis=1).squeeze(0).astype(np.float32)
    inputs = fe(audio_np, sampling_rate=16000, return_tensors='pt')
    hidden = encoder(inputs['input_features'].to(device)).last_hidden_state.squeeze(0)

    feat = F.interpolate(
        hidden.T.unsqueeze(0), size=target_T, mode='linear', align_corners=False
    ).squeeze(0).T.cpu()
    return feat


NETWORK_NAMES = {
    'Vis': 'Зрительная',
    'SomMot': 'Соматомоторная',
    'DorsAttn': 'Дорсальное внимание',
    'VentAttn': 'Вентральное внимание',
    'Limbic': 'Лимбическая',
    'FrontPar': 'Фронтопариетальная',
    'Default': 'Пассивный режим (DMN)',
}


def parse_schaefer_labels(labels):
    """Метки атласа Шафера -> (индекс сети Йео, индекс полушария) для каждого ROI.

    Первая метка атласа — 'Background', её пропускаем (остальные соответствуют
    ROI 1..400, т.е. столбцам матрицы предсказания).
    """
    net_idx = np.zeros(len(labels), dtype=int)
    hemi_idx = np.zeros(len(labels), dtype=int)
    net_keys = list(NETWORK_NAMES)
    for i, lab in enumerate(labels):
        s = lab.decode() if isinstance(lab, bytes) else str(lab)
        parts = s.split('_')  # формат: 7Networks_LH_Vis_1 или 7Networks_RH_Default_pCunPCC_7
        if len(parts) < 3 or parts[1] not in ('LH', 'RH'):
            continue  # Background и прочие служебные метки
        net_idx[i] = net_keys.index(parts[2]) if parts[2] in net_keys else 0
        hemi_idx[i] = 0 if parts[1] == 'LH' else 1
    return net_idx, hemi_idx


def format_roi_name(label):
    """Человекочитаемое имя области: сеть, полушарие, подобласть/номер."""
    s = label.decode() if isinstance(label, bytes) else str(label)
    parts = s.split('_')
    net_name = NETWORK_NAMES.get(parts[2], parts[2])
    hemi = 'лев.' if parts[1] == 'LH' else 'прав.'
    if len(parts) >= 5:
        return f'{net_name} — {hemi}, {parts[3]} #{parts[4]}'
    return f'{net_name} — {hemi}, ROI {parts[3]}'


def plot_network_analysis(pred_np, net_idx, hemi_idx, out_dir, duration=0.0):
    """Динамика активности по 7 функциональным сетям Йео."""
    T = pred_np.shape[0]
    net_keys = list(NETWORK_NAMES)
    names = [NETWORK_NAMES[k] for k in net_keys]
    net_ts = np.zeros((T, len(net_keys)))
    for n in range(len(net_keys)):
        mask = net_idx == n
        if mask.any():
            net_ts[:, n] = pred_np[:, mask].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = plt.cm.tab10(np.linspace(0, 1, len(net_keys)))
    tick_idx, tick_labels = time_ticks(T, duration) if duration > 0 else (None, None)

    im = axes[0, 0].imshow(net_ts.T, aspect='auto', cmap='viridis')
    axes[0, 0].set_yticks(range(len(names)))
    axes[0, 0].set_yticklabels(names, fontsize=9)
    axes[0, 0].set_xlabel('Время в видео (мм:сс)' if duration > 0 else 'Временной интервал (TR)')
    if tick_idx is not None:
        axes[0, 0].set_xticks(tick_idx)
        axes[0, 0].set_xticklabels(tick_labels, fontsize=8)
    axes[0, 0].set_title('Активность функциональных сетей во времени', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0], label='Уровень активации')

    for n in range(len(net_keys)):
        axes[0, 1].plot(net_ts[:, n], marker='o', color=colors[n], label=names[n])
    axes[0, 1].set_title('Динамика сигнала по сетям', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Время в видео (мм:сс)' if duration > 0 else 'Временной интервал (TR)')
    if tick_idx is not None:
        axes[0, 1].set_xticks(tick_idx)
        axes[0, 1].set_xticklabels(tick_labels, fontsize=8)
    axes[0, 1].set_ylabel('Средний BOLD сигнал')
    axes[0, 1].legend(fontsize=8, loc='best')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    mean_net = net_ts.mean(axis=0)
    order = np.argsort(mean_net)
    axes[1, 0].barh([names[i] for i in order], mean_net[order], color=colors[order])
    axes[1, 0].set_title('Рейтинг сетей по средней активации', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Средняя активация за всё видео')

    net_hemi = np.zeros((len(net_keys), 2))
    for n in range(len(net_keys)):
        for h in range(2):
            mask = (net_idx == n) & (hemi_idx == h)
            if mask.any():
                net_hemi[n, h] = pred_np[:, mask].mean()
    x = np.arange(len(net_keys))
    w = 0.38
    axes[1, 1].bar(x - w / 2, net_hemi[:, 0], w, label='Левое полушарие', color='#4C72B0')
    axes[1, 1].bar(x + w / 2, net_hemi[:, 1], w, label='Правое полушарие', color='#DD8452')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(net_keys, rotation=30, ha='right', fontsize=8)
    axes[1, 1].set_title('Сравнение полушарий по сетям', fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=9)

    fig.suptitle('Анализ предсказания Flora по функциональным сетям (Йео-7)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / 'brain_networks_analysis.png'
    plt.savefig(str(path), dpi=200)
    plt.close()
    print(f'📈 Анализ по сетям сохранен: {path}')


def plot_top_regions_named(pred_np, labels, out_dir, top_n=15):
    """Топ самых активных областей с понятными названиями сетей."""
    mean_act = pred_np.mean(axis=0)
    top = np.argsort(mean_act)[::-1][:top_n]
    rows, vals = [], []
    for i in top:
        rows.append(format_roi_name(labels[i]))
        vals.append(mean_act[i])

    fig, ax = plt.subplots(figsize=(12, 7))
    ys = range(top_n)[::-1]
    bars = ax.barh(ys, vals[::-1], color=plt.cm.magma(np.linspace(0.3, 0.9, top_n)))
    ax.set_yticks(list(ys))
    ax.set_yticklabels(rows[::-1], fontsize=9)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f' {v:.3f}', va='center', fontsize=8)
    ax.set_title(f'Топ-{top_n} наиболее активных областей коры (средняя за видео)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Средняя предсказанная активация')
    plt.tight_layout()
    path = out_dir / 'brain_top_regions_named.png'
    plt.savefig(str(path), dpi=200)
    plt.close()
    print(f'🏆 Топ областей сохранен: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None)
    parser.add_argument('--ckpt', default='checkpoints/best-epoch=052-val/pearson_r=0.7278.ckpt')
    parser.add_argument('--n_frames', type=int, default=5)
    parser.add_argument('--out_dir', default='output')
    args = parser.parse_args()

    if args.video is None:
        files = list(Path('videos').glob('*.*'))
        mp4s = [f for f in files if f.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov']]
        if not mp4s:
            print('❌ В папке ./videos/ нет видеофайлов!')
            return
        video_path = mp4s[0]
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            # поддержка шаблонов вида videos\SHUKASHA*
            matches = list(Path('.').glob(args.video)) or list(Path(args.video).parent.glob(Path(args.video).name))
            if not matches:
                print(f'❌ Файл не найден: {args.video}')
                return
            video_path = matches[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)

    print('=' * 60)
    print('🧠 Flora Brain Encoding Model — Инференс видео')
    print('=' * 60)
    print(f'🎬 Видеофайл   : {video_path.name}')
    print(f'⏳ Длительность: {duration:.1f} сек ({duration/60:.2f} мин)')
    print(f'⚙️  Устройство  : {device}')
    print(f'⏱️  Шаги времени: {args.n_frames} (~{duration/args.n_frames:.1f} сек на шаг)')

    print('\n📦 Загрузка весов Flora...')
    model = FloraV3.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    print('\n🔍 Извлечение признаков:')
    frames = extract_video_frames(video_path, n_frames=args.n_frames)
    video_feat = extract_video_features(frames, device)
    audio_feat = extract_audio_features(video_path, device, target_T=args.n_frames)
    text_feat = torch.zeros(args.n_frames, 384)
    print('   [3/3] Текстовый канал: инициализирован (zeros/384)')

    print('\n🧪 Вычисление активности мозга (фМРТ)...')
    with torch.no_grad():
        t_in = text_feat.unsqueeze(0).to(device)
        a_in = audio_feat.unsqueeze(0).to(device)
        v_in = video_feat.unsqueeze(0).to(device)
        
        output = model(t_in, a_in, v_in)
        pred = output['prediction']
        if pred.shape[0] == 400 and pred.shape[1] == args.n_frames:
            pred = pred.T
        pred_np = pred.cpu().numpy()

    print(f'✅ Прогноз активности успешно получен!')
    print(f'   Размерность матрицы: {pred_np.shape} ({pred_np.shape[0]} временных интервалов × {pred_np.shape[1]} областей коры)')

    npy_path = out_dir / 'brain_activity.npy'
    np.save(str(npy_path), pred_np)
    print(f'💾 Данные сохранены: {npy_path}')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    im = axes[0].imshow(pred_np.T, aspect='auto', cmap='inferno')
    axes[0].set_title('fMRI BOLD Activity Prediction (400 Schaefer Atlas Cortical Regions)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Время в видео (мм:сс)')
    axes[0].set_ylabel('Область коры мозга (#1-400)')
    tick_idx, tick_labels = time_ticks(pred_np.shape[0], duration)
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=axes[0], label='Уровень активации')

    mean_act = np.mean(pred_np, axis=0)
    top_idx = np.argsort(mean_act)[-5:][::-1]
    for i in top_idx:
        axes[1].plot(pred_np[:, i], marker='o', label=f'Регион #{i} (mean={mean_act[i]:.3f})')
    axes[1].set_title('Топ-5 наиболее активных областей коры мозга', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Время в видео (мм:сс)')
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels(tick_labels, fontsize=8)
    axes[1].set_ylabel('Предсказанный BOLD сигнал')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = out_dir / 'brain_activation.png'
    plt.savefig(str(fig_path), dpi=200)
    plt.close()
    print(f'📊 График 2D сохранен: {fig_path}')

    # 5b. Детальный анализ по функциональным сетям Йео-7
    print('\n📈 Детальный анализ по функциональным сетям...')
    try:
        from nilearn import datasets
        atlas_info = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        roi_labels = atlas_info.labels[1:]  # без 'Background'
        net_idx, hemi_idx = parse_schaefer_labels(roi_labels)
        plot_network_analysis(pred_np, net_idx, hemi_idx, out_dir, duration=duration)
        plot_top_regions_named(pred_np, roi_labels, out_dir)
    except Exception as e:
        print('  [!] Ошибка сетевого анализа:', e)

    # 6. 3D Brain Visualizations
    print('\n🧠 Построение 3D анатомической модели мозга...')
    try:
        import nibabel as nib
        from nilearn import datasets, plotting, surface
        from nilearn.image import new_img_like

        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        atlas_img = nib.load(atlas.maps)
        atlas_data = atlas_img.get_fdata()

        pred_3d = np.zeros_like(atlas_data, dtype=np.float32)
        for i in range(1, 401):
            pred_3d[atlas_data == i] = mean_act[i - 1]

        stat_img = new_img_like(atlas_img, pred_3d)

        # Glass brain 3D
        fig, ax = plt.subplots(figsize=(14, 5))
        plotting.plot_glass_brain(
            stat_img, display_mode='lyrz', colorbar=True,
            title='Flora fMRI Prediction — 3D Glass Brain', cmap='hot', figure=fig
        )
        glass_path = out_dir / 'brain_3d_glass.png'
        fig.savefig(str(glass_path), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'🔮 3D Glass Brain сохранен: {glass_path}')

        # 3D Cortical Surface (4 anatomical views)
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
        texture_left = surface.vol_to_surf(stat_img, fsaverage.pial_left)
        texture_right = surface.vol_to_surf(stat_img, fsaverage.pial_right)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={'projection': '3d'})
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_left, hemi='left', view='lateral',
            bg_map=fsaverage.sulc_left, cmap='hot', axes=axes[0, 0],
            title='Левое полушарие — латеральный вид'
        )
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_left, hemi='left', view='medial',
            bg_map=fsaverage.sulc_left, cmap='hot', axes=axes[0, 1],
            title='Левое полушарие — медиальный вид'
        )
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_right, hemi='right', view='lateral',
            bg_map=fsaverage.sulc_right, cmap='hot', axes=axes[1, 0],
            title='Правое полушарие — латеральный вид'
        )
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_right, hemi='right', view='medial',
            bg_map=fsaverage.sulc_right, cmap='hot', axes=axes[1, 1],
            title='Правое полушарие — медиальный вид'
        )
        fig.suptitle('3D Модель коры головного мозга — Распределение активности (Flora)', fontsize=14, fontweight='bold', y=0.98)
        surf_path = out_dir / 'brain_3d_surface.png'
        fig.savefig(str(surf_path), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'🌐 3D Поверхность коры сохранена: {surf_path}')

        # Interactive HTML 3D brain
        view_l = plotting.view_surf(fsaverage.infl_left, texture_left, bg_map=fsaverage.sulc_left, cmap='hot', title='Flora 3D Brain (Left Hemishpere)')
        view_l.save_as_html(str(out_dir / 'interactive_3d_brain_left.html'))
        view_r = plotting.view_surf(fsaverage.infl_right, texture_right, bg_map=fsaverage.sulc_right, cmap='hot', title='Flora 3D Brain (Right Hemishpere)')
        view_r.save_as_html(str(out_dir / 'interactive_3d_brain_right.html'))
        print(f'🖱️ Интерактивная 3D Web-модель сохранена в: {out_dir / "interactive_3d_brain_*.html"}')

    except Exception as e:
        print('  [!] Ошибка при генерации 3D анатомических видов:', e)

    print('\n✅ Полный цикл инференса и 3D визуализации завершен!')


if __name__ == '__main__':
    main()
