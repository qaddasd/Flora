"""Generate 6 procedural template clips (mp4, H.264 + AAC) for the Flora web demo.

Each clip: 8 s, 24 fps, 480x360, distinct visuals + matching synthesized audio.
Thumbnails (middle frame) saved as JPEG.
"""

import math
from pathlib import Path

import numpy as np
import av
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VID_DIR = ROOT / "web-demo" / "videos"
VID_DIR.mkdir(parents=True, exist_ok=True)

W, H, FPS, DUR = 480, 360, 24, 8.0
SR = 44100
N_FRAMES = int(FPS * DUR)
rng = np.random.default_rng(7)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
XN, YN = XX / W, YY / H


# ── Visual generators (return HxWx3 uint8 frame at time t) ─────────────────

def frame_ocean(t):
    v = (np.sin(6 * XN + 2.2 * t) + np.sin(9 * YN - 1.7 * t)
         + np.sin(4 * (XN + YN) + t)) / 3
    foam = np.clip(np.sin(14 * YN - 3 * t + 2 * np.sin(5 * XN + t)), 0, 1) ** 6
    r = 20 + 25 * v + 120 * foam
    g = 60 + 60 * v + 160 * foam
    b = 110 + 80 * v + 200 * foam
    return np.dstack([r, g, b])


def frame_drive(t):
    base = np.full((H, W, 3), [8, 8, 14], dtype=np.float32)
    for i, (speed, y0, hue) in enumerate([(2.1, .32, (255, 190, 90)),
                                          (-1.6, .55, (255, 60, 40)),
                                          (2.8, .75, (180, 200, 255))]):
        xs = (np.linspace(0, 1, 14) + speed * t / 8 + i * 0.13) % 1.0
        for x in xs:
            d2 = (XN - x) ** 2 * 4 + (YN - y0 - 0.04 * np.sin(x * 20)) ** 2 * 30
            glow = np.exp(-d2 * 90)[..., None] * np.array(hue, dtype=np.float32)
            base += glow
    road = (YN > 0.6)[..., None] * np.array([14, 12, 16], dtype=np.float32)
    return base + road


def frame_forest(t):
    v = np.zeros((H, W), dtype=np.float32)
    for k in range(5):
        cx = 0.2 * k + 0.1 * np.sin(t * 0.6 + k)
        cy = 0.25 + 0.12 * k + 0.05 * np.cos(t * 0.8 + k * 2)
        d2 = (XN - cx) ** 2 + (YN - cy) ** 2
        v += np.exp(-d2 * (30 + 12 * k))
    v = v / v.max() * (0.75 + 0.25 * np.sin(t * 2))
    trunk = np.clip(1 - np.abs(XN - 0.3 - 0.002 * t) * 30, 0, 1) * 0.4
    r = 15 + 40 * v + 30 * trunk
    g = 45 + 120 * v + 45 * trunk
    b = 20 + 45 * v + 25 * trunk
    return np.dstack([r, g, b])


def frame_fire(t):
    h = np.clip(1.2 - YN * 1.3, 0, 1)
    flick = (np.sin(9 * XN + 7 * t) * np.sin(5 * XN - 5.3 * t)
             + 0.5 * np.sin(17 * XN + 11 * t)) * 0.25
    flame = np.clip(h + flick * h - 0.25, 0, 1) ** 1.6
    core = np.clip(flame * 1.8 - 0.5, 0, 1)
    r = 30 + 225 * flame
    g = 10 + 150 * flame ** 2 + 90 * core
    b = 5 + 60 * core ** 2
    return np.dstack([r, g, b])


def frame_rain(t):
    base = np.zeros((H, W), dtype=np.float32) + 12
    n = 130
    xs = (np.arange(n) * 0.6180339887) % 1.0
    spd = 0.55 + 0.4 * ((np.arange(n) * 0.37) % 1.0)
    ys = (spd * t + xs * 7) % 1.3 - 0.15
    for x, y in zip(xs, ys):
        col = int(x * (W - 1)); row = int(y * H)
        ln = int(10 + 14 * (x % 0.3))
        r0, r1 = max(0, row - ln), min(H, row)
        if r1 > r0:
            base[r0:r1, col] += 90
    glow = 25 * np.exp(-((XN - 0.7) ** 2 + (YN - 0.25) ** 2) * 8)
    r = base * 0.55 + glow * 0.5
    g = base * 0.7 + glow * 0.6
    b = base * 1.15 + glow
    return np.dstack([r, g, b])


def frame_nebula(t):
    v = np.zeros((H, W), dtype=np.float32)
    for k, (fx, fy, ph) in enumerate([(3, 2, 0), (5, 4, 2), (7, 3, 4)]):
        v += np.sin(fx * XN + fy * YN + t * (0.4 + 0.2 * k) + ph) / (k + 1.5)
    v = (v - v.min()) / (v.max() - v.min())
    r = 60 + 140 * np.clip(np.sin(v * 3.1 + t * 0.3), 0, 1)
    g = 30 + 110 * np.clip(np.sin(v * 3.1 + 2 + t * 0.25), 0, 1)
    b = 90 + 160 * np.clip(np.sin(v * 3.1 + 4 + t * 0.2), 0, 1)
    return np.dstack([r, g, b])


# ── Audio generators (mono float32, n samples) ──────────────────────────────

def _t_axis(n):
    return np.arange(n, dtype=np.float32) / SR


def audio_ocean(n):
    t = _t_axis(n)
    noise = rng.standard_normal(n).astype(np.float32)
    k = np.ones(400) / 400
    low = np.convolve(noise, k, mode="same")
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    return 0.5 * low * swell + 0.08 * np.sin(2 * np.pi * 55 * t)


def audio_drive(n):
    t = _t_axis(n)
    hum = 0.25 * np.sin(2 * np.pi * 82 * t) + 0.12 * np.sin(2 * np.pi * 164 * t)
    wob = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    noise = np.convolve(rng.standard_normal(n), np.ones(60) / 60, mode="same")
    return hum * wob + 0.12 * noise


def audio_forest(n):
    t = _t_axis(n)
    out = 0.05 * rng.standard_normal(n).astype(np.float32)
    for start in np.arange(0.3, DUR - 0.4, 0.9):
        i0 = int(start * SR); ln = int(0.25 * SR)
        if i0 + ln > n:
            break
        tt = np.arange(ln) / SR
        f0 = 2200 + 1400 * rng.random()
        chirp = np.sin(2 * np.pi * (f0 * tt + 900 * tt ** 2)) * np.exp(-tt * 14)
        out[i0:i0 + ln] += 0.6 * chirp
    return out


def audio_fire(n):
    noise = rng.standard_normal(n).astype(np.float32)
    low = np.convolve(noise, np.ones(120) / 120, mode="same")
    out = 0.35 * low
    for _ in range(60):
        i0 = int(rng.random() * (n - 300))
        ln = 120 + int(180 * rng.random())
        crack = rng.standard_normal(ln).astype(np.float32) * np.exp(-np.arange(ln) / 40)
        out[i0:i0 + ln] += 0.8 * crack
    return out


def audio_rain(n):
    noise = rng.standard_normal(n).astype(np.float32)
    k = np.exp(-np.arange(25) / 8)
    hiss = np.convolve(noise, k, mode="same")
    return 0.4 * hiss


def audio_nebula(n):
    t = _t_axis(n)
    pad = (np.sin(2 * np.pi * 110 * t) + 0.7 * np.sin(2 * np.pi * 165 * t)
           + 0.5 * np.sin(2 * np.pi * 220 * t + 1))
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.08 * t)
    shimmer = 0.15 * np.sin(2 * np.pi * 880 * t) * np.sin(2 * np.pi * 0.23 * t)
    return 0.22 * pad * lfo + shimmer


CLIPS = [
    ("ocean_waves",  "Ocean Waves",  frame_ocean,  audio_ocean),
    ("night_drive",  "Night Drive",  frame_drive,  audio_drive),
    ("forest_birds", "Forest Birds", frame_forest, audio_forest),
    ("fireplace",    "Fireplace",    frame_fire,   audio_fire),
    ("city_rain",    "City Rain",    frame_rain,   audio_rain),
    ("nebula",       "Nebula Drift", frame_nebula, audio_nebula),
]


def encode_clip(stem, frame_fn, audio_fn):
    out_path = VID_DIR / f"{stem}.mp4"
    container = av.open(str(out_path), mode="w")

    vstream = container.add_stream("libx264", rate=FPS)
    vstream.width, vstream.height = W, H
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}

    astream = container.add_stream("aac", rate=SR)
    astream.layout = "mono"

    thumb = None
    for i in range(N_FRAMES):
        t = i / FPS
        rgb = np.clip(frame_fn(t), 0, 255).astype(np.uint8)
        if i == N_FRAMES // 2:
            thumb = rgb.copy()
        vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for pkt in vstream.encode(vf):
            container.mux(pkt)

    audio = audio_fn(int(DUR * SR))
    audio = np.clip(audio, -1, 1).astype(np.float32)
    # feed in chunks
    chunk = 2048
    for s in range(0, len(audio), chunk):
        part = audio[s:s + chunk][None, :]  # (1, samples) mono
        af = av.AudioFrame.from_ndarray(part, format="flt", layout="mono")
        af.sample_rate = SR
        for pkt in astream.encode(af):
            container.mux(pkt)

    for pkt in vstream.encode():
        container.mux(pkt)
    for pkt in astream.encode():
        container.mux(pkt)
    container.close()

    Image.fromarray(thumb).save(VID_DIR / f"{stem}.jpg", quality=85)
    print(f"  {stem}.mp4  {out_path.stat().st_size/1e6:.2f} MB")


def main():
    print("Encoding template clips ...")
    try:
        av.codec.Codec("libx264", "w")
    except Exception:
        print("libx264 not available!")
        raise
    for stem, title, vf, af in CLIPS:
        encode_clip(stem, vf, af)
    print("Done.")


if __name__ == "__main__":
    main()
