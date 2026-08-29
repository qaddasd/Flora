"""Export Whisper-exact audio preprocessing assets for the browser demo.

whisper_dft.bin : float32[2 * 201 * 400]  -- real & imag rDFT matrices with the
                  Hann window folded in (re first, then imag). Per 400-sample
                  frame x: re = Wr @ x, im = Wi @ x, power = re^2 + im^2.
whisper_mel.bin : float32[80 * 201]       -- mel filterbank (filters @ power).
"""

from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web-demo" / "assets"

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
n_fft, n_mels = fe.n_fft, fe.feature_size          # 400, 80
n_bins = n_fft // 2 + 1                             # 201

# Window exactly as transformers does: np.hanning(n_fft + 1)[:-1]
window = np.hanning(n_fft + 1)[:-1].astype(np.float64)

n = np.arange(n_fft)
j = np.arange(n_bins)[:, None]
ang = 2 * np.pi * j * n / n_fft
Wr = (np.cos(ang) * window).astype("<f4")
Wi = (-np.sin(ang) * window).astype("<f4")

with open(OUT / "whisper_dft.bin", "wb") as f:
    Wr.tofile(f)
    Wi.tofile(f)

mel = np.asarray(fe.mel_filters, dtype="<f4")
if mel.shape == (n_bins, n_mels):                   # some versions store (201, 80)
    mel = mel.T.copy()
assert mel.shape == (n_mels, n_bins), mel.shape     # (80, 201)
mel.tofile(OUT / "whisper_mel.bin")

print(f"whisper_dft.bin: {(Wr.nbytes + Wi.nbytes)/1e3:.0f} KB  shape (2,201,400)")
print(f"whisper_mel.bin: {mel.nbytes/1e3:.0f} KB  shape (80,201)")

# Sanity: compare against the real feature extractor on noise
rng = np.random.default_rng(0)
audio = rng.standard_normal(16000 * 3).astype(np.float32)
ref = fe(audio, sampling_rate=16000, return_tensors="np")["input_features"][0]
print("ref mel shape:", ref.shape)

# Reproduce with matrices (center reflect padding like torch.stft/transformers)
x = np.pad(audio, (0, 480000 - len(audio)))
x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
n_frames = (len(x) - n_fft) // 160 + 1
frames = np.lib.stride_tricks.as_strided(
    x, shape=(n_frames, n_fft), strides=(160 * 4, 4)).copy()
re = frames @ Wr.T
im = frames @ Wi.T
power = re ** 2 + im ** 2
mel_spec = power @ mel.T
log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
mine = log_spec.T[:, :-1]  # drop last frame -> (80, 3000)
print("max|diff| vs transformers extractor:", np.abs(mine - ref).max())
