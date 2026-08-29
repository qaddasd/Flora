"""Export Flora models to ONNX for the WebGPU browser demo.

Exports:
  1. fusion_web.onnx        FloraV3 (export-friendly dense-MoE)  (1,T,384)+(1,T,384)+(1,T,640) -> (1,T,400)
  2. video_encoder_web.onnx MobileViT-S pooled frame encoder    (N,3,256,256) -> (N,640)
  3. audio_encoder_web.onnx Whisper-Tiny encoder                (1,80,3000)   -> (1,1500,384)

Then converts all to FP16 (*_fp16.onnx) for fast WebGPU execution.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flora.v3_model import FloraV3  # noqa: E402

OUT = ROOT / "web-demo" / "models"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "checkpoints" / "best-epoch=052-val" / "pearson_r=0.7278.ckpt"


# ── Export-friendly FloraV3 ─────────────────────────────────────────────────

class FloraV3Export(nn.Module):
    """Dense-MoE reimplementation of FloraV3.forward — numerically identical
    in eval mode, but uses only static, ONNX-friendly ops."""

    def __init__(self, m: FloraV3, max_trs: int = 32):
        super().__init__()
        # Slim the demo model: browser clips never exceed max_trs TRs.
        with torch.no_grad():
            m.text_time_embed.weight = nn.Parameter(m.text_time_embed.weight[:max_trs].clone())
            m.audio_time_embed.weight = nn.Parameter(m.audio_time_embed.weight[:max_trs].clone())
            m.video_time_embed.weight = nn.Parameter(m.video_time_embed.weight[:max_trs].clone())
            m.pos_embed = nn.Parameter(m.pos_embed[:, : max_trs * 3].clone())
        self.m = m

    def _moe_dense(self, layer, x):
        """layer.moe computed densely: all experts + masked-renormalized softmax."""
        moe = layer.moe
        logits = moe.router.gate(x)                      # (B,T,E)
        top_val, top_idx = logits.topk(moe.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        probs = torch.softmax(logits, dim=-1) * mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # (B,T,E)

        out = torch.zeros_like(x)
        for e in range(moe.num_experts):
            h = F.gelu(x @ moe.w1[e] + moe.b1[e])        # (B,T,F)
            y = h @ moe.w2[e] + moe.b2[e]                # (B,T,D)
            out = out + probs[:, :, e].unsqueeze(-1) * y
        return out

    def _attention(self, attn, x, attn_bias):
        """Manual MHA — numerically identical to nn.MultiheadAttention (eval)."""
        B, T3, D = x.shape
        H = attn.num_heads
        hd = D // H
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.split(D, dim=-1)
        q = q.reshape(B, T3, H, hd).transpose(1, 2)
        k = k.reshape(B, T3, H, hd).transpose(1, 2)
        v = v.reshape(B, T3, H, hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)
        w = torch.softmax(scores, dim=-1)
        o = (w @ v).transpose(1, 2).reshape(B, T3, D)
        return F.linear(o, attn.out_proj.weight, attn.out_proj.bias)

    def _block(self, layer, x, use_hrf_bias):
        B, T3, D = x.shape
        normed = layer.norm1(x)

        attn_bias = None
        if use_hrf_bias:
            pos = torch.arange(T3, device=x.device, dtype=torch.float32)
            tr = torch.floor(pos / 3.0)
            dist = (tr.unsqueeze(0) - tr.unsqueeze(1)).abs()
            attn_bias = -layer.log_alpha.exp() * dist   # (T3, T3)

        x = x + self._attention(layer.attn, normed, attn_bias)
        x = x + self._moe_dense(layer, layer.norm2(x))
        return x

    def forward(self, text_feat, audio_feat, video_feat, subject_id):
        m = self.m
        B, T, _ = text_feat.shape

        tp = m.text_proj(text_feat)
        ap = m.audio_proj(audio_feat)
        vp = m.video_proj(video_feat)
        vp = m.video_motion(vp)

        t_idx = torch.arange(T, device=text_feat.device)
        tp = tp + m.text_time_embed(t_idx) + m.modality_embed.weight[0]
        ap = ap + m.audio_time_embed(t_idx) + m.modality_embed.weight[1]
        vp = vp + m.video_time_embed(t_idx) + m.modality_embed.weight[2]

        # Interleave [t1,a1,v1,...]
        x = torch.stack([tp, ap, vp], dim=2).reshape(B, T * 3, m.hidden_dim)
        x = x + m.pos_embed[:, : T * 3]

        for i, layer in enumerate(m.layers):
            x = self._block(layer, x, use_hrf_bias=(i < 2))

        x = m.norm(x)

        # Gated modality pooling
        tokens = x.reshape(B, T, 3, m.hidden_dim)
        pool = tokens.mean(dim=2)
        gates = torch.softmax(m.gate_pool.gate_net(pool), dim=-1)
        fused = (gates.unsqueeze(-1) * tokens).sum(dim=2)   # (B,T,D)

        # HRF conv (causal depthwise)
        fused = m.hrf_conv(fused)

        out = F.gelu(m.output_mlp(m.output_norm(fused)))
        g = m.film.gamma(subject_id).unsqueeze(1)
        b = m.film.beta(subject_id).unsqueeze(1)
        out = g * out + b
        parcels = m.vertex_proj(out)                        # (B,T,400)
        return parcels


class VideoEncoderExport(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.pooler_output   # (N, 640)


class AudioEncoderExport(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state  # (1,1500,384)


def verify(module, args, onnx_path, atol=2e-3):
    import onnxruntime as ort
    with torch.no_grad():
        ref = module(*args).cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {n: a.cpu().numpy() for n, a in zip(names, args)}
    got = sess.run(None, feed)[0]
    err = np.abs(ref - got).max()
    status = "OK " if err < atol else "WARN"
    print(f"  [{status}] max|diff| = {err:.2e}  shape={got.shape}")
    return err


def to_fp16(path: Path):
    # keep_io_types=False: full-fp16 graph I/O. With keep_io_types=True the
    # boundary Cast nodes produced an inconsistent graph that ORT WebGPU
    # rejects ("output arg ... does not match expected type"). The web demo
    # feeds float16 tensors directly (js/util.js f32ToF16).
    from onnxconverter_common import float16
    import onnx
    model = onnx.load(str(path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    out = path.with_name(path.stem + "_fp16.onnx")
    onnx.save(model_fp16, str(out))
    print(f"  fp16 -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")


def main():
    dev = "cpu"
    print("== 1. FloraV3 fusion ==")
    model = FloraV3.load_from_checkpoint(str(CKPT), map_location=dev)
    model.eval()
    wrapper = FloraV3Export(model).eval()

    T = 5
    d_text = torch.randn(1, T, 384)
    d_audio = torch.randn(1, T, 384)
    d_video = torch.randn(1, T, 640)
    d_subj = torch.zeros(1, dtype=torch.long)

    # sanity vs original model
    with torch.no_grad():
        ref = model(d_text, d_audio, d_video, d_subj)["prediction"]
        mine = wrapper(d_text, d_audio, d_video, d_subj)
    print(f"  wrapper vs model max|diff| = {(ref-mine).abs().max().item():.2e}")

    fusion_path = OUT / "fusion_web.onnx"
    torch.onnx.export(
        wrapper, (d_text, d_audio, d_video, d_subj), str(fusion_path),
        input_names=["text_features", "audio_features", "video_features", "subject_id"],
        output_names=["parcels"],
        dynamic_axes={
            "text_features": {1: "time"}, "audio_features": {1: "time"},
            "video_features": {1: "time"}, "parcels": {1: "time"},
        },
        opset_version=17,
        external_data=False,
    )
    print(f"  saved {fusion_path.name} ({fusion_path.stat().st_size/1e6:.1f} MB)")
    verify(wrapper, (d_text, d_audio, d_video, d_subj), fusion_path)

    print("== 2. MobileViT-S video encoder ==")
    from transformers import MobileViTModel
    vm = MobileViTModel.from_pretrained("apple/mobilevit-small").eval()
    vw = VideoEncoderExport(vm)
    d_frames = torch.rand(3, 3, 256, 256)
    video_path = OUT / "video_encoder_web.onnx"
    torch.onnx.export(
        vw, (d_frames,), str(video_path),
        input_names=["pixel_values"], output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "n_frames"}, "embeddings": {0: "n_frames"}},
        opset_version=17,
        external_data=False,
    )
    print(f"  saved {video_path.name} ({video_path.stat().st_size/1e6:.1f} MB)")
    verify(vw, (d_frames,), video_path, atol=5e-3)

    print("== 3. Whisper-Tiny audio encoder ==")
    from transformers import WhisperModel
    wenc = WhisperModel.from_pretrained("openai/whisper-tiny").encoder.eval()
    aw = AudioEncoderExport(wenc)
    d_mel = torch.randn(1, 80, 3000)
    audio_path = OUT / "audio_encoder_web.onnx"
    torch.onnx.export(
        aw, (d_mel,), str(audio_path),
        input_names=["input_features"], output_names=["embeddings"],
        opset_version=17,
        external_data=False,
    )
    print(f"  saved {audio_path.name} ({audio_path.stat().st_size/1e6:.1f} MB)")
    verify(aw, (d_mel,), audio_path, atol=5e-3)

    print("== 4. FP16 conversion ==")
    try:
        for p in [fusion_path, video_path, audio_path]:
            to_fp16(p)
    except ImportError:
        print("  onnxconverter-common missing; skipping fp16 (pip install onnxconverter-common)")

    print("Done.")


if __name__ == "__main__":
    main()
