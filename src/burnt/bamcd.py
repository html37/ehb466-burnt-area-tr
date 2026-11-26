"""BAM-CD benchmark (Sdraka et al. 2024): official model code (external/FLOGA, MIT) + official pretrained weights.

Preprocessing is the one used in FLOGA training (configs/config.json): bands B02,B03,B04,B05,B06,B07,B11,B12,B8A at
20 m, NaN/no data -> 0, `clamp_scale_10000` (min(DN, 10000) / 10000). Output: 2-class logits; burnt probability =
softmax[:, 1]; prediction = probability > 0.5 (identical to FLOGA's argmax).
Large rasters are processed with overlapping 256x256 windows; window outputs are blended with a 2-D triangular weight.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .config import ROOT

WEIGHTS = {"v1": ROOT / "data" / "raw" / "bamcd" / "v1" / "best_segmentation.pt",
           "v2": ROOT / "data" / "raw" / "bamcd" / "v2" / "best_segmentation.pt"}
SIAMESE = {"v1": True, "v2": False}  # FLOGA README / configs/method/bam_cd.json


def load(version="v1", device=None):
    sys.path.insert(0, str(ROOT / "external" / "FLOGA"))
    from models.bam_cd.model import BAM_CD  # noqa: E402
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BAM_CD(encoder_name="resnet101", encoder_weights=None, in_channels=9, classes=2, fusion_mode="conc",
                   activation=None, siamese=SIAMESE[version], decoder_attention_type="scse",
                   decoder_use_batchnorm=True)
    ck = torch.load(WEIGHTS[version], map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, device


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def scale(dn):
    x = np.nan_to_num(np.asarray(dn, np.float32), nan=0.0)
    return np.minimum(x, 10000.0) / 10000.0


@torch.no_grad()
def predict_patches(model, device, pre, post, batch=4):
    """pre/post: (N, 9, 256, 256) DN -> burnt probability (N, 256, 256)."""
    out = []
    for i in range(0, len(pre), batch):
        a = torch.from_numpy(scale(pre[i:i + batch])).to(device)
        b = torch.from_numpy(scale(post[i:i + batch])).to(device)
        out.append(torch.softmax(model(a, b), 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def _weight(size):
    w = 1 - np.abs(np.linspace(-1, 1, size)) * 0.98
    return np.outer(w, w).astype(np.float32)


@torch.no_grad()
def predict_raster(model, device, pre, post, size=256, stride=128, batch=4):
    """Sliding-window inference on (9, H, W) DN rasters -> probability (H, W)."""
    _, H, W = pre.shape
    Hp, Wp = max(H, size), max(W, size)
    rows = list(range(0, Hp - size + 1, stride)); cols = list(range(0, Wp - size + 1, stride))
    if rows[-1] != Hp - size: rows.append(Hp - size)
    if cols[-1] != Wp - size: cols.append(Wp - size)
    pad = lambda a: np.pad(a, ((0, 0), (0, Hp - H), (0, Wp - W)))
    pre, post = pad(pre), pad(post)
    acc = np.zeros((Hp, Wp), np.float32); wsum = np.zeros((Hp, Wp), np.float32)
    w = _weight(size)
    coords = [(r, c) for r in rows for c in cols]
    for i in range(0, len(coords), batch):
        cc = coords[i:i + batch]
        a = np.stack([pre[:, r:r + size, c:c + size] for r, c in cc])
        b = np.stack([post[:, r:r + size, c:c + size] for r, c in cc])
        p = predict_patches(model, device, a, b, batch)
        for (r, c), pp in zip(cc, p):
            acc[r:r + size, c:c + size] += pp * w
            wsum[r:r + size, c:c + size] += w
    return (acc / np.maximum(wsum, 1e-6))[:H, :W]
