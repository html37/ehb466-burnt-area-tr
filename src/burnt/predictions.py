"""Storage convention for predictions, shared by all methods so that evaluation is like-for-like.

Turkey:      outputs/predictions/turkey/<method>/<event>[_s<seed>].npz   keys: pred (uint8 0/1), prob (float16, opt.)
FLOGA test:  outputs/predictions/floga_test/<method>[_s<seed>].npz       keys: pred (packed bits over the canonical
             list of valid FLOGA test pixels, see burnt.floga.test_pixels), n (number of pixels)
"""
from __future__ import annotations

import re

import numpy as np

from .config import ROOT

PRED = ROOT / "outputs" / "predictions"


def save_turkey(method, event, pred, prob=None, seed=None):
    d = PRED / "turkey" / method
    d.mkdir(parents=True, exist_ok=True)
    arrs = {"pred": np.asarray(pred, np.uint8)}
    if prob is not None:
        arrs["prob"] = np.asarray(prob, np.float16)
    np.savez_compressed(d / (f"{event}_s{seed}.npz" if seed is not None else f"{event}.npz"), **arrs)


def load_turkey(method, event):
    """Return {seed or None: pred(bool)} for one method/event."""
    d = PRED / "turkey" / method
    out = {}
    for f in sorted(d.glob(f"{event}*.npz")):
        m = re.fullmatch(rf"{re.escape(event)}(?:_s(\d+))?\.npz", f.name)
        if m:
            out[int(m.group(1)) if m.group(1) else None] = np.load(f)["pred"].astype(bool)
    return out


def load_turkey_prob(method, event, seed=None):
    d = PRED / "turkey" / method
    f = d / (f"{event}_s{seed}.npz" if seed is not None else f"{event}.npz")
    z = np.load(f)
    return z["prob"].astype(np.float32) if "prob" in z.files else z["pred"].astype(np.float32)


def save_floga_test(method, pred, seed=None):
    d = PRED / "floga_test"
    d.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(pred, bool)
    np.savez_compressed(d / (f"{method}_s{seed}.npz" if seed is not None else f"{method}.npz"),
                        pred=np.packbits(pred), n=len(pred))


def load_floga_test(method):
    d = PRED / "floga_test"
    out = {}
    for f in sorted(d.glob(f"{method}*.npz")):
        m = re.fullmatch(rf"{re.escape(method)}(?:_s(\d+))?\.npz", f.name)
        if m:
            z = np.load(f)
            out[int(m.group(1)) if m.group(1) else None] = np.unpackbits(z["pred"])[: int(z["n"])].astype(bool)
    return out


def methods_available(split="turkey"):
    d = PRED / split
    if not d.exists():
        return []
    if split == "turkey":
        return sorted(p.name for p in d.iterdir() if p.is_dir())
    return sorted({re.sub(r"_s\d+$", "", f.stem) for f in d.glob("*.npz")})
