"""Pixel metrics and spatial block bootstrap.

All metrics are computed for the burnt class on valid pixels only.
"""
from __future__ import annotations

import numpy as np


def confusion(pred, ref, valid=None):
    pred = np.asarray(pred, bool)
    ref = np.asarray(ref, bool)
    if valid is not None:
        valid = np.asarray(valid, bool)
        pred, ref = pred[valid], ref[valid]
    tp = int(np.sum(pred & ref)); fp = int(np.sum(pred & ~ref))
    fn = int(np.sum(~pred & ref)); tn = int(np.sum(~pred & ~ref))
    return tp, fp, fn, tn


def from_counts(tp, fp, fn, tn, px_ha=None):
    out = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "iou": tp / (tp + fp + fn) if tp else 0.0,
        "oa": (tp + tn) / max(tp + tn + fp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if px_ha is not None:
        out["pred_area_ha"] = (tp + fp) * px_ha
        out["ref_area_ha"] = (tp + fn) * px_ha
    return out


def metrics(pred, ref, valid=None, px_ha=None):
    return from_counts(*confusion(pred, ref, valid), px_ha=px_ha)


def block_ids(shape, res_m, block_m=1000):
    """Integer id of the block_m x block_m block each pixel belongs to."""
    k = max(int(round(block_m / res_m)), 1)
    rows = np.arange(shape[0]) // k
    cols = np.arange(shape[1]) // k
    ncols = cols.max() + 1
    return rows[:, None] * ncols + cols[None, :]


def block_counts(preds: dict, ref, valid, blocks):
    """Per-block confusion counts for several prediction maps sharing ref/valid/blocks.

    Returns (block_index array, dict name -> (n_blocks, 4) int64 array of tp, fp, fn, tn).
    """
    v = np.asarray(valid, bool)
    b = blocks[v]
    uniq, inv = np.unique(b, return_inverse=True)
    r = np.asarray(ref, bool)[v]
    out = {}
    for name, p in preds.items():
        p = np.asarray(p, bool)[v]
        cnt = np.zeros((len(uniq), 4), np.int64)
        for j, m in enumerate([p & r, p & ~r, ~p & r, ~p & ~r]):
            cnt[:, j] = np.bincount(inv, weights=m, minlength=len(uniq)).astype(np.int64)
        out[name] = cnt
    return uniq, out


def _f1_iou(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        iou = np.where(tp > 0, tp / (tp + fp + fn), 0.0)
    return f1, iou


def block_bootstrap(counts: dict, n_resamples=1000, seed=0, pairs=()):
    """Spatial block bootstrap: resample blocks with replacement (same draw for all methods -> paired).

    `counts` maps method -> (n_blocks, 4) counts (all on the same blocks). `pairs` = [(a, b), ...] for which a
    CI of F1(a) - F1(b) is returned. Several events can be pooled by concatenating their block arrays.
    Returns dict with per-method F1/IoU percentile CIs and per-pair delta-F1 CIs.
    """
    rng = np.random.default_rng(seed)
    nb = next(iter(counts.values())).shape[0]
    idx = rng.integers(0, nb, size=(n_resamples, nb))
    res = {}
    samples = {}
    for name, c in counts.items():
        tot = c[idx].sum(axis=1)  # (n_resamples, 4)
        f1, iou = _f1_iou(tot)
        samples[name] = f1
        res[name] = {"f1_lo": float(np.percentile(f1, 2.5)), "f1_hi": float(np.percentile(f1, 97.5)),
                     "iou_lo": float(np.percentile(iou, 2.5)), "iou_hi": float(np.percentile(iou, 97.5))}
    for a, b in pairs:
        d = samples[a] - samples[b]
        fa, _ = _f1_iou(counts[a].sum(0)); fb, _ = _f1_iou(counts[b].sum(0))
        res[f"{a} - {b}"] = {"delta_f1": float(fa - fb), "lo": float(np.percentile(d, 2.5)),
                             "hi": float(np.percentile(d, 97.5)), "p_le_0": float(np.mean(d <= 0))}
    return res
