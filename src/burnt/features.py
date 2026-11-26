"""Per-pixel features for the lightweight classifiers, grouped for the ablation (CLAUDE.md Phase 5).

Input: pre/post DN arrays with bands last, shape (n, 9) in FLOGA order. Reflectance = min(DN, 10000) / 10000
(FLOGA `clamp_scale_10000`), identical for FLOGA and the Turkish scenes.

G1  dNBR
G2  dNBR, dNBR2, dNDVI, dNDWI, dBAIS2, dMIRBI            (differenced indices, dSI = SI_pre - SI_post)
G3  post-fire bands (9)
G4  pre-fire bands (9)
ALL G2 + G3 + G4 + single-date pre/post indices (12)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indices import all_indices, INDEX_NAMES

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "B8A"]
DIFF = [f"d{n}" for n in INDEX_NAMES]
G1 = ["dNBR"]
G2 = DIFF
G3 = [f"post_{b}" for b in BANDS]
G4 = [f"pre_{b}" for b in BANDS]
SINGLE = [f"pre_{n}" for n in INDEX_NAMES] + [f"post_{n}" for n in INDEX_NAMES]
GROUPS = {"G1": G1, "G2": G2, "G2+G3": G2 + G3, "ALL": G2 + G3 + G4 + SINGLE}


def reflect(dn):
    return np.minimum(np.asarray(dn, np.float32), 10000.0) / 10000.0


def build(pre_dn, post_dn, group="ALL") -> pd.DataFrame:
    """pre_dn, post_dn: (n, 9) arrays (DN). Returns float32 DataFrame with the columns of `group`."""
    pre, post = reflect(pre_dn), reflect(post_dn)
    cols = {}
    bp = {b: pre[:, i] for i, b in enumerate(BANDS)}
    bq = {b: post[:, i] for i, b in enumerate(BANDS)}
    ip, iq = all_indices(bp), all_indices(bq)
    for n in INDEX_NAMES:
        cols[f"d{n}"] = ip[n] - iq[n]
        cols[f"pre_{n}"] = ip[n]
        cols[f"post_{n}"] = iq[n]
    for b in BANDS:
        cols[f"pre_{b}"] = bp[b]
        cols[f"post_{b}"] = bq[b]
    names = GROUPS[group]
    df = pd.DataFrame({k: np.nan_to_num(cols[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                       for k in names})
    return df


def from_samples(df: pd.DataFrame, group="ALL") -> pd.DataFrame:
    pre = df[[f"pre_{b}" for b in BANDS]].to_numpy()
    post = df[[f"post_{b}" for b in BANDS]].to_numpy()
    return build(pre, post, group)


def from_stack(pre_stack, post_stack, mask, group="ALL") -> pd.DataFrame:
    """Channel-first stacks (9, H, W) + boolean mask -> features of masked pixels (row-major order)."""
    pre = pre_stack[:, mask].T
    post = post_stack[:, mask].T
    return build(pre, post, group)
