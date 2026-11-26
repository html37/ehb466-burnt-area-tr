"""dNBR benchmark: index, USGS thresholds, Otsu, grid search of a global threshold."""
from __future__ import annotations

import numpy as np

from .indices import nbr


def dnbr(pre_nir, pre_swir2, post_nir, post_swir2):
    return nbr(pre_nir, pre_swir2) - nbr(post_nir, post_swir2)


def otsu(values, bins=512, lo=-0.5, hi=1.3):
    v = np.clip(values[np.isfinite(values)], lo, hi)
    hist, edges = np.histogram(v, bins=bins)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m = np.cumsum(hist * c)
    mu0 = m / np.maximum(w0, 1); mu1 = (m[-1] - m) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def grid_counts(values, ref, thresholds):
    """Confusion counts for every threshold in one pass (pred = value > thr). values/ref: 1-D valid pixels."""
    values = np.asarray(values, np.float64); ref = np.asarray(ref, bool)
    thr = np.asarray(thresholds)
    # number of positives/negatives with value > thr, via sorted search
    pos = np.sort(values[ref]); neg = np.sort(values[~ref])
    tp = len(pos) - np.searchsorted(pos, thr, side="right")
    fp = len(neg) - np.searchsorted(neg, thr, side="right")
    fn = len(pos) - tp; tn = len(neg) - fp
    return np.stack([tp, fp, fn, tn], 1)
