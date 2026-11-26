"""Access to the cached Turkish test set (built by scripts/10_build_turkey_testset.py).

Only scripts/70_evaluate.py, 80_ablation.py, 90_error_analysis.py, 75_eval_checklabels.py and the demo may import
this module (test isolation, CLAUDE.md rule 2.1; enforced by tests/test_no_leakage.py).
"""
from __future__ import annotations

import json

import numpy as np

from .config import load_config


def load_event(name, cfg=None):
    cfg = cfg or load_config()
    d = cfg["paths"]["cache"] / "turkey"
    z = np.load(d / f"{name}.npz")
    meta = json.loads((d / f"{name}.json").read_text())
    ev = {k: z[k] for k in z.files}
    ev["meta"] = meta
    ev["px_ha"] = meta["res_m"] ** 2 / 1e4
    return ev


def reflectance(stack):
    """uint16 DN (0 = no data) -> float32 reflectance with NaN for no data."""
    r = stack.astype(np.float32) / 10000.0
    r[stack == 0] = np.nan
    return r


def event_names(cfg=None):
    return list((cfg or load_config())["events"].keys())
