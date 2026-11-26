"""Evaluate all methods (and the EFFIS reference itself) on the team's independent check labels, if filled.

Reads labeling/points.csv (column `label` 0/1 filled by a human) + labeling/points_key.csv. If no label is filled
the script says so and exits without output (the paper then lists check labelling as future work).
Points come from a stratified design (inside EFFIS / 0-1 km / 1-3 km); metrics are reported unweighted and per
stratum. Output: outputs/tables/checklabels_eval.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir, ROOT  # noqa: E402
from burnt import turkey, metrics, predictions as P  # noqa: E402


def main():
    cfg = load_config()
    pts = pd.read_csv(ROOT / "labeling" / "points.csv")
    pts = pts.merge(pd.read_csv(ROOT / "labeling" / "points_key.csv"), on=["point_id", "event"])
    pts = pts[pd.to_numeric(pts["label"], errors="coerce").isin([0, 1])].copy()
    if pts.empty:
        print("No check labels filled in labeling/points.csv - nothing to evaluate (see labeling/NASIL_ETIKETLENIR.md).")
        return
    pts["label"] = pts["label"].astype(int).astype(bool)
    rows = []
    for name, g in pts.groupby("event"):
        ev = turkey.load_event(name, cfg)
        rr, cc = g["row"].to_numpy(), g["col"].to_numpy()
        cand = {"EFFIS (reference)": {None: ev["effis"]}}
        for m in P.methods_available("turkey"):
            cand[m] = P.load_turkey(m, name)
        for m, preds in cand.items():
            for seed, pr in preds.items():
                pv = pr[rr, cc]
                rows.append({"event": name, "method": m, "seed": seed, "n_points": len(g),
                             **metrics.metrics(pv, g["label"].to_numpy())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir("tables") / "checklabels_eval.csv", index=False)
    print(df.groupby(["method", "event"])[["precision", "recall", "f1", "oa"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
