"""dNBR benchmark on FLOGA: global threshold tuned on the FLOGA *validation* split, applied unchanged elsewhere.

dNBR = NBR_pre - NBR_post with NBR = (B8A - B12) / (B8A + B12) (FLOGA has no B08 at 20 m; same definition as the
Turkish pipeline). Grid -0.20..1.00 step 0.01, criterion = F1 of the burnt class over all valid val pixels.
Also writes FLOGA-test predictions for the USGS thresholds (0.10, 0.27) and the tuned threshold.
Never reads Turkish data.

Outputs: outputs/tables/dnbr_floga_threshold.json, outputs/tables/dnbr_val_curve.csv,
         outputs/predictions/floga_test/dnbr_*.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir  # noqa: E402
from burnt import floga, predictions as P  # noqa: E402
from burnt.dnbr import dnbr, grid_counts  # noqa: E402
from burnt.metrics import from_counts  # noqa: E402

B = floga.FLOGA_BANDS
I8, I12 = B.index("B8A"), B.index("B12")


def split_dnbr(split, cfg):
    vals, refs = [], []
    for _, p, m in floga.iter_split_pixels(split, cfg):
        pre = p["pre"].astype(np.float32) / 10000; post = p["post"].astype(np.float32) / 10000
        d = dnbr(pre[I8], pre[I12], post[I8], post[I12])
        vals.append(np.nan_to_num(d[m])); refs.append(p["label"][m] == 1)
    return np.concatenate(vals), np.concatenate(refs)


def main():
    cfg = load_config()
    lo, hi, st = cfg["dnbr"]["grid"]
    grid = np.round(np.arange(lo, hi + st / 2, st), 2)
    v, r = split_dnbr("val", cfg)
    gc = grid_counts(v, r, grid)
    curve = pd.DataFrame([{"threshold": t, **from_counts(*map(int, c))} for t, c in zip(grid, gc)])
    curve.to_csv(out_dir("tables") / "dnbr_val_curve.csv", index=False)
    best = curve.loc[curve["f1"].idxmax()]
    res = {"threshold": float(best["threshold"]), "val_f1": float(best["f1"]), "val_iou": float(best["iou"]),
           "val_pixels": int(len(v)), "val_burnt_pixels": int(r.sum()), "grid": [lo, hi, st],
           "band_definition": "NBR=(B8A-B12)/(B8A+B12)"}
    (out_dir("tables") / "dnbr_floga_threshold.json").write_text(json.dumps(res, indent=1))
    print("tuned on FLOGA val:", res)
    t, rt = split_dnbr("test", cfg)
    thr = dict(cfg["dnbr"]["usgs_thresholds"]); thr["dnbr_floga"] = res["threshold"]
    for m, th in thr.items():
        P.save_floga_test(m, t > th)


if __name__ == "__main__":
    main()
