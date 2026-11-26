"""Apply the trained FLOGA models (all seeds, frozen thresholds) to the Turkish test areas.

Inference only: models, features and thresholds were fixed on FLOGA; nothing is fitted or chosen here.
Also measures CPU inference cost (feature extraction + prediction) per 1000 km^2.

Outputs: outputs/predictions/turkey/<model>/<event>_s<k>.npz, outputs/tables/cost_infer_turkey.csv
Usage: python scripts/65_predict_turkey.py [--models lgbm rf logreg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir, ROOT  # noqa: E402
from burnt import turkey, features as F, models as M, predictions as P, cost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgbm", "rf", "logreg"])
    args = ap.parse_args()
    cfg = load_config()
    rows = []
    for name in args.models:
        mdir = ROOT / "outputs" / "models" / name
        for ev_name in turkey.event_names(cfg):
            ev = turkey.load_event(ev_name, cfg)
            valid = ev["valid"]
            area_km2 = valid.size * ev["px_ha"] / 100
            for f in sorted(mdir.glob("seed*.pkl")):
                seed = int(f.stem[4:])
                meta = json.loads(f.with_suffix(".json").read_text())
                clf = M.load(f)
                res = {"method": name, "event": ev_name, "seed": seed, "area_km2": area_km2, "device": "cpu"}
                with cost.measure(res, "infer_"):
                    X = F.from_stack(ev["pre"], ev["post"], valid, meta["group"])
                    prob_v = M.predict_proba(clf, X)
                res["infer_s_per_1000km2"] = res["infer_time_s"] / area_km2 * 1000
                prob = np.zeros(valid.shape, np.float32); prob[valid] = prob_v
                P.save_turkey(name, ev_name, prob > meta["threshold"], prob=prob, seed=seed)
                rows.append(res)
                print(res, flush=True)
    f = out_dir("tables") / "cost_infer_turkey.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old[~old["method"].isin(new["method"].unique())]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)


if __name__ == "__main__":
    main()
