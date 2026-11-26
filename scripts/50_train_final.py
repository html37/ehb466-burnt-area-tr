"""Final training of our models with the tuned hyper-parameters, 5 seeds each (FLOGA train only).

FLOGA val is used only for LightGBM early stopping (sampled val pixels) and for the per-seed decision threshold
(all val pixels, grid 0.05-0.95). Predictions on the FLOGA test split are written for evaluation.
Never reads Turkish data.

Outputs: outputs/models/<model>/seed<k>.pkl (+ meta json), outputs/predictions/floga_test/<model>_s<k>.npz,
         outputs/tables/cost_train.csv, outputs/runs/train_<model>/versions.json
Usage: python scripts/50_train_final.py [--models lgbm rf logreg] [--group ALL] [--tag ""]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir, ROOT  # noqa: E402
from burnt import floga, features as F, models as M, predictions as P, cost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
hpo = __import__("40_hpo")


def test_features(group, cfg):
    pre, post = [], []
    for _, p, m in floga.iter_split_pixels("test", cfg):
        pre.append(p["pre"][:, m].T); post.append(p["post"][:, m].T)
    return F.build(np.concatenate(pre), np.concatenate(post), group)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgbm", "rf", "logreg"])
    ap.add_argument("--group", default=None, help="feature group (default: from best_params file)")
    ap.add_argument("--params", default=None, help="best_params yaml (default configs/best_params_<model>.yaml)")
    ap.add_argument("--tag", default="", help="suffix for method name, e.g. _G1 for ablations")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    seeds = args.seeds or cfg["seeds"]
    rows = []
    for model in args.models:
        bp = yaml.safe_load(Path(args.params or ROOT / "configs" / f"best_params_{model}.yaml").read_text())
        group = args.group or bp["group"]
        name = f"{model}{args.tag}"
        Xtr, ytr, Xva_s, yva_s = hpo.load_train_val(group, cfg)
        Xvf, yvf, _ = hpo.load_val_full(group, cfg)
        Xte = test_features(group, cfg)
        mdir = out_dir("models", name)
        for seed in seeds:
            res = {"method": name, "seed": seed, "group": group, "n_features": Xtr.shape[1],
                   "n_train": len(ytr)}
            params = dict(bp["params"])
            with cost.measure(res, "train_"):
                clf = M.fit(model, params, Xtr, ytr, seed, Xva_s if model == "lgbm" else None,
                            yva_s if model == "lgbm" else None)
            thr, vf1 = M.best_threshold(M.predict_proba(clf, Xvf), yvf)
            with cost.measure(res, "test_infer_"):
                pt = M.predict_proba(clf, Xte)
            res["test_infer_s_per_mpx"] = res["test_infer_time_s"] / len(Xte) * 1e6
            P.save_floga_test(name, pt > thr, seed)
            f = mdir / f"seed{seed}.pkl"
            M.save(clf, f)
            res.update(threshold=thr, val_f1=vf1, model_mb=f.stat().st_size / 2 ** 20, **M.complexity(clf))
            (mdir / f"seed{seed}.json").write_text(json.dumps({"threshold": thr, "group": group, "val_f1": vf1,
                                                               "params": params}, indent=1, default=str))
            rows.append(res)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        cost.write_versions(out_dir("runs", f"train_{name}") / "versions.json")
    f = out_dir("tables") / "cost_train.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old[~old["method"].isin(new["method"].unique())]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)


if __name__ == "__main__":
    main()
