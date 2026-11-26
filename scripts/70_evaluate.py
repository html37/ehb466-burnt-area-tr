"""Final evaluation of all methods on the identical grids (FLOGA test split, Manavgat, Marmaris, Türkiye pooled).

Every method is scored with the same valid mask and EFFIS reference (Türkiye) or the same FLOGA test pixels.
Learned methods: mean ± std over seeds. Türkiye: spatial block bootstrap (1 km blocks, 1000 resamples) 95 % CIs for
F1/IoU and paired CIs for the F1 difference ours − benchmark.

Outputs: outputs/tables/eval_turkey_runs.csv (one row per method/event/seed), eval_turkey.csv (aggregated),
         eval_floga_test.csv, bootstrap_turkey.csv, delta_f1_turkey.csv, domain_shift.csv

Usage: python scripts/70_evaluate.py [--method dnbr_usgs] [--no-bootstrap]
  --method filters by prefix (e.g. "dnbr_usgs" -> dnbr_usgs_low, dnbr_usgs_mod).
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
from burnt import turkey, metrics, predictions as P  # noqa: E402
from burnt.dnbr import dnbr as dnbr_fn  # noqa: E402

OURS = ["lgbm", "rf", "logreg"]
BENCH = ["bamcd_v1", "bamcd_v2", "dnbr_floga", "dnbr_usgs_low", "dnbr_usgs_mod"]


def dnbr_thresholds(cfg):
    thr = dict(cfg["dnbr"]["usgs_thresholds"])
    f = ROOT / "outputs" / "tables" / "dnbr_floga_threshold.json"
    if f.exists():
        thr["dnbr_floga"] = json.loads(f.read_text())["threshold"]
    return thr


def make_dnbr_predictions(cfg, names):
    """dNBR predictions on the cached Turkish stacks (thresholds fixed a priori or tuned on FLOGA val)."""
    thr = dnbr_thresholds(cfg)
    for ev_name in names:
        ev = turkey.load_event(ev_name, cfg)
        B = ev["meta"]["bands"]
        pre, post = turkey.reflectance(ev["pre"]), turkey.reflectance(ev["post"])
        d = dnbr_fn(pre[B.index("B8A")], pre[B.index("B12")], post[B.index("B8A")], post[B.index("B12")])
        for m, t in thr.items():
            P.save_turkey(m, ev_name, (d > t) & ev["valid"], prob=np.nan_to_num(d))


def eval_turkey(cfg, methods, do_boot=True):
    names = turkey.event_names(cfg)
    run_rows, boot_rows, delta_rows = [], [], []
    pooled_counts = {}  # (method, seed) -> list of per-block counts across events
    for ev_name in names:
        ev = turkey.load_event(ev_name, cfg)
        valid, ref = ev["valid"], ev["effis"]
        blocks = metrics.block_ids(valid.shape, ev["meta"]["res_m"], cfg["bootstrap"]["block_m"])
        for m in methods:
            preds = P.load_turkey(m, ev_name)
            if not preds:
                continue
            for seed, pred in preds.items():
                r = metrics.metrics(pred & valid, ref, valid, ev["px_ha"])
                run_rows.append({"method": m, "event": ev_name, "seed": seed, **r})
                _, cnt = metrics.block_counts({m: pred}, ref, valid, blocks)
                pooled_counts.setdefault((m, seed), []).append(cnt[m])
    runs = pd.DataFrame(run_rows)
    if runs.empty:
        return runs, None, None
    # pooled Türkiye = sum of counts of both events
    pooled = []
    for (m, seed), g in runs.groupby(["method", "seed"], dropna=False):
        tp, fp, fn, tn = (int(g[k].sum()) for k in ["tp", "fp", "fn", "tn"])
        px_ha = cfg["gsd"] ** 2 / 1e4
        pooled.append({"method": m, "event": "Türkiye (pooled)", "seed": seed,
                       **metrics.from_counts(tp, fp, fn, tn, px_ha)})
    runs = pd.concat([runs, pd.DataFrame(pooled)], ignore_index=True)

    if do_boot:
        # Bootstrap per event and pooled. For seeded methods use the seed-0 run and the per-seed spread separately.
        def counts_for(m, idx):
            seeds = sorted([k[1] for k in pooled_counts if k[0] == m], key=lambda s: -1 if s is None else s)
            return pooled_counts[(m, seeds[0])][idx] if idx is not None else np.concatenate(pooled_counts[(m, seeds[0])])
        meths = sorted({k[0] for k in pooled_counts})
        ours = [m for m in OURS if m in meths]
        bench = [m for m in BENCH if m in meths]
        pairs = [(o, b) for o in ours for b in bench]
        for i, ev_name in enumerate(names + ["Türkiye (pooled)"]):
            idx = i if ev_name != "Türkiye (pooled)" else None
            c = {m: counts_for(m, idx) for m in meths}
            res = metrics.block_bootstrap(c, cfg["bootstrap"]["n_resamples"], seed=cfg["seed"], pairs=pairs)
            for k, v in res.items():
                if " - " in k:
                    delta_rows.append({"event": ev_name, "pair": k, **v})
                else:
                    boot_rows.append({"event": ev_name, "method": k, **v})
    return runs, pd.DataFrame(boot_rows), pd.DataFrame(delta_rows)


def aggregate(runs):
    keys = ["precision", "recall", "f1", "iou", "oa", "pred_area_ha", "ref_area_ha"]
    g = runs.groupby(["method", "event"], sort=False)
    agg = g[keys].mean().add_suffix("_mean").join(g[keys].std(ddof=0).add_suffix("_std")).join(g.size().rename("n_runs"))
    return agg.reset_index()


def eval_floga_test(cfg, methods):
    from burnt import floga
    try:
        ref = floga.test_reference()
    except FileNotFoundError:
        return pd.DataFrame()
    rows = []
    for m in methods:
        for seed, pred in P.load_floga_test(m).items():
            rows.append({"method": m, "seed": seed, **metrics.metrics(pred, ref)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None, help="evaluate only methods starting with this prefix")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    names = turkey.event_names(cfg)
    make_dnbr_predictions(cfg, names)
    methods = P.methods_available("turkey")
    if args.method:
        methods = [m for m in methods if m.startswith(args.method)]
    tdir = out_dir("tables")
    suffix = f"_{args.method}" if args.method else ""
    runs, boot, delta = eval_turkey(cfg, methods, do_boot=not args.no_bootstrap)
    runs.to_csv(tdir / f"eval_turkey_runs{suffix}.csv", index=False)
    agg = aggregate(runs)
    agg.to_csv(tdir / f"eval_turkey{suffix}.csv", index=False)
    if boot is not None and not boot.empty:
        boot.to_csv(tdir / f"bootstrap_turkey{suffix}.csv", index=False)
    if delta is not None and not delta.empty:
        delta.to_csv(tdir / f"delta_f1_turkey{suffix}.csv", index=False)
    fl_methods = [m for m in P.methods_available("floga_test") if not args.method or m.startswith(args.method)]
    fl = eval_floga_test(cfg, fl_methods) if fl_methods else pd.DataFrame()
    if not fl.empty:
        fl.to_csv(tdir / f"eval_floga_test_runs{suffix}.csv", index=False)
        k = ["precision", "recall", "f1", "iou", "oa"]
        g = fl.groupby("method")
        fagg = g[k].mean().add_suffix("_mean").join(g[k].std(ddof=0).add_suffix("_std")).join(g.size().rename("n_runs"))
        fagg.reset_index().to_csv(tdir / f"eval_floga_test{suffix}.csv", index=False)
        # domain shift: in-domain (FLOGA test) vs out-of-domain (Türkiye pooled) F1
        tp = agg[agg.event == "Türkiye (pooled)"].set_index("method")["f1_mean"]
        ds = fagg["f1_mean"].rename("f1_floga_test").to_frame().join(tp.rename("f1_turkey_pooled"), how="inner")
        ds["delta"] = ds["f1_turkey_pooled"] - ds["f1_floga_test"]
        ds.reset_index().to_csv(tdir / f"domain_shift{suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["method", "event", "precision_mean", "recall_mean", "f1_mean", "iou_mean", "f1_std", "n_runs"]
    print(agg[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
