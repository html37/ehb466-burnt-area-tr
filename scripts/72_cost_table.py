"""Cost table: parameters, model size, training time, CPU/GPU inference time per 1000 km^2, peak memory, and the
accuracy (F1 on FLOGA test and Türkiye pooled) of every method. Also times the dNBR benchmark on CPU.

Inputs: cost_train.csv (50), cost_infer_turkey.csv (65), cost_bamcd.csv (60), eval_turkey.csv, eval_floga_test.csv
Output: outputs/tables/cost_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir, ROOT  # noqa: E402
from burnt import turkey, cost  # noqa: E402
from burnt.dnbr import dnbr as dnbr_fn  # noqa: E402

MAIN = ["dnbr_usgs_low", "dnbr_usgs_mod", "dnbr_floga", "bamcd_v1", "bamcd_v2", "lgbm", "rf", "logreg"]


def time_dnbr(cfg, reps=3):
    rows = []
    for name in turkey.event_names(cfg):
        ev = turkey.load_event(name, cfg)
        B = ev["meta"]["bands"]
        area = ev["valid"].size * ev["px_ha"] / 100
        ts = []
        for _ in range(reps):
            res = {}
            with cost.measure(res, ""):
                pre, post = turkey.reflectance(ev["pre"][[B.index("B8A"), B.index("B12")]]), \
                    turkey.reflectance(ev["post"][[B.index("B8A"), B.index("B12")]])
                d = dnbr_fn(pre[0], pre[1], post[0], post[1]); _ = (d > 0.1) & ev["valid"]
            ts.append(res)
        rows.append({"event": name, "area_km2": area, "infer_time_s": np.median([r["time_s"] for r in ts]),
                     "infer_peak_rss_mb": max(r["peak_rss_mb"] for r in ts)})
    d = pd.DataFrame(rows)
    return float((d["infer_time_s"].sum() / d["area_km2"].sum()) * 1000), float(d["infer_peak_rss_mb"].max())


def main():
    cfg = load_config()
    t = ROOT / "outputs" / "tables"
    rows = {m: {"method": m} for m in MAIN}
    s, rss = time_dnbr(cfg)
    for m in ["dnbr_usgs_low", "dnbr_usgs_mod", "dnbr_floga"]:
        rows[m].update(params=1, model_mb=0.0, train_time_s=0.0, cpu_s_per_1000km2=s, cpu_peak_rss_mb=rss,
                       needs_gpu=False)
    if (t / "cost_bamcd.csv").exists():
        cb = pd.read_csv(t / "cost_bamcd.csv")
        for m, g in cb.groupby("method"):
            if m not in rows:
                continue
            mod = g[g["event"] == "model"]
            inf = g[g["event"] != "model"]
            area = inf.groupby("device")["area_km2"].sum(); tt = inf.groupby("device")["infer_time_s"].sum()
            rows[m].update(params=int(mod["params"].iloc[0]), model_mb=float(mod["weights_fp32_mb"].iloc[0]),
                           checkpoint_mb=float(mod["checkpoint_mb"].iloc[0]), train_time_s=np.nan,
                           needs_gpu="training: yes")
            if "cpu" in area:
                rows[m]["cpu_s_per_1000km2"] = float(tt["cpu"] / area["cpu"] * 1000)
                rows[m]["cpu_peak_rss_mb"] = float(inf[inf.device == "cpu"]["infer_peak_rss_mb"].max())
            if "cuda" in area:
                rows[m]["gpu_s_per_1000km2"] = float(tt["cuda"] / area["cuda"] * 1000)
                rows[m]["peak_vram_mb"] = float(inf[inf.device == "cuda"]["peak_vram_mb"].max())
    if (t / "cost_train.csv").exists():
        ct = pd.read_csv(t / "cost_train.csv")
        for m, g in ct.groupby("method"):
            if m in rows:
                rows[m].update(params=float(g["params"].mean()), trees=float(g["trees"].mean()),
                               model_mb=float(g["model_mb"].mean()), train_time_s=float(g["train_time_s"].mean()),
                               train_time_std=float(g["train_time_s"].std(ddof=0)),
                               train_peak_rss_mb=float(g["train_peak_rss_mb"].max()), needs_gpu=False)
    if (t / "cost_infer_turkey.csv").exists():
        ci = pd.read_csv(t / "cost_infer_turkey.csv")
        for m, g in ci.groupby("method"):
            if m in rows:
                per_seed = g.groupby("seed").apply(lambda x: x["infer_time_s"].sum() / x["area_km2"].sum() * 1000)
                rows[m].update(cpu_s_per_1000km2=float(per_seed.mean()), cpu_s_std=float(per_seed.std(ddof=0)),
                               cpu_peak_rss_mb=float(g["infer_peak_rss_mb"].max()))
    df = pd.DataFrame(rows.values())
    ev = pd.read_csv(t / "eval_turkey.csv")
    f1 = ev[ev["event"] == "Türkiye (pooled)"].set_index("method")["f1_mean"]
    df["f1_turkey"] = df["method"].map(f1)
    if (t / "eval_floga_test.csv").exists():
        fl = pd.read_csv(t / "eval_floga_test.csv").set_index("method")["f1_mean"]
        df["f1_floga_test"] = df["method"].map(fl)
    df.to_csv(out_dir("tables") / "cost_table.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
