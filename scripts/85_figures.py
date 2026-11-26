"""Paper figures from the evaluation outputs (300 dpi PNG + PDF in outputs/figures/).

study_area        EFFIS 2021 fires in Türkiye (lon/lat) + post-fire false-colour of both AOIs with the EFFIS perimeter
compare_<event>   post-fire image, EFFIS reference and TP/FP/FN maps of dNBR (0.10), BAM-CD v1, LightGBM
metrics_ci        F1 per method and event with block-bootstrap 95 % CIs (learned models: seed-0 CI, dot = mean)
cost_accuracy     pooled Turkish F1 vs CPU inference time per 1000 km^2 and vs number of parameters (log axes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir, ROOT  # noqa: E402
from burnt import turkey, predictions as P, reference  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LABEL = {"dnbr_usgs_low": "dNBR>0.10 (USGS)", "dnbr_usgs_mod": "dNBR>0.27 (USGS)", "dnbr_floga": "dNBR (FLOGA-val thr.)",
         "bamcd_v1": "BAM-CD v1", "bamcd_v2": "BAM-CD v2", "lgbm": "LightGBM (ours)", "rf": "Random Forest (ours)",
         "logreg": "LogReg G2 (ours)"}
ORDER = ["dnbr_usgs_mod", "dnbr_usgs_low", "dnbr_floga", "bamcd_v1", "bamcd_v2", "logreg", "rf", "lgbm"]
FIG = out_dir("figures")


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def stretch(rgb):
    out = np.empty_like(rgb)
    for k in range(3):
        b = rgb[..., k]; lo, hi = np.nanpercentile(b, [2, 98])
        out[..., k] = np.clip((b - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(out)


def false_colour(ev):
    B = ev["meta"]["bands"]; r = turkey.reflectance(ev["post"])
    return stretch(np.dstack([r[B.index("B12")], r[B.index("B8A")], r[B.index("B04")]]))


def study_area(cfg):
    effis = reference.load_effis(cfg["paths"]["effis"]).to_crs(4326)
    fig = plt.figure(figsize=(10, 4.2))
    ax0 = fig.add_axes([0.0, 0.1, 0.36, 0.8])
    c = effis.geometry.centroid
    ax0.scatter(c.x, c.y, s=np.clip(effis["area_ha"].astype(float) / 300, 2, 200), c="#999999", alpha=0.6,
                label="EFFIS fires Jul–Aug 2021")
    for name, e in cfg["events"].items():
        p = effis[effis["id"] == e["effis_id"]]
        cc = p.geometry.centroid.iloc[0]
        ax0.scatter(cc.x, cc.y, s=80, c="red", marker="*", zorder=3)
        ax0.annotate(name, (cc.x, cc.y), xytext=(4, 6), textcoords="offset points", fontsize=9, color="red")
    ax0.set_xlabel("longitude (°E)"); ax0.set_ylabel("latitude (°N)"); ax0.set_aspect("equal")
    ax0.set_title("(a) EFFIS burnt areas, Türkiye 2021", fontsize=9); ax0.grid(alpha=0.3); ax0.legend(fontsize=7)
    for i, name in enumerate(turkey.event_names(cfg)):
        ev = turkey.load_event(name, cfg)
        ax = fig.add_axes([0.40 + i * 0.31, 0.1, 0.29, 0.8])
        ax.imshow(false_colour(ev))
        ax.contour(ev["effis"].astype(float), levels=[0.5], colors="yellow", linewidths=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        m = ev["meta"]
        ax.set_title(f"({'bc'[i]}) {name}: post-fire {m['post_date']}\nB12/B8A/B04, yellow = EFFIS", fontsize=8)
    save(fig, "study_area")


def compare_maps(cfg):
    cols = ["#f2f2f2", "#d7301f", "#fdae61", "#2c7bb6", "#ffffff"]
    cmap = ListedColormap(cols)
    runs = pd.read_csv(ROOT / "outputs" / "tables" / "eval_turkey_runs.csv")
    for name in turkey.event_names(cfg):
        ev = turkey.load_event(name, cfg)
        valid, ref = ev["valid"], ev["effis"]
        meths = [m for m in ["dnbr_usgs_low", "bamcd_v1", "lgbm"] if P.load_turkey(m, name)]
        fig, axs = plt.subplots(1, 2 + len(meths), figsize=(3.2 * (2 + len(meths)), 3.4))
        axs[0].imshow(false_colour(ev)); axs[0].set_title("post-fire (B12/B8A/B04)", fontsize=8)
        axs[1].imshow(np.where(valid, ref, np.nan), cmap="Reds", vmin=0, vmax=1.5, interpolation="nearest")
        axs[1].set_title("EFFIS reference", fontsize=8)
        for ax, m in zip(axs[2:], meths):
            preds = P.load_turkey(m, name)
            seed = sorted(preds, key=lambda s: -1 if s is None else s)[0]
            pr = preds[seed]
            e = np.zeros(ref.shape, np.uint8)
            e[pr & ref] = 1; e[pr & ~ref] = 2; e[~pr & ref] = 3; e[~valid] = 4
            ax.imshow(e, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            r = runs[(runs.method == m) & (runs.event == name)]
            r = r[r.seed.isna()] if seed is None else r[r.seed == seed]
            ax.set_title(f"{LABEL[m]}\nF1={r.f1.iloc[0]:.3f}  IoU={r.iou.iloc[0]:.3f}", fontsize=8)
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        fig.legend(handles=[Patch(color=cols[1], label="TP"), Patch(color=cols[2], label="FP (commission)"),
                            Patch(color=cols[3], label="FN (omission)"),
                            Patch(facecolor="white", edgecolor="grey", label="masked")],
                   loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"{name} 2021", fontsize=10)
        save(fig, f"compare_{name}")


def metrics_ci():
    t = ROOT / "outputs" / "tables"
    agg = pd.read_csv(t / "eval_turkey.csv"); boot = pd.read_csv(t / "bootstrap_turkey.csv")
    events = list(agg["event"].unique())
    meths = [m for m in ORDER if m in agg["method"].unique()]
    fig, axs = plt.subplots(1, len(events), figsize=(3.6 * len(events), 3.4), sharey=True)
    for ax, ev in zip(np.atleast_1d(axs), events):
        for i, m in enumerate(meths):
            a = agg[(agg.method == m) & (agg.event == ev)]
            b = boot[(boot.method == m) & (boot.event == ev)]
            if a.empty:
                continue
            y = a["f1_mean"].iloc[0]
            col = "#C44E52" if m in ("lgbm", "rf", "logreg") else ("#4C72B0" if m.startswith("bamcd") else "#777777")
            ax.barh(i, y, color=col, alpha=0.75)
            if len(b):
                ax.errorbar(y, i, xerr=[[y - b["f1_lo"].iloc[0]], [b["f1_hi"].iloc[0] - y]], fmt="none",
                            ecolor="k", capsize=2, lw=0.8)
            ax.text(0.31, i, f"{y:.3f}", va="center", fontsize=7)
        ax.set_yticks(range(len(meths))); ax.set_yticklabels([LABEL[m] for m in meths], fontsize=7)
        ax.set_xlim(0.3, 1.0); ax.set_title(ev, fontsize=9); ax.set_xlabel("F1 (burnt)"); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Türkiye test areas vs EFFIS (error bars: 95 % spatial block-bootstrap CI)", fontsize=9)
    fig.tight_layout()
    save(fig, "metrics_ci")


def cost_accuracy():
    t = ROOT / "outputs" / "tables"
    c = pd.read_csv(t / "cost_table.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, x, xl in [(axs[0], "cpu_s_per_1000km2", "CPU inference time per 1000 km² (s)"),
                      (axs[1], "params", "parameters / tree nodes")]:
        for _, r in c.iterrows():
            if pd.isna(r.get(x)) or pd.isna(r.get("f1_turkey")):
                continue
            col = "#C44E52" if r["method"] in ("lgbm", "rf", "logreg") else ("#4C72B0" if r["method"].startswith("bamcd") else "#777777")
            ax.scatter(r[x], r["f1_turkey"], color=col, s=30)
            ax.annotate(LABEL.get(r["method"], r["method"]), (r[x], r["f1_turkey"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel(xl, fontsize=8); ax.grid(alpha=0.3)
    axs[0].set_ylabel("F1, Türkiye pooled")
    fig.suptitle("Accuracy vs cost", fontsize=9); fig.tight_layout()
    save(fig, "cost_accuracy")


def main():
    cfg = load_config()
    study_area(cfg)
    t = ROOT / "outputs" / "tables"
    if (t / "eval_turkey_runs.csv").exists():
        compare_maps(cfg)
    if (t / "bootstrap_turkey.csv").exists():
        metrics_ci()
    if (t / "cost_table.csv").exists():
        cost_accuracy()


if __name__ == "__main__":
    main()
