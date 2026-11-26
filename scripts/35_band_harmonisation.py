"""Input harmonisation check (CLAUDE.md rule 2.4): per-band reflectance distributions of unburnt pixels,
FLOGA (train patches, after offset removal) vs the Turkish scenes (Planetary Computer, outside EFFIS).

A diagnostic only: nothing is fitted or selected from it.
Outputs: outputs/figures/band_harmonisation.png|pdf, outputs/tables/band_harmonisation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir  # noqa: E402
from burnt import turkey, floga  # noqa: E402

N = 200_000


def main():
    cfg = load_config()
    rng = np.random.default_rng(0)
    B = floga.FLOGA_BANDS
    tr = pd.read_parquet(cfg["paths"]["cache"] / "floga" / "samples_train.parquet")
    tr = tr[(tr["label"] == 0) & (tr["sea"] == 0)]
    tr = tr.sample(min(N, len(tr)), random_state=0)
    dom = {"FLOGA (Greece) pre": tr[[f"pre_{b}" for b in B]].to_numpy() / 1e4,
           "FLOGA (Greece) post": tr[[f"post_{b}" for b in B]].to_numpy() / 1e4}
    for name in turkey.event_names(cfg):
        ev = turkey.load_event(name, cfg)
        m = ev["valid"] & ~ev["effis"] & ~np.isin(ev["pre_scl"], [6]) & ~np.isin(ev["post_scl"], [6])
        idx = np.flatnonzero(m.ravel())
        idx = rng.choice(idx, min(N, len(idx)), replace=False)
        for role in ["pre", "post"]:
            dom[f"{name} {role}"] = ev[role].reshape(9, -1)[:, idx].T / 1e4
    rows = []
    for k, a in dom.items():
        for i, b in enumerate(B):
            q = np.percentile(a[:, i], [5, 25, 50, 75, 95])
            rows.append({"domain": k, "band": b, "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir("tables") / "band_harmonisation.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(10, 7.5))
    styles = {"FLOGA (Greece) pre": ("k", "-"), "FLOGA (Greece) post": ("k", "--"),
              "Manavgat pre": ("#d95f02", "-"), "Manavgat post": ("#d95f02", "--"),
              "Marmaris pre": ("#1b9e77", "-"), "Marmaris post": ("#1b9e77", "--")}
    bins = np.linspace(0, 0.6, 121)
    for ax, (i, b) in zip(axs.ravel(), enumerate(B)):
        for k, a in dom.items():
            c, ls = styles.get(k, ("grey", "-"))
            h, e = np.histogram(np.clip(a[:, i], 0, 0.6), bins=bins, density=True)
            ax.plot((e[1:] + e[:-1]) / 2, h, color=c, ls=ls, lw=1, label=k)
        ax.set_title(b, fontsize=9); ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=6)
    fig.supxlabel("surface reflectance"); fig.supylabel("density")
    fig.suptitle("Unburnt land pixels: FLOGA (offset removed) vs Turkish Planetary-Computer scenes", fontsize=10)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir("figures") / f"band_harmonisation.{ext}", dpi=300, bbox_inches="tight")
    print(df.pivot_table(index="band", columns="domain", values="median").round(3).to_string())


if __name__ == "__main__":
    main()
