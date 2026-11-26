"""Error analysis on the Turkish test areas: FP / FN by ESA WorldCover 2021 class and by dNBR severity band,
plus disagreement maps. Evaluation only (uses frozen predictions).

WorldCover 2021 v200 (10 m, Planetary Computer collection `esa-worldcover`) is mosaicked and resampled to the 20 m
grid with the modal class -> data/cache/turkey/<event>_worldcover.npy.

Outputs: outputs/tables/errors_by_landcover.csv, errors_by_severity.csv,
         outputs/figures/errors_by_landcover.png|pdf, disagreement_<event>.png|pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir  # noqa: E402
from burnt import turkey, predictions as P, stac  # noqa: E402
from burnt.dnbr import dnbr as dnbr_fn  # noqa: E402

WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare/sparse",
      70: "Snow/ice", 80: "Water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss/lichen"}
SEV = [(-np.inf, 0.10, "unburnt (<0.10)"), (0.10, 0.27, "low (0.10-0.27)"), (0.27, 0.44, "mod-low (0.27-0.44)"),
       (0.44, 0.66, "mod-high (0.44-0.66)"), (0.66, np.inf, "high (>=0.66)")]
METHODS = ["dnbr_usgs_low", "dnbr_floga", "bamcd_v1", "lgbm", "rf"]


def worldcover(ev, cfg):
    f = cfg["paths"]["cache"] / "turkey" / f"{ev['meta']['event']}_worldcover.npy"
    if f.exists():
        return np.load(f)
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling
    from rioxarray.merge import merge_arrays
    from affine import Affine
    import xarray as xr
    bbox = ev["meta"]["bbox_wgs84"]
    items = list(stac.catalog().search(collections=["esa-worldcover"], bbox=bbox,
                                       query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    parts = [stac.read_asset(it, "map", bbox).load() for it in items]
    wc = parts[0] if len(parts) == 1 else merge_arrays(parts, nodata=0)
    H, W = ev["meta"]["shape"]
    tr = Affine(*ev["meta"]["transform"])
    tmpl = xr.DataArray(np.zeros((H, W), np.uint8), dims=("y", "x"),
                        coords={"y": tr.f + tr.e * (np.arange(H) + 0.5), "x": tr.c + tr.a * (np.arange(W) + 0.5)})
    tmpl = tmpl.rio.write_crs(ev["meta"]["crs"]).rio.write_transform(tr)
    out = wc.rio.reproject_match(tmpl, resampling=Resampling.mode).values.astype(np.uint8)
    np.save(f, out)
    return out


def main():
    cfg = load_config()
    lc_rows, sev_rows = [], []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    for name in turkey.event_names(cfg):
        ev = turkey.load_event(name, cfg)
        valid, ref = ev["valid"], ev["effis"]
        wc = worldcover(ev, cfg)
        B = ev["meta"]["bands"]
        pre, post = turkey.reflectance(ev["pre"]), turkey.reflectance(ev["post"])
        d = dnbr_fn(pre[B.index("B8A")], pre[B.index("B12")], post[B.index("B8A")], post[B.index("B12")])
        preds = {}
        for m in METHODS:
            p = P.load_turkey(m, name)
            if p:
                preds[m] = p[sorted(p, key=lambda s: -1 if s is None else s)[0]]  # seed 0 / single run
        for m, pr in preds.items():
            tp = pr & ref & valid; fp = pr & ~ref & valid; fn = ~pr & ref & valid
            for code, lab in WC.items():
                c = (wc == code) & valid
                if c.sum() == 0:
                    continue
                lc_rows.append({"event": name, "method": m, "class": lab, "pixels": int(c.sum()),
                                "ref_burnt": int((ref & c).sum()), "tp": int((tp & c).sum()),
                                "fp": int((fp & c).sum()), "fn": int((fn & c).sum())})
            for lo, hi, lab in SEV:
                s = (d >= lo) & (d < hi) & valid
                sev_rows.append({"event": name, "method": m, "severity": lab, "pixels": int(s.sum()),
                                 "ref_burnt": int((ref & s).sum()), "tp": int((tp & s).sum()),
                                 "fp": int((fp & s).sum()), "fn": int((fn & s).sum())})
        # disagreement map: ours (lgbm) vs BAM-CD vs reference
        if "lgbm" in preds and "bamcd_v1" in preds:
            o, b = preds["lgbm"], preds["bamcd_v1"]
            cat = np.zeros(ref.shape, np.uint8)
            cat[o & b & ref] = 1            # both correct burnt
            cat[o & ~b & ref] = 2           # only ours correct
            cat[~o & b & ref] = 3           # only BAM-CD correct
            cat[~o & ~b & ref] = 4          # both miss
            cat[(o | b) & ~ref] = 5         # false alarm by at least one
            cat[~valid] = 6
            cols = ["#f0f0f0", "#1a9850", "#2166ac", "#d6604d", "#762a83", "#fdae61", "#ffffff"]
            labs = ["both TP", "only LightGBM TP", "only BAM-CD TP", "both FN", "FP (either)", "masked"]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(cat, cmap=ListedColormap(cols), vmin=0, vmax=6, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols[1:], labs)], loc="lower center",
                      ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.12), frameon=False)
            ax.set_title(f"{name}: agreement of LightGBM and BAM-CD v1 with EFFIS")
            for ext in ["png", "pdf"]:
                fig.savefig(out_dir("figures") / f"disagreement_{name}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
    lc = pd.DataFrame(lc_rows); sv = pd.DataFrame(sev_rows)
    for df in (lc, sv):
        df["fp_rate_of_pixels"] = df["fp"] / df["pixels"].clip(lower=1)
        df["fn_rate_of_ref"] = df["fn"] / df["ref_burnt"].clip(lower=1)
    lc.to_csv(out_dir("tables") / "errors_by_landcover.csv", index=False)
    sv.to_csv(out_dir("tables") / "errors_by_severity.csv", index=False)
    # chart: FP and FN pixels (ha) by land cover, pooled over events
    g = lc.groupby(["method", "class"])[["fp", "fn", "pixels"]].sum().reset_index()
    keep = lc.groupby("class")["pixels"].sum().sort_values(ascending=False)
    keep = [c for c in keep.index if keep[c] > 0.005 * keep.sum()]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    meths = [m for m in METHODS if m in g["method"].unique()]
    w = 0.8 / max(len(meths), 1); x = np.arange(len(keep))
    for ax, k, t in zip(axs, ["fp", "fn"], ["False positives (commission)", "False negatives (omission)"]):
        for i, m in enumerate(meths):
            s = g[g["method"] == m].set_index("class").reindex(keep)[k].fillna(0) * 0.04  # ha per 20 m pixel
            ax.bar(x + (i - (len(meths) - 1) / 2) * w, s, w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(keep, rotation=30, ha="right", fontsize=8); ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("area (ha), both events"); axs[1].legend(fontsize=7)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir("figures") / f"errors_by_landcover.{ext}", dpi=300, bbox_inches="tight")
    print(lc.groupby(["method", "class"])[["fp", "fn"]].sum().to_string())
    print(sv.groupby(["method", "severity"])[["fp", "fn", "ref_burnt"]].sum().to_string())


if __name__ == "__main__":
    main()
