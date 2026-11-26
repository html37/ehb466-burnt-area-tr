"""End-to-end demo (used by notebooks/00_demo_end_to_end.ipynb; also runs locally).

For each requested event: (1) build the 20 m Sentinel-2 stack from Planetary Computer if not cached,
(2) dNBR with USGS 0.10 and the FLOGA-val-tuned threshold, (3) our LightGBM (FLOGA-trained, seed 0, frozen
threshold), (4) optionally BAM-CD v1 (official weights), (5) metrics vs EFFIS + comparison figure, (6) runtime.

Pre-trained files are downloaded automatically when missing:
  LightGBM models  -> GitHub release v1.0 asset models_lgbm.zip (also committed in outputs/models/lgbm)
  BAM-CD v1 weights -> official FLOGA Dropbox link (1.0 GB)

Usage: python scripts/demo_run.py [--events Manavgat Marmaris] [--bamcd auto|on|off] [--stride 256]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from burnt.config import load_config  # noqa: E402
from burnt import turkey, metrics, features as F, models as M  # noqa: E402
from burnt.dnbr import dnbr as dnbr_fn  # noqa: E402

RELEASE = "https://github.com/html37/ehb466-burnt-area-tr/releases/download/v1.0/"
BAMCD_V1 = ("https://www.dropbox.com/scl/fo/7boya0nvjb0sgmg9l5quo/ALldNkZ5d-DMtUMmUPm5O4A?"
            "rlkey=1p29xzb912mj4hh68k8waac9x&st=rkk9ke5g&dl=1")


def fetch_zip(url, dest):
    print(f"downloading {url[:80]}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=600).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            if m.endswith("/"):
                continue
            out = dest / Path(m).name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(m))


def ensure_models():
    d = ROOT / "outputs" / "models" / "lgbm"
    if not (d / "seed0.pkl").exists():
        fetch_zip(RELEASE + "models_lgbm.zip", d)
    return d


def ensure_bamcd():
    f = ROOT / "data" / "raw" / "bamcd" / "v1" / "best_segmentation.pt"
    if not f.exists():
        fetch_zip(BAMCD_V1, f.parent)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=["Manavgat"])
    ap.add_argument("--bamcd", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--stride", type=int, default=None, help="BAM-CD window stride (default 128 GPU / 256 CPU)")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    timings = {}
    # 1. data
    t = time.time()
    missing = [e for e in a.events if not (cfg["paths"]["cache"] / "turkey" / f"{e}.npz").exists()]
    if missing:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "10_build_turkey_testset.py")], check=True)
    timings["data access + preprocessing"] = time.time() - t
    thr_floga = json.loads((ROOT / "outputs" / "tables" / "dnbr_floga_threshold.json").read_text())["threshold"]
    mdir = ensure_models()
    clf = M.load(mdir / "seed0.pkl"); meta = json.loads((mdir / "seed0.json").read_text())
    use_bam = a.bamcd == "on"
    if a.bamcd == "auto":
        import torch
        use_bam = True
        print("BAM-CD device:", "cuda" if torch.cuda.is_available() else "cpu")
    if use_bam:
        ensure_bamcd()
        from burnt import bamcd
        model, device = bamcd.load("v1")
        stride = a.stride or (128 if device == "cuda" else 256)
    rows, maps = [], {}
    for e in a.events:
        ev = turkey.load_event(e, cfg)
        valid, ref = ev["valid"], ev["effis"]
        B = ev["meta"]["bands"]
        preds = {}
        t = time.time()
        pre, post = turkey.reflectance(ev["pre"]), turkey.reflectance(ev["post"])
        d = dnbr_fn(pre[B.index("B8A")], pre[B.index("B12")], post[B.index("B8A")], post[B.index("B12")])
        preds["dNBR > 0.10 (USGS)"] = (d > 0.10) & valid
        preds[f"dNBR > {thr_floga:.2f} (FLOGA val)"] = (d > thr_floga) & valid
        timings[f"{e}: dNBR"] = time.time() - t
        t = time.time()
        X = F.from_stack(ev["pre"], ev["post"], valid, meta["group"])
        prob = np.zeros(valid.shape, np.float32); prob[valid] = M.predict_proba(clf, X)
        preds["LightGBM (ours)"] = prob > meta["threshold"]
        timings[f"{e}: LightGBM"] = time.time() - t
        if use_bam:
            t = time.time()
            pb = bamcd.predict_raster(model, device, ev["pre"].astype(np.float32), ev["post"].astype(np.float32),
                                      stride=stride)
            preds[f"BAM-CD v1 (stride {stride})"] = (pb > 0.5) & valid
            timings[f"{e}: BAM-CD ({device})"] = time.time() - t
        for k, p in preds.items():
            r = metrics.metrics(p, ref, valid, ev["px_ha"])
            rows.append({"event": e, "method": k, **{m: round(r[m], 4) for m in ["precision", "recall", "f1", "iou"]},
                         "pred_ha": round(r["pred_area_ha"]), "effis_ha": round(r["ref_area_ha"])})
        maps[e] = (ev, preds)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "demo"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "demo_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f2f2f2", "#d7301f", "#fdae61", "#2c7bb6", "#ffffff"])
    for e, (ev, preds) in maps.items():
        fig, axs = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4))
        for ax, (k, p) in zip(axs, preds.items()):
            err = np.zeros(p.shape, np.uint8); ref = ev["effis"]
            err[p & ref] = 1; err[p & ~ref] = 2; err[~p & ref] = 3; err[~ev["valid"]] = 4
            ax.imshow(err, cmap=cmap, vmin=0, vmax=4, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            f1 = df[(df.event == e) & (df.method == k)].f1.iloc[0]
            ax.set_title(f"{k}\nF1 = {f1:.3f}", fontsize=9)
        fig.suptitle(f"{e}: red TP, orange FP, blue FN (reference EFFIS)")
        fig.savefig(out / f"demo_{e}.png", dpi=150, bbox_inches="tight")
        if "inline" in matplotlib.get_backend():
            plt.show()
        plt.close(fig)
    timings["total"] = time.time() - t0
    print("\nruntime (s):"); [print(f"  {k:35s} {v:7.1f}") for k, v in timings.items()]
    (out / "demo_runtime.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
