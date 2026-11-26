"""Prepare FLOGA (official event split) for this project, reading only the needed bytes from Hugging Face.

Stage A  masks:   for every split event read label, SCL cloud masks (pre/post), sea mask, CLC; tile into 256x256
                  patches exactly like FLOGA create_dataset.py; flag positive / cloudy (>10 SCL=9 px in pre or
                  post) / sea (>=90 % sea) / zero (no data) patches -> data/cache/floga/masks/<key>.npz +
                  data/cache/floga/patch_table.parquet
Stage B  select:  per split keep all valid positive patches and the same number of randomly chosen valid negative
                  patches (1:1, FLOGA README), seed 999 (FLOGA config seed) -> data/cache/floga/selected_patches.csv
Stage C  images:  read the pre/post Sentinel-2 windows of the selected patches, drop B01, remove the +1000 BOA
                  offset of the N0500 reprocessing -> data/cache/floga/patches/<split>/<key>_<r>_<c>.npz
Stage D  sample:  pixel sampling for the ML models (paper Sec. V-B): every patch gives N/2 unburnt pixels
                  (>= 10 % from sea/water when present), positive patches also N/2 burnt pixels; cloudy (SCL 3,8,9,10)
                  and label-2 pixels excluded -> data/cache/floga/samples_<split>.parquet (train and val)

All stages are resumable (cached outputs are skipped). Usage: python scripts/20_prepare_floga.py [--stages A B C D]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir  # noqa: E402
from burnt import floga  # noqa: E402

SEED = 999
CLOUDY_SCL = [3, 8, 9, 10]  # pixel-level exclusion for sampling: shadow, cloud medium/high, cirrus


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ stage A
def stage_a_event(key, cfg):
    d = cfg["paths"]["cache"] / "floga" / "masks"
    f = d / (key.replace("/", "_") + ".npz")
    if f.exists():
        z = np.load(f)
        return key, {k: z[k] for k in z.files}
    arrs = {}
    for what in ["label", "cloud_pre", "cloud_post", "sea", "clc"]:
        arrs[what] = floga.open_file(key, what).read_all()[0].astype(np.uint8)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(f)
    return key, arrs


def patch_rows(key, arrs, size, sea_ratio, cloud_max):
    lab = arrs["label"]
    H, W = lab.shape
    sea = floga.sea_binary(arrs["sea"])
    rows = []
    for r, c in floga.patch_grid((H, W), size):
        l = floga.crop(lab, r, c, size)
        cp = floga.crop(arrs["cloud_pre"], r, c, size)
        cq = floga.crop(arrs["cloud_post"], r, c, size)
        s = floga.crop(sea, r, c, size)
        rows.append({"key": key, "r": r, "c": c, "H": H, "W": W,
                     "positive": bool((l == 1).any()), "n_burnt": int((l == 1).sum()),
                     "has_other": bool((l == 2).any()),
                     "cloudy": bool(((cp == floga.CLOUD_SCL).sum() > cloud_max) or ((cq == floga.CLOUD_SCL).sum() > cloud_max)),
                     "sea": bool(s.sum() >= sea_ratio * size * size),
                     "zero": bool((cp == 0).all() or (cq == 0).all())})
    return rows


def stage_a(cfg, ev):
    (cfg["paths"]["cache"] / "floga" / "masks").mkdir(parents=True, exist_ok=True)
    out = cfg["paths"]["cache"] / "floga" / "patch_table.parquet"
    if out.exists():
        log("stage A cached"); return pd.read_parquet(out)
    fc = cfg["floga"]
    keys = ev.loc[ev["key"].notna() & ~ev["dup_dropped"], "key"].tolist()
    rows = []
    def work(k):  # return only the small per-patch table so the mask arrays are freed immediately
        key, arrs = stage_a_event(k, cfg)
        return patch_rows(key, arrs, fc["patch"], fc["sea_ratio"], fc["cloud_max_px"])

    with ThreadPoolExecutor(6) as ex:  # ex.map keeps no finished results alive
        for i, r in enumerate(ex.map(work, keys)):
            rows += r
            if i % 20 == 0:
                log(f"stage A {i+1}/{len(keys)}")
    pt = pd.DataFrame(rows).merge(ev[["key", "set", "year", "fire_id"]].drop_duplicates("key"), on="key")
    pt.to_parquet(out)
    return pt


# ------------------------------------------------------------------ stage B
def stage_b(cfg, pt):
    out = cfg["paths"]["cache"] / "floga" / "selected_patches.csv"
    if out.exists():
        log("stage B cached"); return pd.read_csv(out)
    ok = pt[~pt["cloudy"] & ~pt["sea"] & ~pt["zero"]]
    sel = []
    rng = np.random.default_rng(SEED)
    for split in ["train", "val", "test"]:
        s = ok[ok["set"] == split].sort_values(["key", "r", "c"]).reset_index(drop=True)
        pos = s[s["positive"]]
        neg = s[~s["positive"]]
        n = min(len(pos) * cfg["floga"]["neg_ratio"], len(neg))
        negs = neg.iloc[np.sort(rng.choice(len(neg), size=n, replace=False))]
        sel.append(pd.concat([pos, negs]))
        log(f"stage B {split}: {len(pos)} positive + {n} negative patches "
            f"(eligible negatives {len(neg)}; removed cloudy {int((pt['set'].eq(split) & pt['cloudy']).sum())}, "
            f"sea {int((pt['set'].eq(split) & pt['sea']).sum())}, zero {int((pt['set'].eq(split) & pt['zero']).sum())})")
    sel = pd.concat(sel).reset_index(drop=True)
    sel.to_csv(out, index=False)
    return sel


# ------------------------------------------------------------------ stage C
def stage_c_patch(row, ev_off, cfg):
    d = floga.patch_dir(row["set"], cfg)
    f = d / f"{row['key'].replace('/', '_')}_{row['r']}_{row['c']}.npz"
    if f.exists():
        return f, "cached"
    size = cfg["floga"]["patch"]
    r, c = int(row["r"]), int(row["c"])
    arrs = {}
    for role in ["pre", "post"]:
        raw = floga.open_file(row["key"], role).read_window(r, r + size, c, c + size)
        arrs[role] = floga.to_dn(raw, ev_off[row["key"]][role])
    z = np.load(cfg["paths"]["cache"] / "floga" / "masks" / (row["key"].replace("/", "_") + ".npz"))
    arrs["label"] = floga.crop(z["label"], r, c, size)
    arrs["cloud_pre"] = floga.crop(z["cloud_pre"], r, c, size)
    arrs["cloud_post"] = floga.crop(z["cloud_post"], r, c, size)
    arrs["sea"] = floga.crop(floga.sea_binary(z["sea"]), r, c, size)
    arrs["clc"] = floga.crop(z["clc"], r, c, size)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(f)
    return f, "new"


def stage_c(cfg, ev, sel):
    for s in ["train", "val", "test"]:
        floga.patch_dir(s, cfg).mkdir(parents=True, exist_ok=True)
    off = {}
    for _, e in ev[ev["key"].notna()].iterrows():
        off[e["key"]] = {"pre": 1000 if (e["baseline_pre"] or 0) >= 400 else 0,
                         "post": 1000 if (e["baseline_post"] or 0) >= 400 else 0}
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(stage_c_patch, row, off, cfg) for _, row in sel.iterrows()]
        for i, fu in enumerate(as_completed(futs)):
            fu.result()
            if i % 25 == 0:
                log(f"stage C {i+1}/{len(futs)} ({time.time()-t0:.0f} s)")


# ------------------------------------------------------------------ stage D
def sample_patch(p, n, sea_frac, rng):
    m = floga.pixel_mask(p) & ~np.isin(p["cloud_pre"], CLOUDY_SCL) & ~np.isin(p["cloud_post"], CLOUDY_SCL)
    lab = p["label"]
    burnt = np.flatnonzero(m & (lab == 1))
    unb = m & (lab == 0)
    sea = np.flatnonzero(unb & (p["sea"] == 1))
    land = np.flatnonzero(unb & (p["sea"] == 0))
    half = n // 2
    pick = []
    if len(burnt):
        pick.append(rng.choice(burnt, min(half, len(burnt)), replace=False))
    n_sea = min(len(sea), int(np.ceil(sea_frac * half))) if len(sea) else 0
    if n_sea:
        pick.append(rng.choice(sea, n_sea, replace=False))
    n_land = min(len(land), half - n_sea)
    if n_land:
        pick.append(rng.choice(land, n_land, replace=False))
    return np.concatenate(pick) if pick else np.array([], int)


def stage_d(cfg):
    fc = cfg["floga"]
    for split in ["train", "val"]:
        out = cfg["paths"]["cache"] / "floga" / f"samples_{split}.parquet"
        if out.exists():
            log(f"stage D {split} cached"); continue
        rng = np.random.default_rng(SEED)
        parts = []
        for f in floga.list_patches(split, cfg):
            p = floga.load_patch(f)
            idx = sample_patch(p, fc["n_per_patch"], fc["sea_frac_unburnt"], rng)
            if len(idx) == 0:
                continue
            pre = p["pre"].reshape(9, -1)[:, idx].T
            post = p["post"].reshape(9, -1)[:, idx].T
            df = pd.DataFrame(np.hstack([pre, post]), columns=[f"pre_{b}" for b in floga.FLOGA_BANDS] +
                              [f"post_{b}" for b in floga.FLOGA_BANDS])
            df["label"] = (p["label"].reshape(-1)[idx] == 1).astype(np.uint8)
            df["sea"] = p["sea"].reshape(-1)[idx]
            df["clc"] = p["clc"].reshape(-1)[idx]
            df["patch"] = f.stem
            parts.append(df)
        d = pd.concat(parts, ignore_index=True)
        d.to_parquet(out)
        log(f"stage D {split}: {len(d)} pixels, burnt {int(d.label.sum())}, patches {d.patch.nunique()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["A", "B", "C", "D"])
    args = ap.parse_args()
    cfg = load_config()
    (cfg["paths"]["cache"] / "floga").mkdir(parents=True, exist_ok=True)
    ev = floga.event_table(cfg)
    ev.to_csv(out_dir("tables") / "floga_event_mapping.csv", index=False)
    summ = ev.groupby("set").agg(events=("event_id", "size"), mapped=("key", lambda s: s.notna().sum()),
                                 dup_dropped=("dup_dropped", "sum"))
    log("event mapping\n" + summ.to_string())
    pt = sel = None
    if "A" in args.stages:
        pt = stage_a(cfg, ev)
    if "B" in args.stages:
        pt = pt if pt is not None else pd.read_parquet(cfg["paths"]["cache"] / "floga" / "patch_table.parquet")
        sel = stage_b(cfg, pt)
    if "C" in args.stages:
        sel = sel if sel is not None else pd.read_csv(cfg["paths"]["cache"] / "floga" / "selected_patches.csv")
        stage_c(cfg, ev, sel)
    if "D" in args.stages:
        stage_d(cfg)
    counts = {s: len(floga.list_patches(s, cfg)) for s in ["train", "val", "test"]}
    (out_dir("tables") / "floga_patch_counts.json").write_text(json.dumps(counts, indent=1))
    log("patches on disk", counts)


if __name__ == "__main__":
    main()
