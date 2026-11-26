"""Build the Turkish test set (Manavgat, Marmaris) on the 20 m grid with exactly the FLOGA band set/order.

For each event: select pre/post Sentinel-2 L2A passes (same logic as notebook 01), mosaic the tiles of each pass,
resample every band onto the 20 m grid of the pre-fire B12 mosaic, and save
    data/cache/turkey/<event>.npz : pre, post (9, H, W) uint16 DN (BOA offset applied, 0 = no data),
                                    pre_scl, post_scl (uint8), valid (bool), effis (bool)
    data/cache/turkey/<event>.json: scene ids, tiles, dates, CRS, transform, shape, sha256 of arrays
    outputs/tables/selected_scenes.csv
This script never looks at model outputs; EFFIS is stored only as the evaluation reference.

Usage: python scripts/10_build_turkey_testset.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir  # noqa: E402
from burnt import stac, preprocess, reference  # noqa: E402


def build_event(name, ev, cfg, effis):
    bbox = reference.event_bbox(effis, ev["effis_id"], cfg["aoi_buffer_m"])
    sc = cfg["stac"]
    pre = stac.pick_scene(stac.search(bbox, ev["pre"], sc["collection"], sc["max_tile_cloud"]), bbox,
                          cfg["scl_clear"], latest_first=True, min_clear=sc["min_clear"],
                          max_candidates=sc["max_candidates"])
    post = stac.pick_scene(stac.search(bbox, ev["post"], sc["collection"], sc["max_tile_cloud"]), bbox,
                           cfg["scl_clear"], latest_first=False, min_clear=sc["min_clear"],
                           max_candidates=sc["max_candidates"])
    ref = preprocess.read_dn(pre, "B12", bbox)  # 20 m reference grid (mosaic), as in notebook 01
    stacks = {}
    for role, g in [("pre", pre), ("post", post)]:
        arr = []
        for b in cfg["bands"]:
            da = ref if (role == "pre" and b == "B12") else preprocess.read_dn(g, b, bbox, ref)
            v = da.values
            u = np.where(np.isfinite(v), np.clip(np.rint(v), 1, 65535), 0).astype(np.uint16)
            arr.append(u)
        stacks[role] = np.stack(arr)
    pre_scl = preprocess.scl_on_grid(pre["scl"], ref).astype(np.uint8)
    post_scl = preprocess.scl_on_grid(post["scl"], ref).astype(np.uint8)
    valid = (np.isin(pre_scl, cfg["scl_valid_land"]) & np.isin(post_scl, cfg["scl_valid_land"])
             & np.all(stacks["pre"] > 0, 0) & np.all(stacks["post"] > 0, 0))
    pre_d = pd.Timestamp(pre["day"]); post_d = pd.Timestamp(post["day"])
    gt, sel = reference.effis_mask(effis, ref, pre_d, post_d)
    rows = []
    for role, g in [("pre", pre), ("post", post)]:
        for it in g["items"]:
            rows.append({"event": name, "role": role, "date": str(g["day"]), "item_id": it.id,
                         "platform": it.properties.get("platform"),
                         "mgrs_tile": it.properties["s2:mgrs_tile"],
                         "processing_baseline": it.properties.get("s2:processing_baseline"),
                         "boa_offset": preprocess.boa_offset(it),
                         "tile_cloud_pct": it.properties["eo:cloud_cover"],
                         "aoi_coverage_pct": round(100 * g["coverage"], 2),
                         "aoi_clear_pct": round(100 * g["clear"], 2)})
    arrays = dict(pre=stacks["pre"], post=stacks["post"], pre_scl=pre_scl, post_scl=post_scl,
                  valid=valid, effis=gt)
    meta = {
        "event": name, "effis_id": ev["effis_id"], "bbox_wgs84": bbox,
        "pre_date": str(pre["day"]), "post_date": str(post["day"]),
        "crs": str(ref.rio.crs), "transform": list(ref.rio.transform())[:6], "shape": list(ref.shape),
        "res_m": abs(ref.rio.resolution()[0]), "bands": cfg["bands"],
        "effis_ids_in_reference": [int(i) for i in sel["id"].tolist()],
        "valid_fraction": float(valid.mean()),
        "effis_masked_pct": float(100 * np.sum(gt & ~valid) / max(np.sum(gt), 1)),
        "sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()},
    }
    return arrays, meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    outd = cfg["paths"]["cache"] / "turkey"
    outd.mkdir(parents=True, exist_ok=True)
    effis = reference.load_effis(cfg["paths"]["effis"])
    all_rows = []
    for name, ev in cfg["events"].items():
        npz, js = outd / f"{name}.npz", outd / f"{name}.json"
        if npz.exists() and js.exists() and not args.force:
            print(f"{name}: cached"); all_rows += json.loads(js.read_text())["scenes"]; continue
        t = time.perf_counter()
        arrays, meta, rows = build_event(name, ev, cfg, effis)
        meta["scenes"] = rows
        np.savez_compressed(npz, **arrays)
        js.write_text(json.dumps(meta, indent=1))
        all_rows += rows
        print(f"{name}: pre {meta['pre_date']} post {meta['post_date']} grid {meta['shape']} "
              f"valid {100*meta['valid_fraction']:.1f}% EFFIS masked {meta['effis_masked_pct']:.2f}% "
              f"({time.perf_counter()-t:.0f} s)")
    pd.DataFrame(all_rows).to_csv(out_dir("tables") / "selected_scenes.csv", index=False)


if __name__ == "__main__":
    main()
