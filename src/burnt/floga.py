"""FLOGA dataset access (v2 GeoTIFFs on Hugging Face, read remotely), official event split, patches and sampling.

Data layout on Hugging Face (`orion-ai-lab/FLOGA-GeoTIFFs`, folder sen2_20_mod_500, 62 tar shards):
    <year>/<ID>_<row>_SEN2_2a_20_{before,after,label,clc_mask,sea_mask_before,sea_mask_after}.tiff
    <year>/<ID>_<row>_SEN2_2a_cloud_mask_20_{before,after}.tiff
S2 images: 10 bands B01,B02,B03,B04,B05,B06,B07,B11,B12,B8A, uint16, nodata 65535, processing baseline N0500
(BOA offset +1000). We drop B01 -> exactly the FLOGA-v1 20 m band order, subtract the offset (v1 imagery and the
2021 Planetary-Computer scenes have no offset), and map nodata to 0.
Cloud masks are Sentinel-2 SCL (0 = no data, 9 = cloud high probability). Label: 0 unburnt, 1 burnt, 2 burnt by
another fire of the same year (ignored). Sea mask -> binary with FLOGA's rule: values 2 and 4 -> land, other
non-zero -> sea/water (create_dataset.py).
"""
from __future__ import annotations

import glob
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import load_config, ROOT
from .remote_tiff import HF_BASE, RemoteTiff

V2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "B8A"]
FLOGA_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "B8A"]
BAND_IDX = [V2_BANDS.index(b) for b in FLOGA_BANDS]
S2_NODATA = 65535
CLOUD_SCL = 9

FILES = {
    "pre": "SEN2_2a_20_before", "post": "SEN2_2a_20_after", "label": "SEN2_2a_20_label",
    "clc": "SEN2_2a_20_clc_mask", "sea": "SEN2_2a_20_sea_mask_before",
    "cloud_pre": "SEN2_2a_cloud_mask_20_before", "cloud_post": "SEN2_2a_cloud_mask_20_after",
}


# ---------------------------------------------------------------- index / event mapping
@lru_cache(None)
def load_index(index_dir=None) -> pd.DataFrame:
    index_dir = index_dir or load_config()["paths"]["floga_index"]
    rows = []
    for f in sorted(glob.glob(str(index_dir) + "/*.csv")):
        d = pd.read_csv(f)
        d["shard"] = re.sub(r"\.csv$", "", f.replace("\\", "/").split("/")[-1]).replace("__", "/")
        rows.append(d)
    idx = pd.concat(rows, ignore_index=True)
    m = idx["name"].str.extract(r"^(\d{4})/(\d+)_(\d+)_(.+)\.tiff?$")
    idx["year"], idx["fid"], idx["row"], idx["kind"] = m[0], m[1], m[2], m[3]
    idx = idx.dropna(subset=["year"])
    idx["key"] = idx["year"] + "/" + idx["fid"] + "_" + idx["row"]
    return idx


def _tile(s):
    m = re.search(r"_T(\d\d[A-Z]{3})_", str(s))
    return m.group(1) if m else None


def _baseline(s):
    m = re.search(r"_N(\d{4})_", str(s))
    return int(m.group(1)) if m else None


def event_table(cfg=None) -> pd.DataFrame:
    """Official FLOGA split (v1 rows) mapped to v2 file keys.

    v1 split: (year, event_id) with event_id = row of the v1 annotation shapefile of that year.
    v1 row -> (fire ID, S2 tile) -> first v2 shapefile row with the same (ID, tile) -> v2 key "<year>/<ID>_<row>".
    Rows without a tile (2017 Sentinel-Hub imagery) or without an exact match are matched on ID alone when the v2
    data contain exactly one image for that ID. A v2 key claimed by several splits is kept only in the most
    protected one (test > val > train) so that no image is shared between splits.
    """
    import geopandas as gpd
    cfg = cfg or load_config()
    ann = ROOT / "external" / "FLOGA-annotations" / "polygons"
    split = pd.read_csv(cfg["paths"]["floga_split"])
    idx = load_index()
    present = set(idx.loc[idx["kind"] == FILES["label"], "key"])
    out = []
    for y in sorted(split["year"].unique()):
        g1 = gpd.read_file(ann / "v1" / f"fb_{y}_final_images_4326.shp")
        g2 = gpd.read_file(ann / "v2" / f"fb_{y}_final_images_4326.shp")
        g1["tile"] = g1["SEN_2A_s"].map(_tile)
        g2["tile"] = g2["S2_s"].map(_tile)
        g2 = g2.reset_index().rename(columns={"index": "row"})
        g2["key"] = f"{y}/" + g2["ID"].astype(str) + "_" + g2["row"].astype(str)
        first = g2.groupby(["ID", "tile"])["row"].min()
        by_id = g2[g2["key"].isin(present)].groupby("ID")["row"].unique()
        for _, r in split[split["year"] == y].iterrows():
            v1 = g1.loc[int(r["event_id"])]
            row, how = None, None
            if v1["tile"] is not None and (v1["ID"], v1["tile"]) in first.index:
                row, how = int(first[(v1["ID"], v1["tile"])]), "id+tile"
            elif v1["ID"] in by_id.index and len(by_id[v1["ID"]]) == 1:
                row, how = int(by_id[v1["ID"]][0]), "id-only"
            key = f"{y}/{v1['ID']}_{row}" if row is not None else None
            g2r = g2[g2["row"] == row].iloc[0] if row is not None else None
            out.append({"year": y, "event_id": int(r["event_id"]), "set": r["set"], "fire_id": str(v1["ID"]),
                        "tile": v1["tile"], "key": key if key in present else None, "match": how,
                        "s2_pre": g2r["S2_s"] if g2r is not None else None,
                        "s2_post": g2r["S2_e"] if g2r is not None else None,
                        "start_date": v1["Start date"]})
    ev = pd.DataFrame(out)
    ev["baseline_pre"] = ev["s2_pre"].map(_baseline)
    ev["baseline_post"] = ev["s2_post"].map(_baseline)
    prio = {"test": 0, "val": 1, "train": 2}
    ev["prio"] = ev["set"].map(prio)
    ev = ev.sort_values(["prio", "year", "event_id"])
    ev["dup_dropped"] = ev["key"].notna() & ev.duplicated("key")
    return ev.drop(columns="prio").sort_values(["year", "event_id"]).reset_index(drop=True)


def files_for(key):
    idx = load_index()
    sub = idx[idx["key"] == key].set_index("kind")
    return {k: (HF_BASE + sub.loc[v, "shard"], int(sub.loc[v, "offset"]), int(sub.loc[v, "size"]))
            for k, v in FILES.items() if v in sub.index}


def open_file(key, what):
    url, off, size = files_for(key)[what]
    return RemoteTiff(url, off, size)


# ---------------------------------------------------------------- patches (create_dataset.py rules)
def padding_offset(img_size, out_size):
    """Exact port of FLOGA create_dataset.get_padding_offset -> (top, bot, left, right)."""
    pads = []
    for s, o in zip(img_size, out_size):
        p = int(o - s % o) if s >= o else o - s
        if p != o:
            a = b = p // 2
            if p % 2:
                a += 1
        else:
            a = b = 0
        pads.append((a, b))
    return pads[0][0], pads[0][1], pads[1][0], pads[1][1]


def patch_grid(shape, size=256):
    """Top-left corners (in unpadded image coordinates, may be negative) of the non-overlapping patches."""
    top, bot, left, right = padding_offset(shape, (size, size))
    H, W = shape[0] + top + bot, shape[1] + left + right
    if (shape[0] % size == 0) and (shape[1] % size == 0):
        top = left = 0
        H, W = shape
    return [(r - top, c - left) for r in range(0, H, size) for c in range(0, W, size)]


def crop(a, r, c, size=256, fill=0):
    """Crop (..., H, W) array at (r, c) with zero fill outside."""
    H, W = a.shape[-2:]
    out = np.full(a.shape[:-2] + (size, size), fill, a.dtype)
    r0, r1, c0, c1 = max(r, 0), min(r + size, H), max(c, 0), min(c + size, W)
    if r0 < r1 and c0 < c1:
        out[..., r0 - r: r1 - r, c0 - c: c1 - c] = a[..., r0:r1, c0:c1]
    return out


def sea_binary(sea):
    s = np.asarray(sea).copy()
    s[(s == 2) | (s == 4)] = 0
    return (s != 0).astype(np.uint8)


def to_dn(raw, offset):
    """v2 raw uint16 (nodata 65535, BOA offset) -> FLOGA-v1-like DN (0 = no data)."""
    raw = raw[BAND_IDX] if raw.shape[0] == len(V2_BANDS) else raw
    nod = np.any(raw == S2_NODATA, axis=0) | np.all(raw == 0, axis=0)
    dn = raw.astype(np.int32) - int(offset)
    dn = np.clip(dn, 1, 65534).astype(np.uint16)
    dn[:, nod] = 0
    return dn


# ---------------------------------------------------------------- cached patches
def patch_dir(split, cfg=None):
    cfg = cfg or load_config()
    return cfg["paths"]["cache"] / "floga" / "patches" / split


def list_patches(split, cfg=None):
    return sorted(patch_dir(split, cfg).glob("*.npz"))


def load_patch(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def pixel_mask(p):
    """Pixels used for evaluation/sampling: image data in both dates and label != 2."""
    return (p["label"] != 2) & np.all(p["pre"] > 0, 0) & np.all(p["post"] > 0, 0)


def test_reference(cfg=None):
    """Canonical flattened reference of the FLOGA test split (patch order = sorted file names)."""
    refs = []
    files = list_patches("test", cfg)
    if not files:
        raise FileNotFoundError("FLOGA test patches not prepared")
    for f in files:
        p = load_patch(f)
        refs.append(p["label"][pixel_mask(p)] == 1)
    return np.concatenate(refs)


def iter_split_pixels(split, cfg=None):
    """Yield (patch_name, pre, post, label, mask) for every patch of a split."""
    for f in list_patches(split, cfg):
        p = load_patch(f)
        yield f.stem, p, pixel_mask(p)
