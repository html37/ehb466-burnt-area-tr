"""Sentinel-2 L2A access through the Microsoft Planetary Computer STAC API.

Logic is identical to notebooks/01_dnbr_benchmark.ipynb: search, group tiles of the same pass (same day +
same platform), mosaic the tiles that cover the AOI, pick the clearest scene closest to the fire.
"""
from __future__ import annotations

import numpy as np
import pystac_client
import planetary_computer
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rioxarray.merge import merge_arrays
from shapely.geometry import box, shape
from shapely.ops import unary_union

_CATALOG = None


def catalog(url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"):
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = pystac_client.Client.open(url, modifier=planetary_computer.sign_inplace)
    return _CATALOG


def search(bbox, dates, collection="sentinel-2-l2a", max_cloud=20, url=None):
    s = catalog(url or "https://planetarycomputer.microsoft.com/api/stac/v1").search(
        collections=[collection], bbox=bbox, datetime=f"{dates[0]}/{dates[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}})
    return list(s.items())


def coverage(item, aoi_box) -> float:
    return shape(item.geometry).intersection(aoi_box).area / aoi_box.area


def read_asset(item, asset, bbox):
    """Read only the AOI window of a COG asset (bbox in EPSG:4326)."""
    da = rioxarray.open_rasterio(item.assets[asset].href).squeeze("band", drop=True)
    return da.rio.clip_box(*bbox, crs="EPSG:4326").rio.write_nodata(0)


def read_group(items, asset, bbox):
    """Read one asset for all tiles of a pass and mosaic them (first = largest AOI coverage wins)."""
    aoi_box = box(*bbox)
    parts = []
    for it in sorted(items, key=lambda i: -coverage(i, aoi_box)):
        try:
            da = read_asset(it, asset, bbox).load()
        except Exception:  # tile does not intersect the AOI
            continue
        if parts and da.rio.crs != parts[0].rio.crs:  # other UTM zone -> reproject to the first
            da = da.rio.reproject(parts[0].rio.crs, resolution=abs(parts[0].rio.resolution()[0]),
                                  nodata=0, resampling=Resampling.nearest)
        parts.append(da)
    return parts[0] if len(parts) == 1 else merge_arrays(parts, nodata=0)


def group_by_pass(items, aoi_box):
    groups = {}
    for it in items:
        groups.setdefault((it.datetime.date(), it.properties.get("platform", "")), []).append(it)
    out = []
    for (day, plat), its in groups.items():
        cov = unary_union([shape(i.geometry) for i in its]).intersection(aoi_box).area / aoi_box.area
        out.append(dict(day=day, platform=plat, items=its, coverage=cov))
    return out


def pick_scene(items, bbox, scl_clear, latest_first, min_clear=0.97, max_candidates=6):
    """Starting from the date closest to the fire, return the first pass that covers the AOI and is clear."""
    aoi_box = box(*bbox)
    cands = sorted(group_by_pass(items, aoi_box), key=lambda g: g["day"], reverse=latest_first)[:max_candidates]
    best = None
    for g in cands:
        if g["coverage"] < 0.5:
            continue
        scl = read_group(g["items"], "SCL", bbox)
        g["scl"] = scl
        x0, y0, x1, y1 = transform_bounds("EPSG:4326", scl.rio.crs, *bbox)
        rx, ry = scl.rio.resolution()
        g["clear"] = min(float(np.isin(scl.values, scl_clear).sum()) / ((x1 - x0) * (y1 - y0) / abs(rx * ry)), 1.0)
        if best is None or g["clear"] > best["clear"]:
            best = g
        if g["clear"] >= min_clear:
            break
    if best is None:
        raise RuntimeError("no suitable Sentinel-2 scene found")
    return best
