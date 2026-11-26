"""EFFIS reference perimeters: loading, AOI derivation and rasterisation."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import box


def load_effis(path) -> gpd.GeoDataFrame:
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:3035")
    crs = crs.replace("urn:ogc:def:crs:", "").replace("::", ":")
    effis = gpd.GeoDataFrame.from_features(gj["features"], crs=crs)
    for col in ["initialdate", "finaldate"]:
        effis[col] = pd.to_datetime(effis[col].astype(str).str[:10].str.replace("/", "-"))
    return effis


def event_bbox(effis, effis_id, buffer_m=3000):
    """AOI = bounding box (EPSG:4326) of the EFFIS polygon buffered by `buffer_m`."""
    poly = effis[effis["id"] == effis_id]
    if len(poly) != 1:
        raise KeyError(f"EFFIS id {effis_id} not found")
    return [float(v) for v in poly.buffer(buffer_m).to_crs(4326).total_bounds]


def effis_mask(effis, ref, pre_date, post_date):
    """Rasterise every EFFIS fire that started between the pre and post image dates onto the grid of `ref`."""
    sel = effis[(effis["initialdate"] >= pre_date) & (effis["initialdate"] <= post_date)].to_crs(ref.rio.crs)
    sel = sel[sel.intersects(box(*ref.rio.bounds()))]
    m = features.rasterize(((g, 1) for g in sel.geometry), out_shape=ref.shape,
                           transform=ref.rio.transform(), fill=0, dtype="uint8")
    return m.astype(bool), sel
