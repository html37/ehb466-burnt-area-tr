"""Check-labelling kit (Phase 7): stratified random points for independent visual labelling by the team.

Per event 150 points on valid pixels: 50 inside the EFFIS perimeter, 50 in the 0-1 km ring outside it, 50 in the
1-3 km ring. The `label` column is left empty (to be filled with 0/1 by a human; never filled by code).
Outputs: labeling/points.csv, labeling/points.geojson
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, ROOT  # noqa: E402
from burnt import turkey  # noqa: E402

N_PER_STRATUM = 50


def main():
    cfg = load_config()
    rng = np.random.default_rng(2021)
    rows = []
    for name in turkey.event_names(cfg):
        ev = turkey.load_event(name, cfg)
        m = ev["meta"]; tr = Affine(*m["transform"])
        inside = ev["effis"]
        dist = ndimage.distance_transform_edt(~inside) * m["res_m"]
        strata = {"inside_effis": inside, "ring_0_1km": (~inside) & (dist <= 1000),
                  "ring_1_3km": (~inside) & (dist > 1000) & (dist <= 3000)}
        for s, mask in strata.items():
            idx = np.flatnonzero((mask & ev["valid"]).ravel())
            pick = rng.choice(idx, min(N_PER_STRATUM, len(idx)), replace=False)
            r, c = np.unravel_index(pick, mask.shape)
            x, y = tr * (c + 0.5, r + 0.5)
            for k in range(len(pick)):
                rows.append({"event": name, "stratum": s, "row": int(r[k]), "col": int(c[k]),
                             "x": float(x[k]), "y": float(y[k])})
        meta = m
    df = pd.DataFrame(rows)
    out = []
    for name, g in df.groupby("event", sort=False):
        crs = turkey.load_event(name, cfg)["meta"]["crs"]
        gg = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.x, g.y), crs=crs).to_crs(4326)
        gg["lon"] = gg.geometry.x.round(6); gg["lat"] = gg.geometry.y.round(6)
        mt = turkey.load_event(name, cfg)["meta"]
        gg["pre_date"] = mt["pre_date"]; gg["post_date"] = mt["post_date"]
        gg["copernicus_browser"] = [
            f"https://browser.dataspace.copernicus.eu/?zoom=16&lat={la}&lng={lo}" for la, lo in zip(gg.lat, gg.lon)]
        out.append(gg)
    pts = pd.concat(out, ignore_index=True)
    pts = pts.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle so strata are not obvious
    pts.insert(0, "point_id", [f"P{i:03d}" for i in range(len(pts))])
    pts["label"] = ""
    pts["confidence"] = ""
    pts["comment"] = ""
    cols = ["point_id", "event", "lon", "lat", "pre_date", "post_date", "copernicus_browser", "label",
            "confidence", "comment"]
    (ROOT / "labeling").mkdir(exist_ok=True)
    out_csv = ROOT / "labeling" / "points.csv"
    if out_csv.exists() and pd.read_csv(out_csv)["label"].notna().any():
        print("labeling/points.csv already contains labels - not overwritten"); return
    pts[cols].to_csv(out_csv, index=False)
    # strata and grid positions are kept apart so that the labellers do not see them (blind labelling)
    pts[["point_id", "event", "stratum", "row", "col"]].to_csv(ROOT / "labeling" / "points_key.csv", index=False)
    gpd.GeoDataFrame(pts[cols], geometry=pts.geometry, crs=4326).to_file(
        ROOT / "labeling" / "points.geojson", driver="GeoJSON")
    print(pts.groupby(["event", "stratum"]).size())


if __name__ == "__main__":
    main()
