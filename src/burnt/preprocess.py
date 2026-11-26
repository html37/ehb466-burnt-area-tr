"""Reflectance scaling, SCL masks and resampling to the common 20 m grid."""
from __future__ import annotations

import numpy as np
from rasterio.enums import Resampling

from .stac import read_group


def boa_offset(item) -> float:
    """Processing baseline >= 04.00 (products after 25 Jan 2022 or reprocessed) carry a -1000 BOA offset."""
    try:
        return -1000.0 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def read_dn(group, asset, bbox, ref=None):
    """Mosaic one band of a pass, apply the BOA offset, return float32 DN (reflectance x 10000), NaN = no data.

    If `ref` is given the band is resampled onto its grid (average for 10 m -> 20 m, as in notebook 01).
    """
    raw = read_group(group["items"], asset, bbox)
    da = (raw.astype("float32") + boa_offset(group["items"][0])).where(raw != 0)
    da = da.rio.write_nodata(np.nan)
    if ref is not None and (da.shape != ref.shape or da.rio.transform() != ref.rio.transform()):
        da = da.rio.reproject_match(ref, resampling=Resampling.average)
    return da


def reflectance(group, asset, bbox, ref=None):
    """Surface reflectance (0-1) of one band, exactly as notebook 01 (`(DN + offset) / 10000`)."""
    return read_dn(group, asset, bbox, ref) / 10000.0


def scl_on_grid(scl, ref):
    return scl.rio.reproject_match(ref, resampling=Resampling.nearest).values


def valid_mask(pre_scl, post_scl, valid_classes, *finite_arrays):
    v = np.isin(pre_scl, valid_classes) & np.isin(post_scl, valid_classes)
    for a in finite_arrays:
        v &= np.isfinite(a)
    return v


def floga_scale(dn, clamp=10000.0):
    """FLOGA input scaling (`clamp_scale_10000`): min(DN, 10000) / 10000; NaN -> 0 (FLOGA nan_value)."""
    x = np.minimum(np.nan_to_num(dn, nan=0.0), clamp) / clamp
    return x.astype(np.float32)
