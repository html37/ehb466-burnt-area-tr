"""Spectral indices on surface reflectance (0-1) of the FLOGA 20 m bands.

Band roles at 20 m (FLOGA has no B08 at 20 m, so B8A is the NIR band everywhere):
blue=B02, green=B03, red=B04, re1=B05, re2=B06, re3=B07, nir=B8A, swir1=B11, swir2=B12.
Differenced index dSI = SI_pre - SI_post (FLOGA paper, eq. 1).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _nd(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def nbr(nir, swir2):
    """Normalized Burn Ratio (Key & Benson 2006)."""
    return _nd(nir, swir2)


def nbr2(swir1, swir2):
    return _nd(swir1, swir2)


def ndvi(nir, red):
    return _nd(nir, red)


def ndwi(green, nir):
    """NDWI after McFeeters (1996): (Green - NIR) / (Green + NIR), as in FLOGA Table IV."""
    return _nd(green, nir)


def bais2(red, re2, re3, nir, swir2):
    """Burned Area Index for Sentinel-2 (Filipponi 2018)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1 - np.sqrt(np.clip(re2 * re3 * nir / np.maximum(red, EPS), 0, None))) * \
               ((swir2 - nir) / np.sqrt(np.maximum(swir2 + nir, EPS)) + 1)


def mirbi(swir1, swir2):
    """Mid-Infrared Burn Index (Trigg & Flasse 2001): 10*SWIR_long - 9.8*SWIR_short + 2, reflectance 0-1.

    SWIR_long = B12 (2.2 um), SWIR_short = B11 (1.6 um), following the original definition.
    """
    return 10 * swir2 - 9.8 * swir1 + 2


def band_dict(stack, bands):
    """Channel-first stack (n_bands, ...) -> {band name: array}."""
    return {b: stack[i] for i, b in enumerate(bands)}


INDEX_NAMES = ["NBR", "NBR2", "NDVI", "NDWI", "BAIS2", "MIRBI"]


def all_indices(b: dict) -> dict:
    """All single-date indices from a band dict (reflectance 0-1)."""
    return {
        "NBR": nbr(b["B8A"], b["B12"]),
        "NBR2": nbr2(b["B11"], b["B12"]),
        "NDVI": ndvi(b["B8A"], b["B04"]),
        "NDWI": ndwi(b["B03"], b["B8A"]),
        "BAIS2": bais2(b["B04"], b["B06"], b["B07"], b["B8A"], b["B12"]),
        "MIRBI": mirbi(b["B11"], b["B12"]),
    }
