import numpy as np

from burnt import indices
from burnt.dnbr import dnbr, otsu, grid_counts
from burnt import metrics


def test_nbr_and_dnbr_values():
    assert np.isclose(indices.nbr(0.3, 0.1), 0.5)
    assert np.isclose(dnbr(0.3, 0.1, 0.1, 0.3), 1.0)  # healthy (0.5) -> burnt (-0.5)


def test_index_formulas():
    b = {"B02": 0.05, "B03": 0.08, "B04": 0.06, "B05": 0.1, "B06": 0.2, "B07": 0.25, "B8A": 0.3,
         "B11": 0.2, "B12": 0.1}
    s = indices.all_indices({k: np.float64(v) for k, v in b.items()})
    assert np.isclose(s["NDVI"], (0.3 - 0.06) / 0.36)
    assert np.isclose(s["NDWI"], (0.08 - 0.3) / 0.38)       # McFeeters, FLOGA Table IV
    assert np.isclose(s["NBR2"], (0.2 - 0.1) / 0.3)
    assert np.isclose(s["MIRBI"], 10 * 0.1 - 9.8 * 0.2 + 2)
    exp = (1 - np.sqrt(0.2 * 0.25 * 0.3 / 0.06)) * ((0.1 - 0.3) / np.sqrt(0.4) + 1)
    assert np.isclose(s["BAIS2"], exp)


def test_otsu_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0.0, 0.05, 5000), rng.normal(0.6, 0.05, 5000)])
    t = otsu(v)
    assert 0.15 < t < 0.45


def test_grid_counts_matches_direct():
    rng = np.random.default_rng(2)
    v = rng.normal(size=1000); ref = rng.random(1000) > 0.6
    thr = [-0.5, 0.0, 0.3]
    gc = grid_counts(v, ref, thr)
    for t, row in zip(thr, gc):
        assert tuple(row) == metrics.confusion(v > t, ref)
