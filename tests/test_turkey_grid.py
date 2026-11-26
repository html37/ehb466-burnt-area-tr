"""Mosaic/grid alignment checks on the cached Turkish stacks (skipped if the cache is not built)."""
import json

import numpy as np
import pytest

from burnt.config import load_config

cfg = load_config()
TR = cfg["paths"]["cache"] / "turkey"


@pytest.mark.parametrize("event", list(cfg["events"]))
def test_stack_shapes_and_grid(event):
    if not (TR / f"{event}.npz").exists():
        pytest.skip("Turkish cache not built")
    z = np.load(TR / f"{event}.npz")
    meta = json.loads((TR / f"{event}.json").read_text())
    H, W = meta["shape"]
    assert z["pre"].shape == z["post"].shape == (len(cfg["bands"]), H, W)
    for k in ["pre_scl", "post_scl", "valid", "effis"]:
        assert z[k].shape == (H, W)
    assert meta["bands"] == cfg["bands"]
    assert abs(meta["res_m"] - 20) < 1e-6
    assert abs(meta["transform"][0] - 20) < 1e-6 and abs(meta["transform"][4] + 20) < 1e-6
    assert z["effis"].sum() > 0 and 0.5 < z["valid"].mean() <= 1
    # valid pixels must have data in every band of both dates
    assert np.all(z["pre"][:, z["valid"]] > 0) and np.all(z["post"][:, z["valid"]] > 0)
