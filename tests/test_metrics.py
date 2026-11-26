import numpy as np

from burnt import metrics


def test_basic_counts():
    pred = np.array([1, 1, 0, 0, 1], bool)
    ref = np.array([1, 0, 1, 0, 1], bool)
    m = metrics.metrics(pred, ref)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["f1"], 2 / 3) and np.isclose(m["iou"], 0.5) and np.isclose(m["oa"], 0.6)


def test_valid_mask_excludes_pixels():
    pred = np.array([1, 1, 0], bool); ref = np.array([1, 0, 0], bool); valid = np.array([1, 0, 1], bool)
    m = metrics.metrics(pred, ref, valid)
    assert m["fp"] == 0 and m["f1"] == 1.0


def test_empty_prediction_is_zero_not_nan():
    m = metrics.metrics(np.zeros(4, bool), np.array([1, 0, 0, 0], bool))
    assert m["f1"] == 0.0 and m["precision"] == 0.0


def test_block_bootstrap_identical_methods_zero_delta():
    rng = np.random.default_rng(0)
    ref = rng.random((100, 100)) > 0.7
    pred = ref ^ (rng.random((100, 100)) > 0.9)
    valid = np.ones_like(ref)
    blocks = metrics.block_ids(ref.shape, 20, 1000)  # 50 px blocks -> 4 blocks
    assert len(np.unique(blocks)) == 4
    _, c = metrics.block_counts({"a": pred, "b": pred}, ref, valid, blocks)
    res = metrics.block_bootstrap(c, 200, pairs=[("a", "b")])
    assert res["a - b"]["lo"] == 0 and res["a - b"]["hi"] == 0
    full = metrics.metrics(pred, ref)["f1"]
    assert res["a"]["f1_lo"] <= full <= res["a"]["f1_hi"]
    assert c["a"].sum() == ref.size


def test_block_counts_sum_matches_confusion():
    rng = np.random.default_rng(1)
    ref = rng.random((60, 70)) > 0.5; pred = rng.random((60, 70)) > 0.5; valid = rng.random((60, 70)) > 0.2
    _, c = metrics.block_counts({"p": pred}, ref, valid, metrics.block_ids(ref.shape, 20, 300))
    assert tuple(c["p"].sum(0)) == metrics.confusion(pred, ref, valid)
