"""BAM-CD benchmark: official code + pretrained weights (v1 = paper model, v2 = newer release).

1. Sanity check on the FLOGA test split (patch-wise inference, 256x256, threshold 0.5 = argmax) -> must be close
   to the paper's F1 (88.48, Table V) before the model is used on Türkiye.
2. Inference on the Turkish rasters (sliding window 256, stride 128, blended), inference only - nothing is tuned.
3. Cost: parameters, checkpoint size, GPU and CPU inference time per 1000 km^2, peak RAM / VRAM.

Outputs: outputs/predictions/{floga_test,turkey}/bamcd_<v>*, outputs/tables/bamcd_sanity.csv, cost_bamcd.csv
Usage: python scripts/60_run_bamcd.py [--versions v1 v2] [--skip-cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir  # noqa: E402
from burnt import bamcd, floga, turkey, metrics, predictions as P, cost  # noqa: E402

PAPER_F1 = {"v1": 0.8848}


def floga_split(model, device, split, cfg):
    preds, refs = [], []
    files = floga.list_patches(split, cfg)
    for i in range(0, len(files), 8):
        ps = [floga.load_patch(f) for f in files[i:i + 8]]
        prob = bamcd.predict_patches(model, device, np.stack([p["pre"] for p in ps]).astype(np.float32),
                                     np.stack([p["post"] for p in ps]).astype(np.float32))
        for p, pr in zip(ps, prob):
            m = floga.pixel_mask(p)
            preds.append(pr[m] > 0.5); refs.append(p["label"][m] == 1)
    return np.concatenate(preds), np.concatenate(refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v1", "v2"])
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    sanity, costs = [], []
    for v in args.versions:
        name = f"bamcd_{v}"
        model, device = bamcd.load(v)
        for split in ["test", "val"]:
            if not floga.list_patches(split, cfg):
                continue
            pred, ref = floga_split(model, device, split, cfg)
            m = metrics.metrics(pred, ref)
            sanity.append({"method": name, "split": split, **m, "paper_f1_test": PAPER_F1.get(v)})
            print(name, split, {k: round(m[k], 4) for k in ["precision", "recall", "f1", "iou"]}, flush=True)
            if split == "test":
                P.save_floga_test(name, pred)
        for ev_name in turkey.event_names(cfg):
            ev = turkey.load_event(ev_name, cfg)
            pre, post = ev["pre"].astype(np.float32), ev["post"].astype(np.float32)
            area_km2 = ev["valid"].size * ev["px_ha"] / 100
            for dev in (["cuda"] if device == "cuda" else []) + ([] if args.skip_cpu else ["cpu"]):
                model.to(dev)
                if dev == "cuda":
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                res = {"method": name, "event": ev_name, "device": dev, "area_km2": area_km2}
                with cost.measure(res, "infer_"):
                    prob = bamcd.predict_raster(model, dev, pre, post)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                if dev == "cuda":
                    res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
                res["infer_s_per_1000km2"] = res["infer_time_s"] / area_km2 * 1000
                costs.append(res)
                print(res, flush=True)
            P.save_turkey(name, ev_name, (prob > 0.5) & ev["valid"], prob=prob)
            model.to(device)
        ck = bamcd.WEIGHTS[v]
        costs.append({"method": name, "event": "model", "params": bamcd.n_params(model),
                      "checkpoint_mb": ck.stat().st_size / 2 ** 20,
                      "weights_fp32_mb": bamcd.n_params(model) * 4 / 2 ** 20})
        del model; torch.cuda.empty_cache()
    out = out_dir("tables")
    if sanity:
        pd.DataFrame(sanity).to_csv(out / "bamcd_sanity.csv", index=False)
    pd.DataFrame(costs).to_csv(out / "cost_bamcd.csv", index=False)
    cost.write_versions(out_dir("runs", "bamcd") / "versions.json")


if __name__ == "__main__":
    main()
