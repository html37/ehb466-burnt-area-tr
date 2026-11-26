"""Cost measurement: wall time, peak resident memory (sampled), GPU memory, file size, package versions."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager

import psutil


class PeakRSS:
    """Sample the process RSS every `dt` seconds in a background thread."""

    def __init__(self, dt=0.05):
        self.dt, self.peak, self._stop = dt, 0, threading.Event()
        self.proc = psutil.Process(os.getpid())

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.dt)

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self.t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


@contextmanager
def measure(result: dict, prefix=""):
    """with measure(res, 'train_'): ... -> res['train_time_s'], res['train_peak_rss_mb'], res['train_delta_rss_mb']"""
    with PeakRSS() as pr:
        t = time.perf_counter()
        yield
        result[f"{prefix}time_s"] = time.perf_counter() - t
    result[f"{prefix}peak_rss_mb"] = pr.peak / 2 ** 20
    result[f"{prefix}delta_rss_mb"] = (pr.peak - pr.base) / 2 ** 20


def versions(pkgs=("numpy", "pandas", "scikit-learn", "lightgbm", "optuna", "torch", "rasterio", "xarray",
                   "rioxarray", "geopandas", "segmentation-models-pytorch", "pystac-client", "planetary-computer")):
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "cpu": platform.processor(), "cpu_count": os.cpu_count(),
           "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1)}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def write_versions(path):
    with open(path, "w") as f:
        json.dump(versions(), f, indent=1)
