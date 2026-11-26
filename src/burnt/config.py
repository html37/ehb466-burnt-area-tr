"""Configuration loading and repository paths."""
from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(path: str | pathlib.Path | None = None) -> dict:
    """Read configs/default.yaml (or `path`) and resolve relative paths against the repository root."""
    path = pathlib.Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def out_dir(*parts: str) -> pathlib.Path:
    """Return (and create) a directory under outputs/."""
    p = ROOT.joinpath("outputs", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
