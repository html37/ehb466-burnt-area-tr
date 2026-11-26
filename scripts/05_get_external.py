"""Fetch all external code/metadata/weights (resumable): FLOGA repo, FLOGA-annotations, official split CSV,
BAM-CD v1/v2 weights, and the Hugging Face tar index of FLOGA v2 (20 m).

Usage: python scripts/05_get_external.py [--no-v2-weights]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOGA_COMMIT = "0efa4b37ca665259a69273b33b0d9f9e91421d3f"
SPLIT = "https://www.dropbox.com/scl/fi/vq3tl8w5ex23lt1k7z89e/data_split.csv?rlkey=v3ph1xvfykhiljkg6rzlsytq2&dl=1"
WEIGHTS = {
    "v1": "https://www.dropbox.com/scl/fo/7boya0nvjb0sgmg9l5quo/ALldNkZ5d-DMtUMmUPm5O4A?rlkey=1p29xzb912mj4hh68k8waac9x&st=rkk9ke5g&dl=1",
    "v2": "https://www.dropbox.com/scl/fo/8bhch7e0yhwgfqxsc6i69/ANcImDZyq7EYHVPt66aYuq8?rlkey=oiz26v6r09t4bpr5fize3a9a8&st=4qo50vqh&dl=1",
}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=900).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-v2-weights", action="store_true")
    a = ap.parse_args()
    ext = ROOT / "external"; ext.mkdir(exist_ok=True)
    if not (ext / "FLOGA").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/Orion-AI-Lab/FLOGA", str(ext / "FLOGA")], check=True)
        subprocess.run(["git", "-C", str(ext / "FLOGA"), "checkout", "-q", FLOGA_COMMIT], check=True)
    if not (ext / "FLOGA-annotations").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/Orion-AI-Lab/FLOGA-annotations",
                        str(ext / "FLOGA-annotations")], check=True)
    f = ROOT / "data" / "raw" / "floga" / "data_split.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(get(SPLIT))
    for v, url in WEIGHTS.items():
        if v == "v2" and a.no_v2_weights:
            continue
        d = ROOT / "data" / "raw" / "bamcd" / v
        if not (d / "best_segmentation.pt").exists():
            d.mkdir(parents=True, exist_ok=True)
            print(f"downloading BAM-CD {v} weights...", flush=True)
            with zipfile.ZipFile(io.BytesIO(get(url))) as z:
                for m in z.namelist():
                    if not m.endswith("/"):
                        (d / Path(m).name).write_bytes(z.read(m))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "hf_index_all.py")], check=True, cwd=ROOT)
    print("external resources ready")


if __name__ == "__main__":
    main()
