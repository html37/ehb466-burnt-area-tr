# Hardware used for all experiments

Probed 2026-11-26 (Phase 0).

| Item | Value |
|---|---|
| OS | Windows 10 Home 10.0.19045 (64-bit) |
| CPU | AMD Ryzen 7 6800H, 8 cores / 16 threads |
| RAM | 16 GB (16 371 617 792 bytes) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM, driver 566.07, CUDA 12.7 |
| Free disk | ~94 GB on C: (477 GB total) |
| Python | 3.11.9 in `.venv` (created with `py -3.11 -m venv .venv`) |

Implications for job sizing:
- FLOGA-v2 20 m on Hugging Face is 1.1 TB (62 tar shards) -> cannot be stored; only the needed 256x256 patches are
  read remotely (byte-range reads of the uncompressed GeoTIFFs inside the tars), see `docs/floga_notes.md`.
- BAM-CD (ResNet-101 U-Net, ~84 M params) inference fits in 4 GB VRAM with batch 1-4 at 256x256.
- LightGBM/RF training: up to a few million samples x ~50 features fits in 16 GB RAM; use n_jobs=8.
