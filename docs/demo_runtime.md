# Demo runtime (`notebooks/00_demo_end_to_end.ipynb` → `scripts/demo_run.py`)

Requirement (course 6.3): main demo ≤ 30 min on Colab free tier.

## Measured (fresh `git clone`, empty data cache, both events, BAM-CD enabled)

Machine: AMD Ryzen 7 6800H (8 cores / 16 threads), 16 GB RAM, RTX 3050 Laptop 4 GB (`docs/hardware.md`).
Every run starts from an empty `data/cache/` and downloads the Sentinel-2 scenes from Planetary Computer again.
The GPU run also downloaded the BAM-CD v1 weights (1.0 GB) and cloned the FLOGA model code.

| Step (s) | GPU run (`logs/demo_clean_run.log`) | CPU-only run, `CUDA_VISIBLE_DEVICES=-1` (`logs/demo_clean_run_cpu.log`) |
|---|---|---|
| data access + preprocessing | 93.2 | 94.3 |
| Manavgat: LightGBM | 6.0 | 7.6 |
| Manavgat: BAM-CD | 10.8 (stride 128) | 42.4 (stride 256) |
| Marmaris: LightGBM | 1.2 | 1.2 |
| Marmaris: BAM-CD | 2.5 (stride 128) | 10.2 (stride 256) |
| **total (script)** | **179.6** | **174.1** |
| wall clock | 184 | 186 |

The demo metrics match the main evaluation (e.g. LightGBM seed 0: Manavgat F1 0.8828, Marmaris F1 0.8603).
On CPU the default BAM-CD stride is 256, i.e. no window overlap. This makes BAM-CD F1 slightly different from
the overlapped stride-128 result (Manavgat 0.9076 vs 0.9215, Marmaris 0.8751 vs 0.8665). The paper reports
the stride-128 numbers.

## Colab estimate

The free Colab CPU VM has 2 vCPUs, while the measured machine has 16 threads. The compute-bound steps took about
62 s on this machine. Assuming the work scales with core count (an upper-bound assumption, about 8× slower):

- about 62 × 8 ≈ 8 min of compute;
- plus about 1.5 min of data access, plus about 1–2 min for the 1.0 GB weight download and `pip install`;
- total: well under 30 min.

With a Colab GPU the run takes about 3 min, like the GPU run above.
Use `--bamcd off` to skip BAM-CD entirely (dNBR + LightGBM only, about 2 min).

## Note on emulating 2 cores on Windows

We tried two ways to emulate 2 cores on Windows (local test machine only, not Colab/Linux):

- `OMP_NUM_THREADS=2`
- pinning the process to 2 logical CPUs (`start /affinity 3`)

Both made the CPU run crash with a segfault (exit 139) inside native torch/LightGBM threading, right after BAM-CD
loaded. The unpinned CPU run above works, so the 2-vCPU figure stays an estimate until the notebook is run once
on Colab (listed in `reports/YAPILACAKLAR_INSAN_TR.md`).
