# Burnt-area mapping of the 2021 Manavgat and Marmaris wildfires with Sentinel-2

EHB 466E Remote Sensing term project, Istanbul Technical University, Fall 2026.
Team: Erdem Akkoca (040220057), Hüseyin Temel (040220042).

## Problem

In July–August 2021, large wildfires burned more than 60 000 ha around Manavgat (Antalya) and Marmaris (Muğla).
Agencies such as OGM and AFAD need accurate burnt-area maps for damage assessment and restoration planning.
This project builds an end-to-end Python pipeline that maps burnt areas from bi-temporal Sentinel-2 imagery and
compares methods against independent reference perimeters.

**Proposed contribution (final phase):** a lightweight machine-learning model trained on the Greek FLOGA dataset
and transferred to Türkiye, reaching accuracy close to the deep-learning benchmark BAM-CD at a much lower
computational cost.

## Methods

| Role | Method | Status |
|---|---|---|
| Benchmark 1 | dNBR thresholding (USGS 0.10 / 0.27, Otsu) | done (`notebooks/01_dnbr_benchmark.ipynb`) |
| Benchmark 2 | BAM-CD (Sdraka et al., 2024), pretrained weights | planned |
| Our method | Lightweight pixel classifier trained on FLOGA, tested on Manavgat/Marmaris | planned |

## First results (dNBR benchmark vs. EFFIS)

Evaluation area: EFFIS perimeter + 3 km buffer, 20 m grid, cloud/shadow/water pixels masked.

| Event | Threshold | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| Manavgat | 0.10 | 0.894 | 0.880 | 0.887 | 0.797 |
| Manavgat | 0.27 | 0.960 | 0.689 | 0.802 | 0.669 |
| Manavgat | Otsu (0.335) | 0.971 | 0.597 | 0.740 | 0.587 |
| Marmaris | 0.10 | 0.957 | 0.897 | 0.926 | 0.861 |
| Marmaris | 0.27 | 0.983 | 0.675 | 0.800 | 0.667 |
| Marmaris | Otsu (0.289) | 0.984 | 0.645 | 0.780 | 0.639 |

Thresholds are fixed a priori (USGS) or unsupervised (Otsu); none is tuned on the EFFIS reference.

## How to run

**Google Colab (free tier):** open `notebooks/01_dnbr_benchmark.ipynb` in Colab, set `REPO_URL` in the first cell
and choose *Runtime → Run all*. Runtime is about 1 minute; outputs are written to `outputs/`.

**Local:**

```bash
git clone https://github.com/KULLANICI_ADI/ehb466-burnt-area-tr.git
cd ehb466-burnt-area-tr
pip install -r requirements.txt
jupyter notebook notebooks/01_dnbr_benchmark.ipynb
```

No manual steps: Sentinel-2 scenes are searched and read through the Microsoft Planetary Computer STAC API;
the EFFIS reference perimeters are included in `data/`.

## Outputs

| File | Content |
|---|---|
| `outputs/benchmark_dnbr_metrics.csv` | Precision, recall, F1, IoU, OA, areas per event and threshold |
| `outputs/selected_scenes.csv` | Sentinel-2 scenes used (item ID, date, tile, cloud) |
| `outputs/<event>_overview.png` | Post-fire false-colour image with EFFIS perimeter, dNBR map |
| `outputs/<event>_error_maps.png` | TP / FP / FN maps for each threshold |
| `outputs/<event>_dnbr.tif` | dNBR raster (cached intermediate result) |

## Repository structure

```
data/           reference data (EFFIS perimeters)
notebooks/      end-to-end pipeline notebooks
outputs/        generated figures and tables (re-created by the notebooks)
DATA_MANIFEST.md  sources, licences and access dates of all datasets
```

## Data

See [DATA_MANIFEST.md](DATA_MANIFEST.md) for sources, licences and access dates.

## References

- M. Sdraka et al., "FLOGA: A Machine-Learning-Ready Dataset, a Benchmark, and a Novel Deep Learning Model for
  Burnt Area Mapping With Sentinel-2," *IEEE JSTARS*, vol. 17, 2024, doi:10.1109/JSTARS.2024.3381737.
- C. H. Key and N. C. Benson, "Landscape Assessment (LA): Sampling and Analysis Methods," USDA Forest Service
  Gen. Tech. Rep. RMRS-GTR-164-CD, 2006 (dNBR severity thresholds).
- N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. SMC*, vol. 9, no. 1, 1979.

## AI-use statement

AI assistants (Claude, Gemini) were used for coding help and language editing. The team is responsible for all
code, numbers and text.
