# FLOGA notes (Phase 2)

Sources read: `external/FLOGA` (github.com/Orion-AI-Lab/FLOGA, commit 0efa4b3, 2026-07-07), its README, `configs/`,
`create_dataset.py`, `dataset_utils.py`, `utils.py`, `cd_experiments_utils.py`; `external/FLOGA-annotations`
(github.com/Orion-AI-Lab/FLOGA-annotations); the paper (Sdraka et al., IEEE JSTARS 17, 7801–7824, 2024,
doi:10.1109/JSTARS.2024.3381737).

## Dataset versions and access
| | v1 (paper) | v2 (used here) |
|---|---|---|
| Format | BZip2-compressed HDF5 per year and GSD | GeoTIFF per event, packed in 62 tar shards (20 m) |
| Location | Dropbox shared folder | Hugging Face `orion-ai-lab/FLOGA-GeoTIFFs`, folder `sen2_20_mod_500` |
| Size (20 m) | not listable (folder page needs JavaScript; zip link returns HTML) | 1.1 TB (all shards) |
| S2 products | old processing baselines (N0211–N0213), no BOA offset | reprocessed, baseline N0500, **+1000 BOA offset** |
| S2 bands (20 m) | B02,B03,B04,B05,B06,B07,B11,B12,B8A | B01,B02,B03,B04,B05,B06,B07,B11,B12,B8A (GDAL band descriptions) |
| Licence | data CC BY 4.0 (`DATA_LICENSE`), code MIT (`LICENSE`) | CC BY 4.0 (HF card) |

v2 file names: `<year>/<fireID>_<row>_SEN2_2a_20_{before,after,label,clc_mask,sea_mask_before,sea_mask_after}.tiff`
and `<year>/<fireID>_<row>_SEN2_2a_cloud_mask_20_{before,after}.tiff` (+ MODIS files we do not use). Images are
EPSG:4326 (~0.00021° ≈ 20 m), uncompressed, one row per strip, nodata 65535.

**Access method (our choice, DECISIONS.md):** the tar shards are indexed by reading the 512-byte tar headers with
HTTP range requests (`scripts/hf_tar_index.py`, `scripts/hf_index_all.py` → `data/raw/floga_v2/index/*.csv`), then
the needed GeoTIFFs are read in place (`src/burnt/remote_tiff.py`): full label / SCL / sea / CLC masks (uint8) and
only the 256×256 windows of the selected patches of the S2 images. Hugging Face's `/resolve/` endpoint returns
HTTP 429 under many requests, so the redirect is resolved once and the signed CDN URL is reused.

## Masks and labels
- Label: 0 unburnt, 1 burnt, 2 burnt by another fire of the same year → ignored in training/evaluation
  (FLOGA `fillna` / loss `ignore_index=2`).
- Cloud masks: Sentinel-2 SCL. FLOGA marks a patch cloudy if it has > 10 pixels with SCL = 9 in the pre or post mask.
- Sea mask: values 2 and 4 → land, other non-zero → sea/water (`create_dataset.py`). Patches with ≥ 90 % sea removed.
- CLC 2018 at the 20 m grid (used only for error analysis on FLOGA).

## Normalisation used by BAM-CD
`configs/config.json`: `scale_input = "clamp_scale_10000"` → `min(DN, 10000) / 10000`; NaN → 0 (`nan_value`).
(`sen2_mean/std` exist in the config but are only used with `scale_input = "normalize"`.)

## Official split
`data_split.csv` (Dropbox single-file link, downloadable): 338 events `(year, event_id, set)`; `event_id` is the row
of the v1 annotation shapefile of that year. Mapping to v2 files: v1 row → (fire ID, S2 tile from the product
name) → first v2 shapefile row with the same (ID, tile); 2017 Sentinel-Hub rows without a tile are matched on the
fire ID when v2 has a single image for it. Unmatched events are listed in `outputs/tables/floga_event_mapping.csv`.
The paper's patch-level 1:1 negative sampling is not published as patch lists, so we re-create it with the same
rules (positive patches + an equal number of random valid negative patches per split, seed 999). The paper reports
826 / 189 / 152 train / val / test patches; our counts are in `outputs/tables/floga_patch_counts.json`.

## BAM-CD
- Model code `models/bam_cd/` (U-Net, ResNet-101 encoder, scSE attention in the decoder, concatenation fusion);
  2-class output, prediction = argmax.
- v1 weights (paper model, Siamese): 83 727 022 parameters (paper: 83.7 M) ✔; checkpoint 1.0 GB (includes
  optimiser state). v2 weights (pseudo-Siamese, augmentation): 126 245 998 parameters, checkpoint 1.5 GB.
- Paper Table V (FLOGA test, burnt class, mean of 5 runs): BAM-CD P 85.63, R 91.56, **F1 88.48**, IoU 79.34;
  dNBR (global threshold tuned on train) F1 45.14; best index dMIRBI F1 64.06; RF F1 67.38; SVM F1 68.93.

## Pixel sampling for the ML models (paper Sec. V-B)
Positive patches and an equal number of negative patches; N = 2000 pixels per patch, N/2 burnt and N/2 unburnt,
≥ 10 % of unburnt pixels from sea/water when present. Our interpretation (DECISIONS.md): positive patches give N/2
burnt (all if fewer) + N/2 unburnt; negative patches give N/2 unburnt; pixels with SCL cloud/shadow (3, 8, 9, 10),
label 2 or no data are excluded.
