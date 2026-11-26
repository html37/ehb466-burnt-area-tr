# Decisions log

Format: date — decision — alternatives — reason.

- **2026-11-26 — Python 3.11 venv (`py -3.11 -m venv .venv`).** Alt: 3.14/3.12, conda. Reason: student's explicit
  instruction; 3.11 has wheels for all geo/ML packages on Windows.
- **2026-11-26 — FLOGA source = v2 GeoTIFFs on Hugging Face (`orion-ai-lab/FLOGA-GeoTIFFs`, `sen2_20_mod_500`),
  read remotely by byte range, not the v1 Dropbox HDF files.** Alt: (a) v1 HDF from Dropbox — the shared folder is
  rendered with JavaScript and the folder-zip link returns an HTML page, so no file can be fetched by code;
  (b) download the full v2 20 m set — 1.1 TB, disk has 94 GB. Reason: v2 tars contain *uncompressed* GeoTIFFs whose
  byte offsets can be indexed with HTTP range requests (`scripts/hf_tar_index.py`), so only the 256x256 patches we
  need are transferred (own reader `src/burnt/remote_tiff.py` with `tifffile`; GDAL `/vsicurl/` stalled on these URLs). Consequence: imagery is the reprocessed v2 version,
  which differs from the v1 imagery BAM-CD v1 was trained on (logged in the BAM-CD sanity check).
- **2026-11-26 — Official event split from `data_split.csv` (FLOGA README link, Dropbox single-file download works).**
  The v1 split keys `(year, event_id)` (= row of the v1 annotation shapefile of that year) are matched to the v2
  file names `<ID>_<row>` through (fire ID, S2 tile), see the "v1 split -> v2 files" entry below. Patch-level negative sampling
  of the original pickles is not published, so we re-create it with FLOGA's own `create_dataset.py` rules
  (256x256 non-overlapping patches, drop all-zero, cloudy and >=90 % sea patches, 1:1 positive:negative, fixed seed).
- **2026-11-26 — v2 band handling.** v2 S2 GeoTIFFs have 10 bands (B01,B02,B03,B04,B05,B06,B07,B11,B12,B8A per the
  GDAL band descriptions) and come from baseline N0500 (+1000 BOA offset, product names in FLOGA-annotations v2).
  We drop B01 (-> exactly the v1 / BAM-CD order) and subtract 1000 (v1 and the 2021 Planetary-Computer scenes,
  baseline 03.01, have no offset). Nodata 65535 -> 0. Alt: keep the offset — would shift all reflectances by 0.1
  relative to Türkiye and to what BAM-CD v1 was trained on.
- **2026-11-26 — v1 split -> v2 files.** 327 of 338 split events map to a v2 image (328 by fire ID + S2 tile,
  5 by fire ID only; 11 unmatched because v2 has no image for them: 9 train, 2 test). Unmatched events are dropped
  and listed in `outputs/tables/floga_event_mapping.csv`.
- **2026-11-26 — Patch filters re-implemented from create_dataset.py**: 256x256 grid with FLOGA's padding, drop
  patches with >10 SCL=9 pixels in pre or post (cloudy), >=90 % sea, or no data (SCL=0 everywhere); then per split
  all positive patches + the same number of random negative patches (seed 999 = FLOGA config seed). The "zero patch"
  test uses SCL=0 instead of the S2 band, because it avoids downloading every negative patch.
- **2026-11-26 — Pixel sampling interpretation (paper Sec. V-B).** Positive patches: N/2 burnt (all if fewer) + N/2
  unburnt; negative patches: N/2 unburnt; >=10 % of unburnt from sea when the patch has sea; pixels with SCL
  3/8/9/10, label 2 or no data excluded; N = 2000 (paper). Alt: N pixels from negative patches too — would give a
  3:1 unburnt:burnt ratio; ours gives ~2:1 and the decision threshold is tuned on val anyway.
- **2026-11-26 — Reflectance for ML features = min(DN,10000)/10000** (FLOGA `clamp_scale_10000`), same for FLOGA
  and Türkiye; dNBR benchmark uses DN/10000 without clamp (as notebook 01). NDWI = McFeeters (FLOGA Table IV);
  MIRBI = 10·B12 − 9.8·B11 + 2 (Trigg & Flasse 2001; FLOGA Table IV prints the SWIR subscripts the other way).
- **2026-11-26 — HPO objective** = best-threshold F1 on all valid FLOGA-val pixels (subset of 6 M random pixels if
  > 12 M); LightGBM early stopping on the sampled val pixels (log-loss). Final decision threshold re-tuned per seed
  on val. Ablation groups reuse the ALL hyper-parameters (no extra tuning, only early stopping/threshold).
- **2026-11-26 — BAM-CD on Türkiye**: 256x256 windows, stride 128, triangular blending; threshold 0.5 (= argmax).
  Cost is reported per 1000 km² of raster (AOI bounding box) for all methods.
- **2026-11-27 — FLOGA sample sizes (stage B/D result).** Cloud filter of create_dataset.py (>10 SCL=9 px per patch)
  leaves 269/92/81 positive patches (train/val/test) + equal negatives; samples: train 710 625 px (226 937 burnt),
  val 244 450 px (76 296 burnt). Kept as is (faithful to FLOGA rules; enough for pixel classifiers). Full FLOGA-val
  = 10 811 245 valid px (< 12 M) -> HPO objective uses all val pixels, no subset.
- **2026-11-27 — Harmonisation check passed** (`outputs/figures/band_harmonisation.png`, `band_harmonisation.csv`):
  unburnt-land medians of FLOGA (after −1000 offset) and Türkiye agree within ~0.01–0.03 reflectance in every band;
  no 0.1 shift, so the offset handling is correct. Visible difference: Marmaris is darker in B11/B12 and its pre
  scene has very low B02 (clearer atmosphere / different land cover) — a genuine domain difference, not a scaling error.
- **2026-11-27 — dNBR threshold tuned on FLOGA val = 0.24** (val F1 0.678), applied unchanged to FLOGA test and Türkiye.
- **2026-11-27 — BAM-CD sanity check passed** (`outputs/tables/bamcd_sanity.csv`): v1 F1 on our FLOGA test split
  0.909 vs 0.885 in the paper (Table V); v2 0.913. The +2.4 pt difference is plausible because our test set uses
  the v2 imagery and re-created negative patches (same events); band order/normalisation are therefore correct.
  BAM-CD is used on Türkiye without changes.
