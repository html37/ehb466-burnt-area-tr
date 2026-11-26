# Decisions log

Format: date — decision — alternatives — reason.

- **2026-11-26 — Python 3.11 venv (`py -3.11 -m venv .venv`).** Alt: 3.14/3.12, conda. Reason: student's explicit
  instruction; 3.11 has wheels for all geo/ML packages on Windows.
- **2026-11-26 — FLOGA source = v2 GeoTIFFs on Hugging Face (`orion-ai-lab/FLOGA-GeoTIFFs`, `sen2_20_mod_500`),
  read remotely by byte range, not the v1 Dropbox HDF files.** Alt: (a) v1 HDF from Dropbox — the shared folder is
  rendered with JavaScript and the folder-zip link returns an HTML page, so no file can be fetched by code;
  (b) download the full v2 20 m set — 1.1 TB, disk has 94 GB. Reason: v2 tars contain *uncompressed* GeoTIFFs whose
  byte offsets can be indexed with HTTP range requests (`scripts/hf_tar_index.py`), so only the 256x256 patches we
  need are transferred (GDAL `/vsisubfile/` over `/vsicurl/`). Consequence: imagery is the reprocessed v2 version,
  which differs from the v1 imagery BAM-CD v1 was trained on (logged in the BAM-CD sanity check).
- **2026-11-26 — Official event split from `data_split.csv` (FLOGA README link, Dropbox single-file download works).**
  The v1 split keys `(year, event_id)` are matched to the v2 file names `<ID>_<rowid>` via `rowid == event_id`
  (FLOGA-annotations README: v1 files are keyed by the shapefile row index per year). Patch-level negative sampling
  of the original pickles is not published, so we re-create it with FLOGA's own `create_dataset.py` rules
  (256x256 non-overlapping patches, drop all-zero, cloudy and >=90 % sea patches, 1:1 positive:negative, fixed seed).
