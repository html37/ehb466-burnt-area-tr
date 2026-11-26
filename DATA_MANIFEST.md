# Data manifest

Every dataset and pretrained model used in this project, with source, licence, access date and how the code obtains
it (course requirement, Sections 6 and 11). No dataset needs a manual download: every script fetches what it needs.

| Dataset / model | Use in project | Source and access by the code | Licence | Accessed |
|---|---|---|---|---|
| Sentinel-2 Level-2A (B02–B07, B8A, B11, B12, SCL), 2021 scenes of tiles 36SUF, 36SUG, 35SNA, 35SPA | Input imagery of the Turkish test areas, pre- and post-fire | Microsoft Planetary Computer STAC, collection `sentinel-2-l2a`; `scripts/10_build_turkey_testset.py` (`src/burnt/stac.py`) reads only the AOI windows of the COGs. Item IDs: `outputs/tables/selected_scenes.csv` | Copernicus Sentinel data: free, full and open (Copernicus data policy) | 2026-11-26 |
| EFFIS burnt-area perimeters, Türkiye, 2021-07-01 – 2021-08-31 | Reference data for the Turkish test areas | European Forest Fire Information System (data request form); stored as `data/effis_TR_2021.json` (EPSG:3035, 251 polygons, `map_source = sentinel2`) | © European Union, 2000–2021; reuse authorised provided the source is acknowledged | 2026-11 (EFFIS e-mail) |
| FLOGA v2 (Sentinel-2 20 m pre/post imagery, labels by the Hellenic Fire Service, SCL cloud masks, sea mask, CORINE 2018), Greece 2017–2021 | Training / validation data of our models; public benchmark (official test split) | Hugging Face `orion-ai-lab/FLOGA-GeoTIFFs`, folder `sen2_20_mod_500`; read remotely by HTTP byte ranges (`scripts/hf_index_all.py`, `scripts/20_prepare_floga.py`, `src/burnt/remote_tiff.py`) — only the masks and the selected 256×256 patches are transferred, not the 1.1 TB of shards | CC BY 4.0 (Sdraka et al., 2024) | 2026-11-26 |
| FLOGA official split (`data_split.csv`) | Event-level train/val/test split | Dropbox link in the FLOGA README (single-file download); copy at `data/raw/floga/data_split.csv` (downloaded by `scripts/05_get_external.py`) | CC BY 4.0 (part of FLOGA) | 2026-11-26 |
| FLOGA annotation polygons v1/v2 | Mapping of the v1 split to v2 file names (fire ID, S2 product, processing baseline) | `git clone https://github.com/Orion-AI-Lab/FLOGA-annotations` into `external/` | MIT | 2026-11-26 |
| BAM-CD code | Benchmark 2 | `git clone https://github.com/Orion-AI-Lab/FLOGA` into `external/` by `scripts/05_get_external.py` (commit 0efa4b3), wrapped by `src/burnt/bamcd.py` | MIT | 2026-11-26 |
| BAM-CD pretrained weights v1 (paper model) and v2 | Benchmark 2 | Dropbox links in the FLOGA README (zip download), downloaded by `scripts/05_get_external.py` to `data/raw/bamcd/` | distributed with the MIT-licensed FLOGA repository; credited to Sdraka et al. | 2026-11-26 |
| ESA WorldCover 10 m 2021 v200 | Error analysis by land cover | Planetary Computer collection `esa-worldcover` (`scripts/90_error_analysis.py`) | CC BY 4.0 (© ESA WorldCover project 2021 / contains modified Copernicus Sentinel data) | 2026-11-26 |

## Notes on the EFFIS reference
- EFFIS IDs of the study areas: **51931** (Manavgat, 54 769 ha, 2021-07-28 – 2021-08-06) and **51942**
  (Marmaris, 11 548 ha, 2021-07-29 – 2021-08-05). All EFFIS fires that started between the pre- and post-fire image
  dates and intersect the AOI are rasterised into the reference (`src/burnt/reference.py`).
- The perimeters were mapped by EFFIS from Sentinel-2 imagery, i.e. from the same sensor as our input; the reference
  is therefore independent in method but not in data. A blind check-labelling kit is provided in `labeling/`.

## Notes on FLOGA v2
- v2 imagery is the Copernicus reprocessing (baseline N0500, +1000 BOA offset), not the imagery of the paper (v1);
  we remove the offset and band B01 (see `docs/floga_notes.md`, `DECISIONS.md`).
- 327 of the 338 events of the official split have a v2 image; the mapping is in
  `outputs/tables/floga_event_mapping.csv`.
