# Data manifest

Every dataset used in this project, with its source, licence and access date (course requirement, Section 6 and 11).

| Dataset | Use in project | Source / access | Licence | Accessed |
|---|---|---|---|---|
| Sentinel-2 Level-2A (bands B04, B8A, B12, SCL) | Input imagery, pre- and post-fire | Microsoft Planetary Computer STAC, collection `sentinel-2-l2a`, read directly by the notebook | Copernicus Sentinel data: free, full and open access (Copernicus data policy) | 2026-11-08 |
| EFFIS burnt-area perimeters, Türkiye, 2021-07-01 – 2021-08-31 | Reference (validation) data | European Forest Fire Information System, Data Request Form; stored as `data/effis_TR_2021.json` (EPSG:3035, 251 polygons, `map_source = sentinel2`) | © European Union, 2000–2018; reuse authorised provided the source is acknowledged | 2026-11-__ (date of EFFIS e-mail) |
| FLOGA (Sentinel-2 + MODIS, Greek wildfires 2017–2021) | Training data for our model; BAM-CD benchmark | github.com/Orion-AI-Lab/FLOGA | to be copied from the repository licence | planned |
| BAM-CD pretrained weights | Benchmark 2 | github.com/Orion-AI-Lab/FLOGA | to be copied from the repository licence | planned |

## Notes on the EFFIS reference

- EFFIS IDs used for the study areas: **51931** (Manavgat, 54 769 ha, 2021-07-28 – 2021-08-06) and
  **51942** (Marmaris, 11 548 ha, 2021-07-29 – 2021-08-05).
- The perimeters in this file were mapped by EFFIS from Sentinel-2 imagery. Our method also uses Sentinel-2,
  so the reference is produced by a different method but from the same sensor; an additional team-made
  check labelling is planned for the final phase.

## Sentinel-2 scenes

The exact scenes (STAC item IDs, dates, tile, cloud cover) are selected automatically and written to
`outputs/selected_scenes.csv` on every run.
