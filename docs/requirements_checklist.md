# Requirements checklist (from `EHB466E_Project_Plan_and_Grading_2026_10_18.docx`)

Each course requirement → where this repository satisfies it. Updated as phases complete.

## Section 3 — Problem acceptability
| Requirement | Where satisfied |
|---|---|
| Real, specific: named place, period, user | Manavgat & Marmaris, Jul–Aug 2021; users OGM, AFAD (README, paper §I) |
| Measurable vs independent reference | EFFIS perimeters (`data/effis_TR_2021.json`); metrics in `outputs/tables/` |
| Remote sensing adds value | Burnt-area maps derived from Sentinel-2 L2A (paper §I) |
| Data available legally | Sentinel-2 (Copernicus open), EFFIS (attribution), FLOGA (CC BY 4.0) — `DATA_MANIFEST.md` |
| Benchmark exists | dNBR (Key & Benson 2006), BAM-CD (Sdraka et al. 2024) |
| Feasible on Colab/laptop | `docs/hardware.md`, `docs/demo_runtime.md` |

## Section 4 — Mandatory scope
| Requirement | Where satisfied |
|---|---|
| Problem + proposed solution stated | README, paper §I |
| End-to-end pipeline in code | `src/burnt/`, `scripts/`, `run_all.py` |
| ≥1 satellite dataset | Sentinel-2 L2A (Planetary Computer), FLOGA S2 |
| Quantitative validation vs independent reference, standard metric | `scripts/70_evaluate.py` (P, R, F1, IoU, OA) |
| Comparison with ≥1 published benchmark | dNBR + BAM-CD, `scripts/30_tune_dnbr.py`, `scripts/60_run_bamcd.py` |
| IEEE conference paper | `paper/main.tex`, `paper/main.pdf` |

## Section 5 — Benchmark and originality
| Requirement | Where satisfied |
|---|---|
| Run benchmark on our data (open code / faithful reimpl.) | BAM-CD from official repo + weights (`src/burnt/bamcd.py`) |
| Score on public benchmark dataset if exists | FLOGA official test split (`outputs/tables/floga_test_*.csv`) |
| Related work | paper §II |
| Fair comparison: same data, split, metrics | identical Turkish grid/mask/reference; same FLOGA test split (`src/burnt/metrics.py`) |
| 1–3 original contributions | paper §I bullets: low-cost transfer, first quantitative FLOGA→TR test |
| Measured advantage with uncertainty + runtime/cost | 5 seeds, block-bootstrap CIs, ΔF1 CIs, cost table (`scripts/70_evaluate.py`) |
| Ablation | `scripts/80_ablation.py` |
| Honesty where not better | paper §VI Discussion |

## Section 6 — Software and data
| Requirement | Where satisfied |
|---|---|
| Runs end to end, no manual steps | `run_all.py` |
| Method and benchmark in same pipeline | `scripts/70_evaluate.py` |
| Main demo ≈30 min on Colab free | `notebooks/00_demo_end_to_end.ipynb`, `docs/demo_runtime.md` |
| Data manifest with source + licence | `DATA_MANIFEST.md` |
| Python tools | requirements.txt / requirements-lock.txt / environment.yml |

## Section 8 — Final presentation (28 Dec 2026)
| Requirement | Where satisfied |
|---|---|
| 12 min + Q&A, every member presents own part | `reports/final_presentation_outline.md` |
| Demo ≤ 3 min | demo script in the outline |

## Section 9 — Final paper (4 Jan 2027)
| Requirement | Where satisfied |
|---|---|
| IEEE two-column, 6–8 pages incl. figures | `paper/main.pdf` |
| Figures/numbers regenerable from repo | `paper/numbers.tex` from `scripts/95_make_paper_numbers.py` |
| Abstract ≤200 words + index terms | paper |
| Intro with 2–3 contribution bullets | paper §I |
| Related work + benchmark | paper §II |
| Study area + data with source/licence | paper §III |
| Method: architecture, method, benchmark impl., parameters | paper §IV |
| Experiments: fair comparison, uncertainty, ablation, error analysis | paper §V |
| Discussion: where better, failures, limits | paper §VI |
| Conclusion + future work | paper §VII |
| Code & data availability (URL + release tag) | paper, release `v1.0` |
| CRediT author contributions + AI-use statement | paper end matter |
| IEEE references | `paper/refs.bib` |

## Section 10 — Teams
| Requirement | Where satisfied |
|---|---|
| Each member owns named parts; everyone codes and writes | `reports/YAPILACAKLAR_INSAN_TR.md` (human task), CRediT section |
| Oral check: any member can explain any code | `reports/SUNUM_VE_SORU_CEVAP_TR.md` code walk-through |
| Reuse with attribution & licences | FLOGA code MIT, data CC BY 4.0 credited (README, DATA_MANIFEST, paper) |
| AI-use statement | README, paper |
