# Progress

Legend: [x] done · [~] in progress · [ ] not started

## Phase 0 — Bootstrap
- [x] Read `_inputs/*` (course plan, proposal, FLOGA paper); no interim_feedback / team_roles files present
- [x] `docs/requirements_checklist.md`
- [x] `docs/hardware.md`
- [x] `.venv` (Python 3.11) + packages (torch 2.14.1+cu126); `requirements-lock.txt`
- [x] git remote + `gh auth status` OK (account html37, repo scope)
- [x] PROGRESS / DECISIONS / BLOCKERS / .gitignore

## Phase 1 — Refactor into `src/` and reproduce dNBR table
- [x] src/burnt modules + configs/default.yaml
- [x] unit tests (metrics, indices, Otsu, grid alignment, no-leakage)
- [x] `scripts/70_evaluate.py --method dnbr_usgs` reproduces Section 0 exactly (diff 0.000)

## Phase 2 — FLOGA acquisition and loader
- [x] FLOGA clone, annotations, split CSV, BAM-CD v1/v2 weights, remote index of 62 HF tars, docs/floga_notes.md
- [~] scripts/20_prepare_floga.py running (stage A masks -> B selection -> C patches -> D samples)

## Phase 3 — Turkish test set
- [x] scripts/10_build_turkey_testset.py (same scenes as notebook 01; 9 FLOGA bands, SCL, valid, EFFIS)
- [ ] band harmonisation figure (needs FLOGA samples)
## Phase 4 — Benchmarks (dNBR tuned, BAM-CD)
## Phase 5 — Our model + HPO
## Phase 6 — Evaluation, ablation, error analysis
## Phase 7 — Check-labelling kit
- [x] labeling/points.csv (+key, geojson), NASIL_ETIKETLENIR.md, scripts/75_eval_checklabels.py; labels = human task
## Phase 8 — Demo notebook
## Phase 9 — Reproducibility & GitHub release
## Phase 10 — IEEE paper
## Phase 11 — Presentation material
## Phase 12 — Turkish reports

## Last done / next
- Last: all pipeline scripts written (30,35,40,50,60,65,70,72,80,85,90,95, demo_run, run_all); refs verified.
- Next: when 20_prepare_floga finishes -> 30, 35, 40 (HPO lgbm/rf/logreg), 50, 60, 65, 80, 70, 72, 90, 85, 95.
