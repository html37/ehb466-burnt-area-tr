# Progress

Legend: [x] done · [~] in progress · [ ] not started

## Phase 0 — Bootstrap
- [x] Read `_inputs/*` (course plan, proposal, FLOGA paper); no interim_feedback / team_roles files present
- [x] `docs/requirements_checklist.md`, `docs/hardware.md`
- [x] `.venv` (Python 3.11) + packages; `requirements-lock.txt`; git remote + `gh auth` OK
- [x] PROGRESS / DECISIONS / BLOCKERS / .gitignore (+ `.gitattributes` LF normalisation)

## Phase 1 — Refactor into `src/` and reproduce dNBR table
- [x] src/burnt modules + configs/default.yaml; unit tests; `70_evaluate.py --method dnbr_usgs` reproduces Section 0

## Phase 2 — FLOGA acquisition and loader
- [x] FLOGA clone, split CSV, BAM-CD v1/v2 weights, remote index of the HF tars, docs/floga_notes.md
- [x] scripts/20_prepare_floga.py: 327 events, 538/184/162 patches, 710 625 train / 244 450 val sampled pixels

## Phase 3 — Turkish test set
- [x] scripts/10_build_turkey_testset.py; band harmonisation check passed (`band_harmonisation.png`)

## Phase 4 — Benchmarks
- [x] dNBR USGS + FLOGA-val-tuned threshold (0.24)
- [x] BAM-CD v1/v2 sanity check on FLOGA test (0.909 / 0.913 vs paper 0.885); Türkiye inference GPU + CPU cost

## Phase 5 — Our model + HPO
- [x] HPO LightGBM (60 trials), RF (14 trials, 2 h budget), LogReg (60); best params in configs/
- [x] Final training 5 seeds; RF pickles git-ignored (1.1 GB each)

## Phase 6 — Evaluation, ablation, error analysis
- [x] 70_evaluate (block bootstrap, ΔF1), 72 cost table, 80 ablation, 90 error analysis (WorldCover + severity), 85 figures

## Phase 7 — Check-labelling kit
- [x] labeling kit + 75_eval_checklabels.py; labelling itself = optional human task (BLOCKERS.md)

## Phase 8 — Demo notebook
- [x] notebook + scripts/demo_run.py; clean-clone runs GPU 184 s / CPU 186 s wall (docs/demo_runtime.md)

## Phase 9 — Reproducibility & GitHub release
- [x] README (auto results table), DATA_MANIFEST, CITATION.cff, environment files, tests pass
- [x] GitHub release v1.0 (models_lgbm.zip, outputs zip, paper PDF)

## Phase 10 — IEEE paper
- [x] paper/main.pdf, 7 pages (≈6 pages body + references), all numbers from numbers.tex, tectonic build

## Phase 11 — Presentation material
- [x] reports/final_presentation_outline.md (key numbers auto-filled)

## Phase 12 — Turkish reports
- [x] BITIRME_RAPORU_TR.md, SUNUM_VE_SORU_CEVAP_TR.md (results auto-filled), YAPILACAKLAR_INSAN_TR.md

## Last done / next
- Last: demo runtime documented; release v1.0 published. All phases done.
- Next: only human tasks (reports/YAPILACAKLAR_INSAN_TR.md, BLOCKERS.md).
