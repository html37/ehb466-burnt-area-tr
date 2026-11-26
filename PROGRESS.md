# Progress

Legend: [x] done · [~] in progress · [ ] not started

## Phase 0 — Bootstrap
- [x] Read `_inputs/*` (course plan, proposal, FLOGA paper); no interim_feedback / team_roles files present
- [x] `docs/requirements_checklist.md`
- [x] `docs/hardware.md`
- [~] `.venv` (Python 3.11) + packages; pin `requirements-lock.txt`
- [x] git remote + `gh auth status` OK (account html37, repo scope)
- [x] PROGRESS / DECISIONS / BLOCKERS / .gitignore

## Phase 1 — Refactor into `src/` and reproduce dNBR table
- [ ] src/burnt modules + configs/default.yaml
- [ ] unit tests
- [ ] `scripts/70_evaluate.py --method dnbr_usgs` reproduces Section 0 (±0.003)

## Phase 2 — FLOGA acquisition and loader
- [~] FLOGA clone (`external/FLOGA`), notes, split CSV, remote index of HF v2 tars
- [ ] patch extraction + pixel sampling parquet

## Phase 3 — Turkish test set
## Phase 4 — Benchmarks (dNBR tuned, BAM-CD)
## Phase 5 — Our model + HPO
## Phase 6 — Evaluation, ablation, error analysis
## Phase 7 — Check-labelling kit
## Phase 8 — Demo notebook
## Phase 9 — Reproducibility & GitHub release
## Phase 10 — IEEE paper
## Phase 11 — Presentation material
## Phase 12 — Turkish reports

## Last done / next
- Last: Phase 0 docs; FLOGA data access strategy (HF range reads).
- Next: finish env install, Phase 1 refactor.
