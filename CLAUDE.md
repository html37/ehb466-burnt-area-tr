# CLAUDE.md — EHB 466E term project: burnt-area mapping, Manavgat & Marmaris 2021

You are the engineering and writing agent for this project. Your job is to take it from its current state to a
finished, reproducible, graded deliverable: code, experiments, GitHub repository, IEEE paper, final-presentation
material and a Turkish final report for the student. Work autonomously from start to finish.

---

## 0. Context

- **Course:** EHB 466E Remote Sensing, İTÜ, Fall 2026. Full rules: `_inputs/EHB466E_Project_Plan_and_Grading_2026_10_18.docx` (read it first; it overrides anything here if they conflict).
- **Team:** Hüseyin Temel (040220042), Erdem Akkoca (040220057). Project proposal: `_inputs/ehb_466_konu.docx`.
- **Deadlines:** final presentation **Mon 28 Dec 2026** (12 min + 3 min demo), final paper **Mon 4 Jan 2027 23:59** (IEEE conference, 6–8 pages + code).
- **Repository:** `https://github.com/html37/ehb466-burnt-area-tr` (this folder is its clone, branch `main`).
- **Problem:** map burnt area of the July–Aug 2021 Manavgat (EFFIS id 51931, 54 769 ha) and Marmaris (EFFIS id 51942, 11 548 ha) wildfires from bi-temporal Sentinel-2 L2A; users: OGM, AFAD.
- **Benchmarks:** (1) dNBR thresholding, (2) BAM-CD (Sdraka et al., IEEE JSTARS 2024, `_inputs/FLOGA_paper.pdf`, code + pretrained weights at `github.com/Orion-AI-Lab/FLOGA`).
- **Our contribution:** a lightweight pixel classifier (LightGBM primary, Random Forest secondary) trained on the Greek **FLOGA** dataset and transferred **without any Turkish training data** to Manavgat/Marmaris; claim = accuracy close to BAM-CD at a much lower cost (params, time, memory, no GPU) + first quantitative test of FLOGA→Türkiye transfer (the FLOGA paper only shows qualitative transfer examples).
- **Already done (in repo):** `notebooks/01_dnbr_benchmark.ipynb` — STAC access via Microsoft Planetary Computer, same-pass multi-tile mosaicking, SCL masking, B8A/B12 dNBR at 20 m, EFFIS rasterisation, metrics. Reference results (AOI = EFFIS polygon + 3 km buffer):

  | Event | Thr | P | R | F1 | IoU |
  |---|---|---|---|---|---|
  | Manavgat | 0.10 | 0.886 | 0.878 | 0.882 | 0.789 |
  | Manavgat | 0.27 | 0.956 | 0.684 | 0.797 | 0.663 |
  | Marmaris | 0.10 | 0.954 | 0.897 | 0.925 | 0.860 |
  | Marmaris | 0.27 | 0.983 | 0.675 | 0.800 | 0.667 |

  Scenes: Manavgat pre 2021-07-20 / post 2021-08-09 (S2B, tiles 36SUF+36SUG); Marmaris pre 2021-07-28 / post 2021-08-07 (S2A, tiles 35SNA+35SPA).
- **Extra inputs** (use if present): `_inputs/interim_feedback.md` (instructor's interim comments — address every point), `_inputs/interim_slides.*`, `_inputs/team_roles.md` (who did what).

---

## 1. Operating rules

1. **Never ask the user questions and never wait for confirmation.** When a choice is needed, pick the most defensible option, record it in `DECISIONS.md` (date, decision, alternatives, reason) and continue.
2. **Blockers:** if something truly needs a human (authentication, a manual download behind a login, manual labelling, a missing input file), write it to `BLOCKERS.md` with exact instructions for the student, apply the fallback defined in this file, and keep going. Never stop the whole project for one blocker.
3. **State files (keep them current — they let a fresh session resume):**
   - `PROGRESS.md` — phase checklist with status, what was done last, what is next. Update after every task.
   - `DECISIONS.md`, `BLOCKERS.md` as above.
   - `logs/` — stdout of every long run (`*.log`), never committed if > 5 MB.
4. **Resuming:** at the start of every session read `PROGRESS.md`, `DECISIONS.md`, `BLOCKERS.md`, then continue from the first unfinished task. Never redo finished work unless a later finding requires it (log why).
5. **Long jobs** (downloads, HPO, training): run in the background with logs, poll the log, and do other independent tasks meanwhile. Make every long job resumable (skip already-cached outputs).
6. **Honesty:** never invent, round in your favour or hand-type a number. Every number in README, paper, slides and report must come from a file in `outputs/` produced by committed code. If a result is worse than expected, report it and explain it (the course requires honesty, Section 5.6).
7. **Language:** code, comments, README, paper, slides in **English**. `reports/BITIRME_RAPORU_TR.md` and `reports/SUNUM_VE_SORU_CEVAP_TR.md` in **Turkish**, plain language (the student is not a remote-sensing expert).
8. **References:** cite only works you have verified exist (fetch the DOI/arXiv/publisher page). No made-up references or DOIs.
9. **Do not touch** `_inputs/` (read-only, git-ignored). Do not commit course documents, credentials, raw datasets or files > 50 MB.

---

## 2. Non-negotiable scientific rules

1. **Test isolation.** Manavgat and Marmaris are the **test set**. They are never used to choose thresholds, hyper-parameters, features, sampling, early stopping or model variants. All tuning happens on the **FLOGA validation split**. Turkish data are touched only by the final evaluation script.
2. **Event-level splits.** Use FLOGA's official train/val/test splits (by event/patch). Never split pixels of one event randomly across train and test.
3. **Fair comparison** (course 5.3): dNBR, BAM-CD and our models are evaluated on the identical Turkish grid (20 m, same AOI, same valid-pixel mask, same EFFIS reference) with the same metrics; on FLOGA they use the same official test split.
4. **Input harmonisation.** FLOGA and Planetary-Computer Sentinel-2 must have the same band set, band order, resolution (20 m) and reflectance scaling. Verify by comparing per-band distributions (FLOGA vs Türkiye, unburnt pixels) and log the check. Planetary Computer L2A with processing baseline ≥ 04.00 needs the −1000 BOA offset (2021 scenes normally do not).
5. **Uncertainty** (course 5.5): every learned model is trained with **5 seeds**; report mean ± std. On the Turkish test areas also report 95 % confidence intervals from a **spatial block bootstrap** (1 km × 1 km blocks, 1000 resamples) for F1/IoU and for the F1 difference between methods.
6. **Ablation** (course 5.5): show which part of our method makes the difference (feature groups, see Phase 5).
7. **Main demo ≤ 30 min on Colab free tier** (course 6.3); cached intermediate results and pre-trained model files are allowed and must be downloadable by the code itself.
8. **Seeds and determinism:** fix seeds everywhere (`numpy`, `random`, LightGBM, sklearn, torch), record package versions in every result folder.

---

## 3. Target repository structure

```
README.md  DATA_MANIFEST.md  LICENSE  CITATION.cff  requirements.txt  environment.yml
PROGRESS.md  DECISIONS.md  BLOCKERS.md  run_all.py
configs/        default.yaml (paths, AOIs, dates, bands), best_params_*.yaml
src/burnt/      __init__.py, stac.py (search, pass grouping, mosaic), preprocess.py (scaling, masks,
                resampling), indices.py, reference.py (EFFIS), floga.py (loader, sampling), features.py,
                models.py (LightGBM/RF train-predict-save), bamcd.py (wrapper around FLOGA code), dnbr.py,
                metrics.py (incl. block bootstrap), cost.py (timing, memory), plots.py
scripts/        00_check_env.py, 10_build_turkey_testset.py, 20_prepare_floga.py, 30_tune_dnbr.py,
                40_hpo.py, 50_train_final.py, 60_run_bamcd.py, 70_evaluate.py, 80_ablation.py,
                90_error_analysis.py, 95_make_paper_numbers.py
notebooks/      00_demo_end_to_end.ipynb (Colab, ≤ 30 min), 01_dnbr_benchmark.ipynb (keep), 02_results.ipynb
tests/          pytest unit tests (metrics, indices, mosaic shapes, no-leakage check)
outputs/        tables/*.csv, figures/*.png|pdf, models/ (small), runs/<run_id>/ (config + versions + metrics)
paper/          main.tex (IEEEtran conference), refs.bib, numbers.tex (auto-generated), figures/, main.pdf
reports/        BITIRME_RAPORU_TR.md, SUNUM_VE_SORU_CEVAP_TR.md, final_presentation_outline.md
labeling/       check-label kit (Phase 7)
external/       FLOGA repo clone (git-ignored)
data/           effis_TR_2021.json (committed); raw/, cache/ (git-ignored)
```

`run_all.py` reproduces every table and figure in order, skipping steps whose outputs exist (`--force` to redo).

---

## 4. Phases

Work through the phases in order (parallelise independent tasks while long jobs run). Each phase ends with:
acceptance criteria met → `PROGRESS.md` updated → commit(s) → `git push`.

### Phase 0 — Bootstrap
- Read `_inputs/*` (docx via `python-docx` or `pandoc`, pdf via `pdftotext`/`pypdf`). Extract every requirement of the course plan into `docs/requirements_checklist.md` (requirement → where it is satisfied).
- Probe hardware: OS, CPU cores, RAM, free disk, GPU (`nvidia-smi`), Python version. Write to `docs/hardware.md`; size all later jobs to it.
- Create a virtual environment (`python -m venv .venv`; use conda/mamba only if wheels fail on this OS). Install from `requirements.txt` (add: lightgbm, scikit-learn, optuna, torch (CUDA build if GPU), segmentation-models-pytorch, h5py, pyyaml, tqdm, psutil, pytest, python-docx, pypdf, seaborn, rioxarray, geopandas, pystac-client, planetary-computer). Pin exact versions after install (`pip freeze > requirements-lock.txt`).
- Check `git remote -v` and `gh auth status` (or a test `git push --dry-run`). If push is impossible → `BLOCKERS.md` (tell the student to run `gh auth login`), keep committing locally, push later.
- Create `PROGRESS.md`, `DECISIONS.md`, `BLOCKERS.md`, `.gitignore` additions.

### Phase 1 — Refactor the existing pipeline into `src/` and reproduce
- Move notebook-01 logic into `src/burnt/` modules + `configs/default.yaml`; keep notebook 01 working (it may import from `src`).
- Unit tests for metrics, indices, Otsu, mosaic grid alignment.
- **Acceptance:** `scripts/70_evaluate.py --method dnbr_usgs` reproduces the table in Section 0 within ±0.003 on every metric, with the same scenes.

### Phase 2 — FLOGA acquisition and loader
- Clone `github.com/Orion-AI-Lab/FLOGA` into `external/FLOGA`. Read its README, configs and data-loading code: dataset download links, file format, band list and order, GSD options, normalisation statistics, official splits, cloud/sea masks, licence. Summarise in `docs/floga_notes.md` and add the licence to `DATA_MANIFEST.md`.
- Check dataset size vs free disk. Download what is needed for the **20 m** setting (all years if disk allows; otherwise the minimum that still covers the official train/val/test splits — log the choice). Downloads must be resumable.
- Build `src/burnt/floga.py`: iterate events/patches with pre/post images, label, cloud and sea masks.
- **Pixel sampling for ML** (follow the paper, Section V-B): from positive patches and an equal number of negative patches, sample N pixels per patch, N/2 burnt and N/2 unburnt, ≥ 10 % of unburnt from sea/water when present; exclude cloudy pixels. Default N = 2000 (paper's choice); if RAM is short, reduce and log. Save train/val samples as parquet in `data/cache/`.
- **Fallback if FLOGA cannot be obtained** (dead link, login wall, disk): use the other large 2021 Turkish EFFIS events in `data/effis_TR_2021.json` — Gündoğmuş 51797, Milas 51941, Köyceğiz 52429, Yatağan 51935 — as train/val (leave-one-event-out validation) with exactly the same pipeline, keep Manavgat/Marmaris as test, and reframe the contribution as "transfer across Turkish events"; BAM-CD then runs with its pretrained weights only. Record in `BLOCKERS.md` + `DECISIONS.md` and keep going.

### Phase 3 — Turkish test set
- `scripts/10_build_turkey_testset.py`: for both events, build the 20 m mosaicked pre/post stacks with **exactly the FLOGA band set/order**, SCL masks, valid mask, EFFIS reference raster, and save as compressed GeoTIFF/NPZ in `data/cache/turkey/` + metadata (scene IDs, tiles, CRS, transform, checksums).
- Harmonisation check (rule 2.4): per-band histograms FLOGA-unburnt vs Türkiye-unburnt → `outputs/figures/band_harmonisation.png` + note in `DECISIONS.md`.

### Phase 4 — Benchmarks
- **dNBR:** (a) fixed USGS thresholds 0.10 and 0.27 (literature values), (b) global threshold found by grid search (step 0.01) maximising F1 on the **FLOGA val** split, then applied unchanged to FLOGA test and Türkiye. Use the same NIR/SWIR band definition as FLOGA's dNBR if the repo defines one; otherwise B8A/B12; log it.
- **BAM-CD:** use the FLOGA repo's model code and pretrained weights (v1 = paper model; also v2 if available, report both, primary = v1 for faithfulness). Wrap it in `src/burnt/bamcd.py`: same normalisation as training, sliding-window inference with overlap on the Turkish rasters, output probability → 0.5 threshold.
  - **Sanity check (acceptance):** evaluate on the FLOGA test split; F1 must be close to the paper's reported value (within a few points). If not, debug band order/normalisation/GSD before using it on Türkiye. Log the result either way.
  - Record cost: parameter count, weights size, inference time per 1000 km² on this machine's GPU (if any) **and** CPU, peak RAM/VRAM.
- If no GPU: run BAM-CD on CPU (inference only, it is feasible); if a step is too slow, process in tiles and log the runtime.

### Phase 5 — Our model + hyper-parameter optimisation
- **Features** (`src/burnt/features.py`), grouped for ablation:
  - G1 `dNBR` only
  - G2 differenced indices: dNBR, dNBR2, dNDVI, dNDWI, dBAIS2, dMIRBI (check formulas in the FLOGA paper Table IV / standard refs)
  - G3 post-fire bands (all FLOGA bands)
  - G4 pre-fire bands
  - ALL = G2 + G3 + G4 (+ pre/post single-date indices)
- **Models:** LightGBM (primary), Random Forest (secondary), Logistic Regression on G2 (tiny baseline).
- **HPO** (`scripts/40_hpo.py`): Optuna TPE, objective = F1 on FLOGA **val** pixels (full val patches, not only the sampled pixels, if memory allows; otherwise a large fixed val sample — log it). Budget: 60 trials or 2 h per model, whichever first, MedianPruner. Search spaces:
  - LightGBM: `num_leaves` 15–255 (log), `max_depth` −1 or 4–16, `learning_rate` 0.01–0.3 (log), `n_estimators` up to 2000 with early stopping on val, `min_child_samples` 10–200, `feature_fraction` 0.5–1.0, `bagging_fraction` 0.5–1.0, `lambda_l1/l2` 1e-8–10 (log).
  - RF: `n_estimators` 100–500, `max_depth` 8–32 or None, `min_samples_leaf` 1–50, `max_features` {sqrt, 0.3, 0.5}.
  - After HPO, tune the **decision threshold** on val probabilities (grid 0.05–0.95).
  - Save study (`outputs/hpo/*.db`), best params (`configs/best_params_<model>.yaml`), optimisation-history and parameter-importance plots.
- **Final training** (`scripts/50_train_final.py`): best params, 5 seeds, train on FLOGA train (val used only for early stopping / threshold). Save models (`outputs/models/`, keep each < 50 MB; if larger, attach to a GitHub release instead of committing).
- Record cost: train time, inference time per 1000 km² on CPU, model size, peak RAM, number of trees/leaves.

### Phase 6 — Evaluation, ablation, error analysis
- `scripts/70_evaluate.py`: all methods × {FLOGA test, Manavgat, Marmaris, Türkiye pooled}: Precision, Recall, F1, IoU, OA, predicted vs reference area (ha); mean ± std over seeds; block-bootstrap 95 % CIs on Türkiye; paired bootstrap CI of ΔF1 (ours − BAM-CD, ours − dNBR). Tables → `outputs/tables/*.csv` and LaTeX snippets.
- **Domain-shift result:** in-domain (FLOGA test) vs out-of-domain (Türkiye) drop per method — a key finding.
- `scripts/80_ablation.py`: LightGBM with G1, G2, G2+G3, ALL (same tuned params family, re-tuned briefly on val if needed — log it); report val and Türkiye F1.
- `scripts/90_error_analysis.py`: FP/FN by ESA WorldCover 2021 class (Planetary Computer collection `esa-worldcover`) and by dNBR severity band; maps of disagreements; short interpretation (unburnt islands inside EFFIS perimeters, agriculture, shadows, water).
- Figures (300 dpi PNG + PDF): study area map, method comparison maps per event (RGB, reference, dNBR, BAM-CD, ours), metric bar chart with CIs, accuracy-vs-cost scatter (F1 vs inference time / params, log axes), ablation chart, HPO history, error-by-land-cover chart.

### Phase 7 — Check-labelling kit (human step, optional)
- Generate `labeling/points.csv` + `points.geojson`: stratified random points (e.g. 150 per event: inside EFFIS, 0–1 km outside, 1–3 km outside), with links/coordinates and a 1-page Turkish instruction `labeling/NASIL_ETIKETLENIR.md` (inspect pre/post imagery in Copernicus Browser, fill `label` 0/1).
- `scripts/75_eval_checklabels.py` evaluates all methods on these points **if** labels are filled; otherwise the paper states this as future work. Add to `BLOCKERS.md` as optional human task. Do not fabricate labels.

### Phase 8 — Demo notebook
- `notebooks/00_demo_end_to_end.ipynb` for Colab free: clones repo, installs, downloads cached models/weights (GitHub release or FLOGA links), runs data access → preprocessing → dNBR + BAM-CD + ours on Manavgat (and Marmaris if time allows) → metrics table + figures. Print total runtime; must be ≤ 30 min on CPU-only Colab (use GPU only if available). Add an "Open in Colab" badge to README.
- Test it end-to-end locally in a clean venv (and record runtime in `docs/demo_runtime.md`).

### Phase 9 — Reproducibility and GitHub
- `README.md`: problem, contributions, results table (auto-generated), how to run (Colab + local), repo map, data, citation, AI-use statement.
- `DATA_MANIFEST.md`: every dataset with source, licence, access date, how the code obtains it.
- `CITATION.cff`, `environment.yml`, `requirements-lock.txt`, `pytest` passing.
- GitHub release **`v1.0`** with models and a zip of `outputs/` attached (`gh release create`). The paper's "Code and data availability" section cites repo URL + tag.

### Phase 10 — IEEE paper (`paper/`)
- IEEEtran conference template; **6–8 pages including figures**, references not counted. Required sections (course Section 9): Abstract (≤ 200 words) + Index Terms; Introduction (problem, user, solution, 2–3 contribution bullets); Related Work and Benchmark; Study Area and Data (source + licence of every dataset); Method (software architecture diagram, our method, benchmark implementations, parameters); Experiments and Results (fair comparison, uncertainty, ablation, error analysis); Discussion (where/why better, failures, limitations: EFFIS also derived from Sentinel-2, single-season test, two events); Conclusion and Future Work; Code and Data Availability; Author Contributions (CRediT) + AI-use statement; References (IEEE style, verified).
- All numbers via `\input{numbers.tex}` generated by `scripts/95_make_paper_numbers.py` from `outputs/tables` — no hand-typed results.
- CRediT roles: take them from `_inputs/team_roles.md`; if absent, write a clearly marked placeholder `\textbf{[TODO: confirm roles]}` and add it to `BLOCKERS.md` (do not invent who did what).
- AI-use statement must truthfully state that Claude Code (Anthropic) was used for code generation, experiment orchestration and drafting text, and that the authors reviewed and are responsible for all content.
- Compile to `paper/main.pdf` (`tectonic` or `latexmk`; install tectonic if neither exists). Check page count, figure readability, no overfull boxes; fix until clean.

### Phase 11 — Final presentation material
- `reports/final_presentation_outline.md`: 12-minute slide-by-slide outline following the final-presentation rubric (problem/solution 15, comparison & contribution 25, results & validation 20, demo 15, each member's part 10, Q&A 15), which figure goes on which slide, speaker split between the two members, and a 3-minute demo script.

### Phase 12 — Turkish final report for the student
- `reports/BITIRME_RAPORU_TR.md`: what the project does; every concept explained from zero (bands, NBR/dNBR, masks, mosaics, train/val/test, HPO, bootstrap, ablation, BAM-CD, transfer); exactly what was done in each phase and why; all final results with interpretation; limitations; how to re-run; list of files.
- `reports/SUNUM_VE_SORU_CEVAP_TR.md`: likely instructor questions (why Greece→Türkiye, why EFFIS, independence of the reference, why these thresholds, leakage prevention, why LightGBM, cost measurement, what if worse than BAM-CD, AI use) with short answers in Turkish and English; a code walk-through so each member can explain any file (course 10: oral check).
- `reports/YAPILACAKLAR_INSAN_TR.md`: remaining human tasks (Erdem's own commits/sections, CRediT confirmation, optional check labels, proofreading the paper in their own words, rehearsal).

---

## 5. Git workflow

- Commit after every completed task (small, meaningful commits; message format `phase N: <what>`). End each message with a blank line and `Co-Authored-By: Claude <noreply@anthropic.com>` (transparent AI use).
- `git push` after every phase (and before any long job). Never `push --force`, never rewrite published history, never delete branches.
- Never commit: `_inputs/`, `data/raw/`, `data/cache/`, `external/`, `.venv/`, logs > 5 MB, any file > 50 MB, secrets.
- Tag `v1.0` only when the definition of done below is met; later fixes → `v1.0.1`.

---

## 6. Definition of done (map to the grading rubric)

- [ ] Pipeline runs end to end from code with no manual step; `run_all.py` regenerates every table/figure.
- [ ] Demo notebook ≤ 30 min on Colab free, runtime documented.
- [ ] dNBR (USGS + FLOGA-val-tuned), BAM-CD and our models evaluated on identical FLOGA-test and Turkish grids.
- [ ] HPO done on FLOGA val only; best params and studies saved; Turkish data never used for tuning (a unit test asserts the HPO/threshold scripts never load Turkish data).
- [ ] 5 seeds, mean ± std; block-bootstrap CIs; ΔF1 CIs; cost table; ablation; error analysis by land cover.
- [ ] README, DATA_MANIFEST (all licences), CITATION.cff, environment files, tests pass, release `v1.0`.
- [ ] `paper/main.pdf` compiles, 6–8 pages, all required sections, numbers auto-generated, references verified.
- [ ] Presentation outline + demo script; Turkish final report, Q&A guide, human-TODO list.
- [ ] `PROGRESS.md` shows every phase done; `BLOCKERS.md` lists only genuinely human tasks.

When everything is done, print a short Turkish summary to the console: final results table, paths of the paper PDF and the Turkish report, release URL, and the human-TODO list.
