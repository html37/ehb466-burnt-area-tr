# Final presentation outline — 28 Dec 2026 (12 min talk + ≤ 3 min demo + 5 min Q&A)

Rubric (course Section 8): problem & solution 15 · comparison & contribution 25 · results & validation 20 ·
demo 15 · each member presents own part, delivery & timing 10 · Q&A 15.
Numbers on the slides: copy them from the auto-generated block at the end of this file (from `outputs/tables`).

**Speaker split (placeholder — confirm the real task split, see `reports/YAPILACAKLAR_INSAN_TR.md`):**
Speaker A = slides 1–5 and the demo; Speaker B = slides 6–11. Each speaker presents the parts they owned in the
project (CRediT roles in the paper).

| # | Time | Slide | Content | Figure / table | Rubric |
|---|---|---|---|---|---|
| 1 | 0:00–0:40 | Title | Title, names, İTÜ EHB 466E; one-sentence claim: "a CPU-only model trained only on Greek fires maps the 2021 Manavgat/Marmaris burns close to a deep network at a fraction of its cost" | — | Problem 15 |
| 2 | 0:40–1:40 | Problem & users | 2021 fires, > 60 000 ha; OGM/AFAD need burnt-area maps fast; manual mapping is slow; no Turkish labelled dataset | `outputs/figures/study_area.png` | Problem 15 |
| 3 | 1:40–2:40 | Proposed solution | Bi-temporal Sentinel-2 → per-pixel LightGBM trained on FLOGA (Greece) → applied unchanged to Türkiye; contributions (3 bullets) | architecture (paper Fig. 1) | Problem 15, Contribution 25 |
| 4 | 2:40–3:40 | Data | Sentinel-2 L2A via Planetary Computer (mosaic, SCL masks, 20 m); EFFIS reference; FLOGA (326 events, CC BY 4.0) read remotely from a 1.1 TB release; licences | `band_harmonisation.png` | Results/validation 20 |
| 5 | 3:40–4:40 | Benchmarks & fair comparison | dNBR (USGS + FLOGA-tuned), BAM-CD official weights (sanity check on FLOGA test vs paper 88.48); identical grid, mask, reference; Turkish data only for testing (unit test) | `bamcd_sanity.csv` | Comparison 25 |
| 6 | 4:40–5:40 | Our method | 36 features (indices + bands), LightGBM; HPO with Optuna on FLOGA val; 5 seeds; threshold on val | `hpo_history_lgbm.png` | Contribution 25 |
| 7 | 5:40–7:10 | Main results | F1 / IoU per event + pooled with 95 % block-bootstrap CIs; ΔF1 ours − BAM-CD and ours − dNBR with CIs | `metrics_ci.png`, paper Table II | Results 20 |
| 8 | 7:10–8:10 | Maps | TP/FP/FN maps for both events (dNBR, BAM-CD, ours) | `compare_Manavgat.png`, `compare_Marmaris.png` | Results 20 |
| 9 | 8:10–9:10 | Cost advantage | Params, model size, CPU/GPU time per 1000 km², RAM; accuracy-vs-cost plot | `cost_accuracy.png`, paper Table III | Contribution 25 |
| 10 | 9:10–10:10 | Ablation & domain shift | G1 → G2 → G2+G3 → ALL; FLOGA-test vs Türkiye F1 per method (domain shift) | `ablation.png`, `domain_shift.csv` | Comparison 25 |
| 11 | 10:10–11:10 | Error analysis & limitations | FP/FN by land cover (WorldCover), unburnt islands, agriculture; severity table: EFFIS-burnt pixels with almost no spectral change explain most of BAM-CD's recall advantage (paper Table VII); EFFIS is also Sentinel-2-based; two events, one season; honest statement where BAM-CD is better | `errors_by_landcover.png`, `disagreement_*.png` | Results 20 |
| 12 | 11:10–12:00 | Conclusion | Contributions recap, future work (check labels, more Turkish events, SAR), repository + release v1.0 QR code | — | Problem 15 |
| D | 12:00–15:00 | Demo (≤ 3 min) | see script below | live Colab or recording | Demo 15 |

## Demo script (≤ 3 min) — record a backup video beforehand

1. (0:00) Open the README on GitHub → click **Open in Colab** (notebook `00_demo_end_to_end.ipynb`). Say: "everything
   runs from code: data access, preprocessing, three methods, metrics".
2. (0:20) Show cell 1–2 already executed (clone + pip). Explain that pretrained LightGBM models come from release v1.0
   and BAM-CD weights from the official FLOGA link.
3. (0:40) Run (or show the finished output of) `demo_run.py --events Manavgat`: point at the printed log lines —
   Sentinel-2 scenes found on Planetary Computer, tiles mosaicked, 20 m grid.
4. (1:30) Show the metrics table (dNBR 0.10, dNBR FLOGA threshold, LightGBM, BAM-CD) and the error map figure
   (red TP, orange FP, blue FN).
5. (2:20) Show the runtime print: total minutes and the LightGBM vs BAM-CD inference seconds on CPU.
6. (2:45) Close: "full experiment with 5 seeds and bootstrap: `python run_all.py`".
Tip: run the notebook once ~1 hour before the talk so the Colab VM has the packages cached; if the network fails,
play the recorded video.

## Key numbers (auto-generated)
<!-- KEYNUMBERS:START -->
- FLOGA test F1: BAM-CD v1 0.909 (makale 0.8848), LightGBM 0.864, RF 0.865, dNBR (FLOGA eşiği) 0.661
- Türkiye birleşik F1: LightGBM 0.878 [0.871, 0.887], BAM-CD v1 0.912, BAM-CD v2 0.925, dNBR 0.10 0.889, dNBR FLOGA eşiği 0.824
- Manavgat F1: LightGBM 0.882, BAM-CD v1 0.921; Marmaris F1: LightGBM 0.855, BAM-CD v1 0.866
- ΔF1 (LightGBM − BAM-CD v1), birleşik: -0.033 [-0.040, -0.027]; (LightGBM − dNBR FLOGA): +0.055
- Maliyet: LightGBM 35 834 parametre (ağaç düğümü), 1.9 MB, CPU 2.9 s/1000 km²; BAM-CD v1 83 727 022 parametre, CPU 50.2 s, GPU 5.4 s/1000 km²
- HPO: LightGBM 60 deneme, doğrulama F1 0.845; RF 14 deneme (2 saat sınırı), F1 0.846; dNBR eşiği 0.24
- Ablasyon (FLOGA doğrulama F1): G1 0.678, G2 0.839, G2+G3 0.841, ALL 0.844
- Veri: FLOGA 327/338 olay, yama train/val/test 538/184/162, eğitim örneği 710 625 piksel
<!-- KEYNUMBERS:END -->
