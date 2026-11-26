"""Reproduce every table and figure of the project in order, skipping steps whose outputs already exist.

    python run_all.py            # run missing steps only
    python run_all.py --force    # redo every step (scripts that cache intermediate data keep their caches)
    python run_all.py --list     # show the steps and whether they are done

Order: Turkish test set -> FLOGA preparation -> dNBR tuning -> band harmonisation -> HPO -> final training ->
BAM-CD -> Turkish predictions -> ablation models -> evaluation -> cost table -> ablation report -> error analysis
-> figures -> paper numbers -> unit tests.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "scripts"
T = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

STEPS = [
    ("Turkish test set (Planetary Computer)", ["10_build_turkey_testset.py"],
     [ROOT / "data/cache/turkey/Manavgat.npz", ROOT / "data/cache/turkey/Marmaris.npz"]),
    ("FLOGA patches and samples (Hugging Face)", ["20_prepare_floga.py"],
     [ROOT / "data/cache/floga/samples_train.parquet", T / "floga_patch_counts.json"]),
    ("dNBR threshold on FLOGA val", ["30_tune_dnbr.py"], [T / "dnbr_floga_threshold.json"]),
    ("Band harmonisation check", ["35_band_harmonisation.py"], [FIG / "band_harmonisation.png"]),
    ("HPO LightGBM", ["40_hpo.py", "--model", "lgbm"], [ROOT / "configs/best_params_lgbm.yaml"]),
    ("HPO Random Forest", ["40_hpo.py", "--model", "rf"], [ROOT / "configs/best_params_rf.yaml"]),
    ("HPO Logistic Regression (G2)", ["40_hpo.py", "--model", "logreg"], [ROOT / "configs/best_params_logreg.yaml"]),
    ("Final training, 5 seeds", ["50_train_final.py"], [T / "cost_train.csv"]),
    ("BAM-CD (FLOGA test sanity + Türkiye)", ["60_run_bamcd.py"], [T / "cost_bamcd.csv", T / "bamcd_sanity.csv"]),
    ("Predict Türkiye with our models", ["65_predict_turkey.py"], [T / "cost_infer_turkey.csv"]),
    ("Ablation models", ["80_ablation.py", "--train"], [ROOT / "outputs/models/lgbm_G2_G3/seed4.pkl"]),
    ("Evaluation (all methods, bootstrap)", ["70_evaluate.py"], [T / "eval_turkey.csv", T / "delta_f1_turkey.csv"]),
    ("Cost table", ["72_cost_table.py"], [T / "cost_table.csv"]),
    ("Ablation report", ["80_ablation.py", "--report"], [T / "ablation.csv"]),
    ("Error analysis", ["90_error_analysis.py"], [T / "errors_by_landcover.csv"]),
    ("Figures", ["85_figures.py"], [FIG / "metrics_ci.png", FIG / "cost_accuracy.png"]),
    ("Paper numbers", ["95_make_paper_numbers.py"], [ROOT / "paper/numbers.tex"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for name, cmd, outs in STEPS:
        done = all(o.exists() for o in outs)
        if a.list:
            print(f"[{'x' if done else ' '}] {name}: {' '.join(cmd)}")
            continue
        if done and not a.force:
            print(f"skip  {name} (outputs exist)")
            continue
        print(f"run   {name}: {' '.join(cmd)}", flush=True)
        t = time.time()
        subprocess.run([sys.executable, str(S / cmd[0]), *cmd[1:]], check=True, cwd=ROOT)
        print(f"done  {name} in {time.time() - t:.0f} s", flush=True)
    if not a.list:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)
        print(f"all steps finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
