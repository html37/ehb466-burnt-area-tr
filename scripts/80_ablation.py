"""Ablation: which feature group makes the difference? LightGBM with G1, G2, G2+G3 and ALL.

Decision (DECISIONS.md): every group reuses the LightGBM hyper-parameters tuned for ALL (configs/best_params_lgbm.yaml),
with early stopping and the decision threshold re-fitted per seed on FLOGA val; no re-tuning on Turkish data.
`ALL` is the main model `lgbm` itself.

--train : train the ablation models (5 seeds, FLOGA only) and predict Türkiye (calls 50_train_final / 65_predict).
--report: after scripts/70_evaluate.py, collect val / FLOGA-test / Türkiye F1 per group
          -> outputs/tables/ablation.csv, outputs/figures/ablation.png|pdf
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import out_dir, ROOT  # noqa: E402

GROUPS = {"G1": "lgbm_G1", "G2": "lgbm_G2", "G2+G3": "lgbm_G2_G3", "ALL": "lgbm"}
PY = sys.executable


def train():
    for g, name in GROUPS.items():
        if g == "ALL":
            continue
        tag = name[len("lgbm"):]
        if not (ROOT / "outputs" / "models" / name / "seed4.pkl").exists():
            subprocess.run([PY, str(ROOT / "scripts" / "50_train_final.py"), "--models", "lgbm", "--group", g,
                            "--tag", tag], check=True)
        subprocess.run([PY, str(ROOT / "scripts" / "65_predict_turkey.py"), "--models", name], check=True)


def report():
    t = ROOT / "outputs" / "tables"
    tr = pd.read_csv(t / "eval_turkey.csv")
    fl = pd.read_csv(t / "eval_floga_test.csv")
    rows = []
    for g, name in GROUPS.items():
        vf = [json.loads(f.read_text())["val_f1"] for f in sorted((ROOT / "outputs" / "models" / name).glob("seed*.json"))]
        r = {"group": g, "method": name, "n_features": None, "val_f1_mean": np.mean(vf), "val_f1_std": np.std(vf)}
        x = fl[fl["method"] == name]
        if len(x):
            r.update(floga_test_f1_mean=float(x["f1_mean"].iloc[0]), floga_test_f1_std=float(x["f1_std"].iloc[0]))
        for ev in tr["event"].unique():
            x = tr[(tr["method"] == name) & (tr["event"] == ev)]
            if len(x):
                r[f"{ev}_f1_mean"] = float(x["f1_mean"].iloc[0]); r[f"{ev}_f1_std"] = float(x["f1_std"].iloc[0])
        meta = json.loads(sorted((ROOT / "outputs" / "models" / name).glob("seed*.json"))[0].read_text())
        from burnt import features as F
        r["n_features"] = len(F.GROUPS[meta["group"]])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(t / "ablation.csv", index=False)
    print(df.round(4).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("val_f1", "FLOGA val"), ("floga_test_f1", "FLOGA test"), ("Manavgat_f1", "Manavgat"),
            ("Marmaris_f1", "Marmaris")]
    cols = [c for c in cols if f"{c[0]}_mean" in df]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = 0.8 / len(cols); x = np.arange(len(df))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (c, lab) in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, df[f"{c}_mean"], w, yerr=df[f"{c}_std"], capsize=2,
               label=lab, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n({n} feat.)" for g, n in zip(df["group"], df["n_features"])])
    ax.set_ylabel("F1 (burnt)"); ax.set_ylim(0, 1); ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.set_title("Ablation of feature groups (LightGBM, mean ± std over 5 seeds)")
    ax.grid(axis="y", alpha=0.3)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir("figures") / f"ablation.{ext}", dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.train:
        train()
    if a.report:
        report()


if __name__ == "__main__":
    main()
