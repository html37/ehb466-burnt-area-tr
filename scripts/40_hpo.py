"""Hyper-parameter optimisation on the FLOGA validation split only (Optuna TPE + MedianPruner).

Training data: sampled FLOGA-train pixels (data/cache/floga/samples_train.parquet).
Objective: F1 (burnt class) on FLOGA-val pixels at the best decision threshold (grid 0.05-0.95). Val pixels =
all valid pixels of the val patches if <= MAX_VAL rows, otherwise a fixed uniform random subset (logged).
LightGBM early stopping uses the sampled val pixels (samples_val.parquet, log-loss).
Budget per model: 60 trials or 2 h, whichever first. Never reads Turkish data.

Outputs: outputs/hpo/<model>_<group>.db (study), configs/best_params_<model>[_<group>].yaml,
         outputs/figures/hpo_history_<model>.png|pdf, hpo_importance_<model>.png|pdf
Usage: python scripts/40_hpo.py --model lgbm [--group ALL] [--trials 60] [--timeout 7200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnt.config import load_config, out_dir, ROOT  # noqa: E402
from burnt import floga, features as F, models as M  # noqa: E402

MAX_VAL = 12_000_000
SUBSET_VAL = 6_000_000


def load_train_val(group, cfg):
    c = cfg["paths"]["cache"] / "floga"
    tr = pd.read_parquet(c / "samples_train.parquet")
    va = pd.read_parquet(c / "samples_val.parquet")
    return F.from_samples(tr, group), tr["label"].to_numpy(), F.from_samples(va, group), va["label"].to_numpy()


def load_val_full(group, cfg, seed=0):
    """Features of all valid FLOGA-val pixels (cached as parquet per group)."""
    f = cfg["paths"]["cache"] / "floga" / f"valfull_{group.replace('+', '_')}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return d.drop(columns="label"), d["label"].to_numpy(), json.loads(f.with_suffix(".json").read_text())
    pre, post, lab = [], [], []
    for _, p, m in floga.iter_split_pixels("val", cfg):
        pre.append(p["pre"][:, m].T); post.append(p["post"][:, m].T); lab.append(p["label"][m] == 1)
    pre, post, lab = np.concatenate(pre), np.concatenate(post), np.concatenate(lab)
    info = {"val_pixels_total": int(len(lab)), "subset": False}
    if len(lab) > MAX_VAL:
        idx = np.sort(np.random.default_rng(seed).choice(len(lab), SUBSET_VAL, replace=False))
        pre, post, lab = pre[idx], post[idx], lab[idx]
        info.update(subset=True, subset_size=SUBSET_VAL)
    X = F.build(pre, post, group)
    d = X.copy(); d["label"] = lab.astype(np.uint8)
    d.to_parquet(f); f.with_suffix(".json").write_text(json.dumps(info))
    return X, lab.astype(np.uint8), info


def space(trial, model):
    if model == "lgbm":
        depth = trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12, 14, 16])
        return {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True), "max_depth": depth,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": 2000,
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "colsample_bytree": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "subsample": trial.suggest_float("bagging_fraction", 0.5, 1.0), "subsample_freq": 1,
                "reg_alpha": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("lambda_l2", 1e-8, 10, log=True)}
    if model == "rf":
        md = trial.suggest_categorical("max_depth", [8, 12, 16, 20, 24, 28, 32, "None"])
        mf = trial.suggest_categorical("max_features", ["sqrt", "0.3", "0.5"])
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": None if md == "None" else int(md),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
                "max_features": mf if mf == "sqrt" else float(mf)}
    if model == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
    raise ValueError(model)


class PruneCallback:
    """LightGBM callback: report val log-loss every 50 iterations to Optuna and prune if needed."""

    def __init__(self, trial):
        self.trial = trial

    def __call__(self, env):
        if env.iteration % 50 == 0 and env.evaluation_result_list:
            self.trial.report(env.evaluation_result_list[0][2], step=env.iteration)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lgbm", "rf", "logreg"])
    ap.add_argument("--group", default=None)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    group = args.group or ("G2" if args.model == "logreg" else "ALL")
    cfg = load_config()
    seed = cfg["seed"]
    Xtr, ytr, Xva_s, yva_s = load_train_val(group, cfg)
    Xvf, yvf, vinfo = load_val_full(group, cfg)
    print(f"train {Xtr.shape} (burnt {ytr.mean():.3f}), val-sample {Xva_s.shape}, val-full {Xvf.shape} {vinfo}",
          flush=True)

    def objective(trial):
        p = space(trial, args.model)
        t = time.time()
        if args.model == "lgbm":
            import lightgbm as lgb
            clf = M.make("lgbm", p, seed)
            clf.fit(Xtr, ytr, eval_set=[(Xva_s, yva_s)], eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False), PruneCallback(trial)])
            trial.set_user_attr("best_iteration", int(clf.best_iteration_ or p["n_estimators"]))
        else:
            clf = M.fit(args.model, p, Xtr, ytr, seed)
        prob = M.predict_proba(clf, Xvf)
        thr, f1 = M.best_threshold(prob, yvf)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("fit_predict_s", time.time() - t)
        return f1

    hpo = out_dir("hpo")
    name = f"{args.model}_{group.replace('+', '_')}"
    study = optuna.create_study(direction="maximize", study_name=name,
                                storage=f"sqlite:///{(hpo / (name + '.db')).as_posix()}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200))
    done = len([t for t in study.trials if t.state.is_finished()])
    if done < args.trials:
        study.optimize(objective, n_trials=args.trials - done, timeout=args.timeout, gc_after_trial=True)
    bt = study.best_trial
    params = dict(bt.params)
    if args.model == "lgbm":
        params = {"num_leaves": params["num_leaves"], "max_depth": params["max_depth"],
                  "learning_rate": params["learning_rate"], "min_child_samples": params["min_child_samples"],
                  "colsample_bytree": params["feature_fraction"], "subsample": params["bagging_fraction"],
                  "subsample_freq": 1, "reg_alpha": params["lambda_l1"], "reg_lambda": params["lambda_l2"],
                  "n_estimators": 2000}
    elif args.model == "rf":
        params["max_depth"] = None if params["max_depth"] == "None" else int(params["max_depth"])
        params["max_features"] = params["max_features"] if params["max_features"] == "sqrt" else float(params["max_features"])
    out = {"model": args.model, "group": group, "params": params, "val_f1": float(bt.value),
           "threshold": bt.user_attrs.get("threshold"), "best_iteration": bt.user_attrs.get("best_iteration"),
           "n_trials": len(study.trials),
           "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
           "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
           "val_info": vinfo}
    fn = ROOT / "configs" / (f"best_params_{args.model}.yaml" if args.group is None
                             else f"best_params_{args.model}_{group.replace('+', '_')}.yaml")
    fn.write_text(yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out, sort_keys=False))
    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    figs = out_dir("figures")
    tag = args.model if args.group is None else name
    ax = plot_optimization_history(study); ax.set_title(f"HPO history ({args.model}, {group}) - F1 on FLOGA val")
    for ext in ["png", "pdf"]:
        ax.figure.savefig(figs / f"hpo_history_{tag}.{ext}", dpi=300, bbox_inches="tight")
    plt.close("all")
    if len(study.trials) > 2 and len(bt.params) > 1:
        try:
            ax = plot_param_importances(study)
            for ext in ["png", "pdf"]:
                ax.figure.savefig(figs / f"hpo_importance_{tag}.{ext}", dpi=300, bbox_inches="tight")
        except Exception as e:  # importance needs >1 completed trial with variance
            print("importance plot skipped:", e)
        plt.close("all")


if __name__ == "__main__":
    main()
