"""Generate paper/numbers.tex (LaTeX macros) and paper/tab_*.tex (tables) from outputs/tables/*.csv.

No number in the paper is typed by hand: every value comes from these generated files.
Macro naming: \\<Metric><Method><Where>, e.g. \\FoneLgbmPooled, \\IoUBamcdOneManavgat, \\FoneDnbrFlogaFloga.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
PAPER = ROOT / "paper"

MNAME = {"lgbm": "Lgbm", "rf": "Rf", "logreg": "Logreg", "bamcd_v1": "BamcdOne", "bamcd_v2": "BamcdTwo",
         "dnbr_usgs_low": "DnbrLow", "dnbr_usgs_mod": "DnbrMod", "dnbr_floga": "DnbrFloga",
         "lgbm_G1": "AblGOne", "lgbm_G2": "AblGTwo", "lgbm_G2_G3": "AblGTwoThree"}
LABEL = {"dnbr_usgs_mod": "dNBR $>$ 0.27 (USGS)", "dnbr_usgs_low": "dNBR $>$ 0.10 (USGS)",
         "dnbr_floga": "dNBR, FLOGA-val thr.", "bamcd_v1": "BAM-CD v1", "bamcd_v2": "BAM-CD v2",
         "logreg": "LogReg (G2, ours)", "rf": "Random Forest (ours)", "lgbm": "LightGBM (ours)"}
EVN = {"Manavgat": "Manavgat", "Marmaris": "Marmaris", "Türkiye (pooled)": "Pooled"}
ORDER = ["dnbr_usgs_mod", "dnbr_usgs_low", "dnbr_floga", "bamcd_v1", "bamcd_v2", "logreg", "rf", "lgbm"]
lines = []
_defined = {}


def mac(name, val, fmt="{:.3f}"):
    name = re.sub(r"[^A-Za-z]", "", name)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        s = "n/a"
    elif isinstance(val, str):
        s = val
    else:
        s = fmt.format(val)
    if name in _defined:  # same quantity emitted by several loops (e.g. reference area): define once
        if _defined[name] != s:
            raise ValueError(f"macro {name} defined with two values: {_defined[name]} vs {s}")
        return
    _defined[name] = s
    lines.append(f"\\newcommand{{\\{name}}}{{{s}}}")


def num_words(n):
    return f"{int(round(n)):,}".replace(",", "\\,")


def main():
    ev = pd.read_csv(T / "eval_turkey.csv")
    for _, r in ev.iterrows():
        if r["method"] not in MNAME:
            continue
        w = EVN[r["event"]]; m = MNAME[r["method"]]
        for k, lab in [("f1", "Fone"), ("iou", "IoU"), ("precision", "Prec"), ("recall", "Rec"), ("oa", "OA")]:
            mac(f"{lab}{m}{w}", r[f"{k}_mean"])
            mac(f"{lab}Std{m}{w}", r[f"{k}_std"])
        mac(f"Area{m}{w}", num_words(r["pred_area_ha_mean"]), "{}")
        mac(f"RefArea{w}", num_words(r["ref_area_ha_mean"]), "{}")
    boot = pd.read_csv(T / "bootstrap_turkey.csv")
    for _, r in boot.iterrows():
        if r["method"] in MNAME:
            w = EVN[r["event"]]; m = MNAME[r["method"]]
            mac(f"FoneLo{m}{w}", r["f1_lo"]); mac(f"FoneHi{m}{w}", r["f1_hi"])
    dl = pd.read_csv(T / "delta_f1_turkey.csv")
    for _, r in dl.iterrows():
        a, b = r["pair"].split(" - ")
        if a in MNAME and b in MNAME:
            w = EVN[r["event"]]
            p = f"Delta{MNAME[a]}{MNAME[b]}{w}"
            mac(p, r["delta_f1"], "{:+.3f}"); mac(p + "Lo", r["lo"], "{:+.3f}"); mac(p + "Hi", r["hi"], "{:+.3f}")
    if (T / "eval_floga_test.csv").exists():
        fl = pd.read_csv(T / "eval_floga_test.csv")
        for _, r in fl.iterrows():
            if r["method"] in MNAME:
                m = MNAME[r["method"]]
                for k, lab in [("f1", "Fone"), ("iou", "IoU"), ("precision", "Prec"), ("recall", "Rec")]:
                    mac(f"{lab}{m}Floga", r[f"{k}_mean"]); mac(f"{lab}Std{m}Floga", r[f"{k}_std"])
    if (T / "bamcd_sanity.csv").exists():
        s = pd.read_csv(T / "bamcd_sanity.csv")
        for _, r in s.iterrows():
            mac(f"Fone{MNAME[r['method']]}{'Val' if r['split'] == 'val' else 'FlogaSanity'}", r["f1"])
    th = json.loads((T / "dnbr_floga_threshold.json").read_text())
    mac("DnbrFlogaThr", th["threshold"], "{:.2f}"); mac("DnbrFlogaValFone", th["val_f1"])
    cnt = json.loads((T / "floga_patch_counts.json").read_text())
    for k, v in cnt.items():
        mac(f"Patches{k.capitalize()}", str(v))
    em = pd.read_csv(T / "floga_event_mapping.csv")
    mac("EventsMapped", str(int((em["key"].notna() & ~em["dup_dropped"]).sum()))); mac("EventsSplit", str(len(em)))
    for s in ["train", "val", "test"]:
        mac(f"Events{s.capitalize()}", str(int(((em["set"] == s) & em["key"].notna()).sum())))
    for split in ["train", "val"]:
        f = ROOT / "data" / "cache" / "floga" / f"samples_{split}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["label"])
            mac(f"Samples{split.capitalize()}", num_words(len(d)), "{}")
            mac(f"SamplesBurnt{split.capitalize()}", num_words(d["label"].sum()), "{}")
    for model in ["lgbm", "rf", "logreg"]:
        f = ROOT / "configs" / f"best_params_{model}.yaml"
        if f.exists():
            import yaml
            bp = yaml.safe_load(f.read_text())
            m = MNAME[model]
            mac(f"HpoValFone{m}", bp["val_f1"]); mac(f"HpoTrials{m}", str(bp["n_trials"]))
            mac(f"HpoComplete{m}", str(bp["n_complete"])); mac(f"HpoPruned{m}", str(bp["n_pruned"]))
            for k, v in bp["params"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    mac(f"Hp{m}{k.title().replace('_', '')}", v, "{:.3g}" if isinstance(v, float) else "{}")
                else:
                    mac(f"Hp{m}{k.title().replace('_', '')}", str(v))
    # cost
    c = pd.read_csv(T / "cost_table.csv").set_index("method")
    for m, r in c.iterrows():
        if m not in MNAME:
            continue
        M = MNAME[m]
        mac(f"Params{M}", num_words(r["params"]) if pd.notna(r.get("params")) else "n/a", "{}")
        mac(f"CpuTime{M}", r.get("cpu_s_per_1000km2"), "{:.1f}")
        mac(f"GpuTime{M}", r.get("gpu_s_per_1000km2"), "{:.1f}")
        mac(f"ModelMB{M}", r.get("model_mb"), "{:.1f}")
        mac(f"TrainTime{M}", r.get("train_time_s"), "{:.0f}")
        mac(f"CpuRam{M}", r.get("cpu_peak_rss_mb"), "{:.0f}")
    if "lgbm" in c.index and "bamcd_v1" in c.index:
        mac("SpeedupCpuLgbmBamcd", c.loc["bamcd_v1", "cpu_s_per_1000km2"] / c.loc["lgbm", "cpu_s_per_1000km2"], "{:.0f}")
        mac("ParamRatioBamcdLgbm", c.loc["bamcd_v1", "params"] / c.loc["lgbm", "params"], "{:.0f}")
    # ablation
    if (T / "ablation.csv").exists():
        ab = pd.read_csv(T / "ablation.csv")
        for _, r in ab.iterrows():
            g = {"G1": "GOne", "G2": "GTwo", "G2+G3": "GTwoThree", "ALL": "All"}[r["group"]]
            for col, lab in [("val_f1_mean", "Val"), ("floga_test_f1_mean", "Floga"), ("Manavgat_f1_mean", "Manavgat"),
                             ("Marmaris_f1_mean", "Marmaris")]:
                if col in r:
                    mac(f"Abl{g}{lab}", r[col])
    # domain shift
    if (T / "domain_shift.csv").exists():
        ds = pd.read_csv(T / "domain_shift.csv")
        for _, r in ds.iterrows():
            if r["method"] in MNAME:
                mac(f"Shift{MNAME[r['method']]}", r["delta"], "{:+.3f}")
    # harmonisation
    if (T / "band_harmonisation.csv").exists():
        bh = pd.read_csv(T / "band_harmonisation.csv")
        med = bh.pivot_table(index="band", columns="domain", values="median")
        fl = med["FLOGA (Greece) pre"]
        tr = med[[c for c in med.columns if c.endswith(" pre") and not c.startswith("FLOGA")]].mean(1)
        mac("HarmMaxAbsDiff", float((tr - fl).abs().max()))
    write_severity()  # adds macros, so it must run before numbers.tex is written
    (PAPER / "numbers.tex").write_text("% AUTO-GENERATED by scripts/95_make_paper_numbers.py - do not edit\n" +
                                       "\n".join(lines) + "\n", encoding="utf-8")
    write_tables(ev, boot, c)
    write_scenes()
    write_readme(ev, boot, c)
    copy_figures()
    print(f"{len(lines)} macros written to paper/numbers.tex")


def write_severity():
    """Omission rate of the EFFIS-burnt pixels per dNBR severity band (both events pooled) -> tab_severity.tex,
    plus the share of EFFIS-burnt pixels that show (almost) no spectral change (dNBR < 0.10)."""
    f = T / "errors_by_severity.csv"
    if not f.exists():
        return
    sv = pd.read_csv(f).groupby(["method", "severity"], sort=False)[["ref_burnt", "fn", "pixels", "fp"]].sum()
    meths = [m for m in ["dnbr_floga", "bamcd_v1", "rf", "lgbm"] if m in sv.index.get_level_values(0)]
    sev = list(dict.fromkeys(sv.index.get_level_values(1)))
    ref = sv.loc[meths[0]]["ref_burnt"]
    share = ref / ref.sum()
    mac("SevShareUnchanged", 100 * float(share.iloc[0]), "{:.1f}")
    rows = []
    for s in sev:
        cells = [s.replace("<", "$<$").replace(">=", "$\\geq$"), f"{100 * share[s]:.1f}"]
        for m in meths:
            r = sv.loc[(m, s)]
            cells.append(f"{100 * r['fn'] / r['ref_burnt']:.1f}" if r["ref_burnt"] else "--")
        rows.append(" & ".join(cells) + r" \\")
    for m in meths:
        r = sv.loc[(m, sev[0])]
        mac(f"SevOmitUnchanged{MNAME.get(m, m)}", 100 * r["fn"] / r["ref_burnt"], "{:.1f}")
    head = " & ".join([r"dNBR band", r"\% of ref."] + [LABEL[m].replace(" (ours)", "") for m in meths]) + r" \\"
    tex = [r"\begin{tabular}{lr" + "r" * len(meths) + "}", r"\toprule", head, r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_severity.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_scenes():
    """Sentinel-2 scenes of the Turkish test set (from outputs/tables/selected_scenes.csv) -> tab_scenes.tex."""
    f = T / "selected_scenes.csv"
    if not f.exists():
        return
    s = pd.read_csv(f, dtype={"processing_baseline": str})
    rows = []
    for (ev, role), g in s.groupby(["event", "role"], sort=False):
        rows.append(" & ".join([ev, role, str(g["date"].iloc[0]), g["platform"].iloc[0].replace("Sentinel-", "S"),
                                "+".join(sorted(g["mgrs_tile"])), str(g["processing_baseline"].iloc[0]),
                                f"{g['aoi_clear_pct'].min():.1f}"]) + r" \\")
    tex = [r"\begin{tabular}{llllllr}", r"\toprule",
           r"Event & Date & & Sat. & Tiles & Baseline & Clear \% \\", r"\midrule", *rows, r"\bottomrule",
           r"\end{tabular}"]
    tex[2] = r"Event & Role & Date & Sat. & Tiles & Baseline & AOI clear \% \\"
    (PAPER / "tab_scenes.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


PAPER_FIGS = ["study_area", "compare_Manavgat", "compare_Marmaris", "metrics_ci", "cost_accuracy", "ablation",
              "errors_by_landcover", "band_harmonisation", "hpo_history_lgbm"]


def copy_figures():
    """Copy the (vector) figures used by the paper from outputs/figures to paper/figures."""
    import shutil
    d = PAPER / "figures"
    d.mkdir(exist_ok=True)
    for n in PAPER_FIGS:
        f = ROOT / "outputs" / "figures" / f"{n}.pdf"
        if f.exists():
            shutil.copy2(f, d / f.name)


def write_readme(ev, boot, cost):
    """Replace the block between RESULTS markers in README.md with a table generated from the outputs."""
    fl = pd.read_csv(T / "eval_floga_test.csv").set_index("method") if (T / "eval_floga_test.csv").exists() else None
    lab = {k: v.replace("$>$", ">") for k, v in LABEL.items()}
    out = ["| Method | FLOGA test F1 | Manavgat F1 | Marmaris F1 | Türkiye pooled F1 [95% CI] | CPU s / 1000 km² | Params |",
           "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        if m not in ev["method"].values:
            continue
        def f(e):
            r = ev[(ev.method == m) & (ev.event == e)].iloc[0]
            return f"{r['f1_mean']:.3f}" + (f" ± {r['f1_std']:.3f}" if r["f1_std"] > 0 else "")
        b = boot[(boot.method == m) & (boot.event == "Türkiye (pooled)")]
        ci = f" [{b['f1_lo'].iloc[0]:.3f}, {b['f1_hi'].iloc[0]:.3f}]" if len(b) else ""
        flt = (f"{fl.loc[m, 'f1_mean']:.3f}" + (f" ± {fl.loc[m, 'f1_std']:.3f}" if fl.loc[m, 'f1_std'] > 0 else "")) if fl is not None and m in fl.index else "–"
        cpu = f"{cost.loc[m, 'cpu_s_per_1000km2']:.1f}" if m in cost.index and pd.notna(cost.loc[m, "cpu_s_per_1000km2"]) else "–"
        par = num_words(cost.loc[m, "params"]).replace("\\,", " ") if m in cost.index and pd.notna(cost.loc[m, "params"]) else "–"
        out.append(f"| {lab[m]} | {flt} | {f('Manavgat')} | {f('Marmaris')} | {f('Türkiye (pooled)')}{ci} | {cpu} | {par} |")
    rd = ROOT / "README.md"
    s = rd.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in s and b in s:
        s = s[:s.index(a) + len(a)] + "\n" + "\n".join(out) + "\n" + s[s.index(b):]
        rd.write_text(s, encoding="utf-8")


def fmt_pm(m, s, bold=False):
    x = f"{m:.3f}" + (f"$\\pm${s:.3f}" if s and s > 0 else "")
    return f"\\textbf{{{x}}}" if bold else x


def write_tables(ev, boot, cost):
    # Table: Turkish results (F1, IoU, P, R per event + pooled F1 CI)
    rows = []
    fl = pd.read_csv(T / "eval_floga_test.csv").set_index("method") if (T / "eval_floga_test.csv").exists() else None
    best = {e: ev[(ev.event == e) & ev.method.isin(ORDER)]["f1_mean"].max() for e in EVN}
    for m in ORDER:
        if m not in ev["method"].values:
            continue
        cells = [LABEL[m]]
        if fl is not None and m in fl.index:
            cells.append(fmt_pm(fl.loc[m, "f1_mean"], fl.loc[m, "f1_std"]))
        else:
            cells.append("--")
        for e in ["Manavgat", "Marmaris"]:
            r = ev[(ev.method == m) & (ev.event == e)].iloc[0]
            cells += [fmt_pm(r["f1_mean"], r["f1_std"], np.isclose(r["f1_mean"], best[e])), f"{r['iou_mean']:.3f}"]
        r = ev[(ev.method == m) & (ev.event == "Türkiye (pooled)")].iloc[0]
        b = boot[(boot.method == m) & (boot.event == "Türkiye (pooled)")]
        ci = f"[{b['f1_lo'].iloc[0]:.3f}, {b['f1_hi'].iloc[0]:.3f}]" if len(b) else ""
        cells += [fmt_pm(r["f1_mean"], r["f1_std"], np.isclose(r["f1_mean"], best["Türkiye (pooled)"])), ci]
        rows.append(" & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & FLOGA test & \multicolumn{2}{c}{Manavgat} & \multicolumn{2}{c}{Marmaris} & \multicolumn{2}{c}{T\"urkiye pooled} \\",
           r"Method & F1 & F1 & IoU & F1 & IoU & F1 & 95\% CI \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    # Table: cost
    rows = []
    for m in ["dnbr_usgs_low", "bamcd_v1", "bamcd_v2", "logreg", "rf", "lgbm"]:
        if m not in cost.index:
            continue
        r = cost.loc[m]
        g = lambda k, f: (f.format(r[k]) if k in r and pd.notna(r[k]) else "--")
        rows.append(" & ".join([LABEL[m].replace(" (ours)", "").replace(" (USGS)", ""),
                                (num_words(r["params"]) if pd.notna(r.get("params")) else "--"),
                                g("model_mb", "{:.1f}"), g("train_time_s", "{:.0f}"),
                                g("cpu_s_per_1000km2", "{:.1f}"), g("gpu_s_per_1000km2", "{:.1f}"),
                                g("cpu_peak_rss_mb", "{:.0f}"), g("f1_turkey", "{:.3f}")]) + r" \\")
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"Method & Params & MB & Train s & CPU s & GPU s & RAM MB & F1 TR \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_cost.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    if (T / "ablation.csv").exists():
        ab = pd.read_csv(T / "ablation.csv")
        rows = []
        for _, r in ab.iterrows():
            cells = [r["group"].replace("+", "+"), str(int(r["n_features"])), fmt_pm(r["val_f1_mean"], r["val_f1_std"])]
            for c in ["floga_test_f1", "Manavgat_f1", "Marmaris_f1"]:
                cells.append(fmt_pm(r[f"{c}_mean"], r[f"{c}_std"]) if f"{c}_mean" in r and pd.notna(r[f"{c}_mean"]) else "--")
            rows.append(" & ".join(cells) + r" \\")
        tex = [r"\begin{tabular}{lrcccc}", r"\toprule",
               r"Features & \# & FLOGA val & FLOGA test & Manavgat & Marmaris \\", r"\midrule", *rows,
               r"\bottomrule", r"\end{tabular}"]
        (PAPER / "tab_ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
