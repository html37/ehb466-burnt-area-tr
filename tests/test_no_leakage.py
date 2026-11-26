"""Test isolation (CLAUDE.md rule 2.1): tuning/training scripts must never load Turkish data.

Static check of the source of every script that chooses thresholds, hyper-parameters, features or models, and of
the src modules they import; plus a dynamic check that importing them does not import burnt.turkey.
"""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUNING_SCRIPTS = ["20_prepare_floga.py", "30_tune_dnbr.py", "40_hpo.py", "50_train_final.py"]
FORBIDDEN_TOKENS = ["burnt.turkey", "from burnt import turkey", "turkey.load_event", "effis_TR_2021",
                    "Manavgat", "Marmaris", "cache/turkey", "\"turkey\"", "'turkey'", "reference.effis_mask"]
ALLOWED_TURKEY_USERS = {"10_build_turkey_testset.py", "60_run_bamcd.py", "65_predict_turkey.py", "70_evaluate.py",
                        "75_eval_checklabels.py", "80_ablation.py", "90_error_analysis.py",
                        "95_make_paper_numbers.py", "85_figures.py", "35_band_harmonisation.py", "72_cost_table.py", "74_make_labeling_kit.py", "75_eval_checklabels.py"}


def burnt_imports(src):
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("burnt"):
            if node.module == "burnt":
                mods |= {a.name for a in node.names}
            else:
                mods.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            mods |= {a.name.split(".")[1] for a in node.names if a.name.startswith("burnt.")}
    return mods


@pytest.mark.parametrize("script", TUNING_SCRIPTS)
def test_tuning_scripts_do_not_touch_turkey(script):
    src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    for tok in FORBIDDEN_TOKENS:
        assert tok not in src, f"{script} contains forbidden token {tok!r}"
    mods = burnt_imports(src)
    assert "turkey" not in mods
    # transitively: src modules imported by the script must not use the Turkish data either
    seen, todo = set(), set(mods)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        f = ROOT / "src" / "burnt" / f"{m}.py"
        if f.exists():
            msrc = f.read_text(encoding="utf-8")
            assert "turkey" not in burnt_imports(msrc), f"burnt.{m} imports burnt.turkey"
            todo |= burnt_imports(msrc)
    assert "turkey" not in seen


@pytest.mark.parametrize("script", ["40_hpo.py", "30_tune_dnbr.py"])
def test_import_does_not_load_turkey(script):
    for k in [k for k in sys.modules if k == "burnt.turkey"]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location(script[:-3].replace(".", "_"), ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "burnt.turkey" not in sys.modules


def test_only_whitelisted_scripts_use_turkey():
    for f in (ROOT / "scripts").glob("*.py"):
        if "turkey" in burnt_imports(f.read_text(encoding="utf-8")):
            assert f.name in ALLOWED_TURKEY_USERS, f"{f.name} uses Turkish data but is not an evaluation script"
