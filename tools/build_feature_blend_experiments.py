"""Compare partial primary feature TTA against E0053's full averaging.

Freeze tight55 before inference; omit the obsolete proxy-based parameter sweep.
Detection logits and their spatial transformations are untouched.
"""
import copy
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = root / "research_members/naveen/experiments/E0053_harmonic_frontier"
original = json.loads((parent / "biohub-e0053-harmonic-frontier.ipynb").read_text())
materialized = (root / "local_runs/frontier_20260909/harmonic/output/tracking_repo/scripts/predict_unet_transformer.py").read_text()
needle = "                unet_out = _unet_acc / _nv"
assert materialized.count(needle) == 1
parser = argparse.ArgumentParser()
parser.add_argument("--next-batch", action="store_true", help="Build E0061/E0062 without rewriting previous experiments")
args = parser.parse_args()
variants = [("E0061", 0.25), ("E0062", 0.0)] if args.next_batch else [("E0056", 0.75), ("E0057", 0.50)]
for exp, weight in variants:
    nb = copy.deepcopy(original)
    nb["cells"][0]["source"] = ("".join(nb["cells"][0]["source"]) +
        '\nos.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n').splitlines(True)
    replacement = f"                unet_out = {1-weight} * unet_out + {weight} * (_unet_acc / _nv)"
    compile(materialized.replace(needle, replacement), exp, "exec")
    patch = f'''
_feature_source = _ps.read_text()
_feature_old = {needle!r}
_feature_new = {replacement!r}
if _feature_source.count(_feature_old) != 1:
    raise RuntimeError("Primary feature-blend anchor mismatch")
_feature_source = _feature_source.replace(_feature_old, _feature_new, 1)
compile(_feature_source, str(_ps), "exec")
_ps.write_text(_feature_source)
print("PRIMARY_FEATURE_TTA_BLEND", {weight})
'''
    source = "".join(nb["cells"][4]["source"])
    anchor = "def list_test_stems() -> list[str]:"
    assert source.count(anchor) == 1
    nb["cells"][4]["source"] = source.replace(anchor, patch + "\n" + anchor).splitlines(True)
    nb["cells"][10]["source"] = '''
selected_label = "frozen_tight55"
selected_config = {}
(WORKING_DIR / "ppsweep_selected.json").write_text(json.dumps({
    "selected": selected_label,
    "overrides": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "held_out_stems": val_stems,
    "selection": "Frozen before execution; no proxy-based selection"
}, indent=2))
'''.splitlines(True)
    audit = "".join(nb["cells"][-1]["source"])
    audit = audit.replace('[("base", {}), ("source_selected", selected_config)]', '[("frozen_tight55", {})]')
    nb["cells"][-1]["source"] = audit.splitlines(True)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"{exp.lower()}-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{exp}:{i}", "exec")
    slug = f"biohub-{exp.lower()}-primary-tta-{int(weight*100)}"
    folder = root / "research_members/naveen/experiments" / f"{exp}_primary_tta_{int(weight*100)}"
    folder.mkdir(exist_ok=True)
    (folder / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    meta = json.loads((parent / "kernel-metadata.json").read_text())
    meta.update(id=f"naveenlx111249971939/{slug}", title=f"Biohub {exp} Primary TTA {int(weight*100)}",
                code_file=f"{slug}.ipynb")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "provenance.json").write_text(json.dumps({
        "parent": "E0053 source_selected tight55", "primary_feature_tta_weight": weight,
        "hypothesis": "Retain some native orientation features while averaging the remainder to improve association.",
        "unchanged": "Detection logits and all spatial transforms; secondary model; graph parameters frozen to tight55.",
        "validation": "Same eight diagnostic movies; official metric; compare pre-ILP detector hashes to E0053.",
        "selection": "No proxy sweep. No claim of independently held-out detector training."
    }, indent=2) + "\n")
    print(folder)
