"""E0066: cached-graph DeepCenter TTA audit; no local model execution."""
import ast
import copy
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_dir = root / "research_members/naveen/experiments/E0065_kaggle_division_ranker"
nb = json.loads((source_dir / "biohub-e0065-kaggle-division-ranker.ipynb").read_text())
nb["cells"] = copy.deepcopy(nb["cells"][:6])
public = json.loads((root / "local_runs/frontier_20260910/dctta/biohub-lf-dctta.ipynb").read_text())
text = "".join(public["cells"][5]["source"])
fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "deepcenter_heatmap_for_frame")
public_fn = ast.get_source_segment(text, fn)
true_fn = public_fn.replace(
    'torch_mod.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)',
    'tensor.flip((-2, -1)).transpose(-1, -2)').replace(
    'torch_mod.rot90(model(at).transpose(-1, -2), -1, dims=(-2, -1))',
    'model(at).transpose(-1, -2).flip((-2, -1))')
assert true_fn != public_fn
baseline = "".join(nb["cells"][5]["source"])
check = '''    for key in ("edge_tp","edge_fp","edge_fn","division_tp","division_fp","division_fn","num_pred_nodes"):
        assert getattr(result,key) == int(expected.loc[stem,key]), (stem,key)
'''
assert baseline.count(check) == 1
summary = '''
_summaries = []
def record_config(label):
    _summaries.append(dict(config=label, **_official_summarise(official_rows)))
    for embryo in sorted({r["stem"].split("_")[0] for r in official_rows}):
        subset = [r for r in official_rows if r["stem"].startswith(embryo+"_")]
        _summaries.append(dict(config=label, embryo=embryo, **_official_summarise(subset)))
    (WORKING_DIR / "deepcenter_tta_summary.json").write_text(json.dumps(_summaries,indent=2))
record_config("baseline_single_view")
'''
nb["cells"].append(dict(cell_type="code", metadata={}, source=summary.splitlines(True)))
for label, function in [("public_eight_view", public_fn), ("true_d4_eight_view", true_fn)]:
    run = baseline.replace(check, "")
    run = run.replace('"BASELINE_VERIFIED"', f'"VARIANT_SCORED_{label}"')
    run = run.replace('"official_validator_results.csv"', f'"{label}_results.csv"')
    run = run.replace('"official_validator_summary.json"', f'"{label}_summary.json"')
    run = run.replace('f"{stem}_nodes.csv"', f'f"{label}_{{stem}}_nodes.csv"')
    run = run.replace('f"{stem}_edges.csv"', f'f"{label}_{{stem}}_edges.csv"')
    source = function + '\nos.environ["BIOHUB_DEEPCENTER_TTA"] = "1"\n' + run + f'\nrecord_config("{label}")\n'
    nb["cells"].append(dict(cell_type="code", metadata={}, source=source.splitlines(True)))
for i, cell in enumerate(nb["cells"]):
    cell.update(id=f"e0066-{i:02d}", execution_count=None, outputs=[])
    compile("".join(cell["source"]), f"E0066:{i}", "exec")
folder = root / "research_members/naveen/experiments/E0066_cached_deepcenter_tta"
folder.mkdir(exist_ok=True)
slug = "biohub-e0066-cached-deepcenter-tta"
(folder / f"{slug}.ipynb").write_text(json.dumps(nb,indent=1)+"\n")
meta = json.loads((source_dir / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}", title="Biohub E0066 Cached Deepcenter TTA", code_file=f"{slug}.ipynb")
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(folder / "provenance.json").write_text(json.dumps({
    "parent": "E0053 cached raw graphs / public 0.946",
    "source": "sjlee101/biohub-lf-dctta; repair-gate function only",
    "configs": ["baseline_single_view", "public_eight_view", "true_d4_eight_view"],
    "frozen": "Detector and edge inference; tight55; DeepCenter threshold 0.25",
    "checks": "Exact official baseline counts; fresh per-call filter caches; official aggregate and embryo scores",
    "compute": "Kaggle T4 only; diagnostic notebook; no submission"
},indent=2)+"\n")
print(folder)
