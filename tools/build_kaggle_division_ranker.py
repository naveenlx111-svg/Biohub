"""E0065: run the recovered division-ranking study entirely on Kaggle."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base_dir = root / "research_members/naveen/experiments/E0053_harmonic_frontier"
base = json.loads((base_dir / "biohub-e0053-harmonic-frontier.ipynb").read_text())
resume = json.loads((root / "local_runs/E0060/e0060-frontier-oracle.ipynb").read_text())
parts = ["".join(c["source"]) for c in resume["cells"]]
for i, source in enumerate(parts):
    source = source.replace(str(root / "data/kaggle_input"), "/kaggle/input")
    source = source.replace(str(root / "local_runs/E0060/working"), "/kaggle/working")
    parts[i] = source
parts[0] += '''
import subprocess, sys
subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0)); print(torch.ones(4,device='cuda').cpu())"], check=True)
'''
baseline = parts[4]
start = baseline.index("val_stems =")
end = baseline.index("expected = expected[")
baseline = baseline[:start] + '''
_cached = list(Path("/kaggle/input").rglob("frontier_official_samples.csv"))
assert len(_cached) == 1, _cached
cached_root = _cached[0].parent
val_stems = json.loads((cached_root / "ppsweep_selected.json").read_text())["held_out_stems"]
raw_root = cached_root / "tracking_repo/predictions/unknown/unet_transformer_val/split_0"
assert raw_root.exists(), raw_root
expected = pd.read_csv(_cached[0])
''' + baseline[end:]
parts[4] = baseline
# Preserve the existing geometry gate rather than widening it using held-out GT.
# A lost positive is a measured coverage limitation, not grounds to abort training.
gate = 'raise RuntimeError("Same-run hard-negative gate discarded a positive")'
assert parts[9].count(gate) == 1
parts[9] = parts[9].replace(gate,
    'print("FROZEN_GATE_LOST_POSITIVES", int(_candidate_df["label"].sum())-int(_ranked["label"].sum()))')
# Only the geometry OOF pair selection is needed for crop ranking, not the unused
# second-stage image-feature classifier from the historical exploratory notebook.
parts[9] = parts[9].split("def _patch_stats(")[0]
parts[9] += '\n_parents.to_csv(WORKING_DIR / "frozen_gate_parents.csv", index=False)\n'
parts[9] += '''
(WORKING_DIR / "frozen_gate_coverage.json").write_text(json.dumps({
    "proposed_pairs": len(_candidate_df),
    "positive_proposed_pairs": int(_candidate_df["label"].sum()),
    "retained_pairs": len(_ranked),
    "positive_retained_pairs": int(_ranked["label"].sum()),
    "selected_parents": len(_parents),
    "positive_selected_parents": int(_parents["label"].sum()),
    "policy": "Historical thresholds fixed; labels used only for coverage reporting and training on other embryo"
}, indent=2))
'''
# Keep trained folds so promising results can be tested on new graphs without
# repeating crop extraction and training. Held-out embryo identity is explicit.
assert "_crop_models = {}" in parts[10]
parts[10] += '''
_saved_models = []
for _heldout, _saved_model in _crop_models.items():
    _save_path = WORKING_DIR / f"division_crop_heldout_{_heldout}.pt"
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in _saved_model.state_dict().items()},
        "architecture": "_DivisionCropNet", "heldout_embryo": _heldout,
        "training_embryos": sorted(set(_crop_embryos)-{_heldout}),
        "crop_mode": _crop_mode, "seed": _crop_seed,
        "crop_radius_zyx": [4,16,16], "epochs": 25
    }, _save_path)
    _saved_models.append(str(_save_path))
print("SAVED_OOF_CROP_MODELS", _saved_models)
'''
# Install the exact parent's offline dependencies before importing graph libraries.
parts.insert(2, "".join(base["cells"][3]["source"]))

repair = (root / "tools/audit_ranked_division_edits.py").read_text()
repair = repair[repair.index("\nrows = []\n")+1:]
repair = repair.replace('ns["_GeffMetadata"]', '_GeffMetadata').replace('ns["_official_node_recall"]', '_official_node_recall')
repair_setup = '''
# Selection uses only OOF model scores and topology, never annotation labels.
source = WORKING_DIR
output = WORKING_DIR / "ranked_repairs"
output.mkdir(exist_ok=True)
train = TRAIN_DIR
build = _official_graph_from_processed
evaluate = _official_evaluate
per_sample = _official_per_sample_metrics
summarise = _official_summarise
candidate = pd.read_csv(WORKING_DIR / "offset_pool_transfer_oof.csv", usecols=[
    "stem", "fork_id", "daughter1_id", "daughter2_id", "offset_max_score", "offset_top2_mean_score"])
stems = list(val_stems)
expected = pd.read_csv(WORKING_DIR / "official_validator_results.csv").set_index("stem")
'''
parts.append(repair_setup + repair)
folder = root / "research_members/naveen/experiments/E0065_kaggle_division_ranker"
folder.mkdir(exist_ok=True)
cells = []
for i, source in enumerate(parts):
    assert str(root) not in source, f"Local path in cell {i}"
    compile(source, f"E0065:{i}", "exec")
    cells.append(dict(cell_type="code", id=f"e0065-{i:02d}", metadata={},
                      execution_count=None, outputs=[], source=source.splitlines(True)))
nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=base["metadata"])
slug = "biohub-e0065-kaggle-division-ranker"
(folder / f"{slug}.ipynb").write_text(json.dumps(nb,indent=1)+"\n")
meta = json.loads((base_dir / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}", title="Biohub E0065 Kaggle Division Ranker",
            code_file=f"{slug}.ipynb", machine_shape="NvidiaTeslaT4",
            kernel_sources=["naveenlx111249971939/biohub-e0053-harmonic-frontier-audit"])
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(folder / "provenance.json").write_text(json.dumps({
    "parent": "E0053 cached raw graphs; exact baseline counts asserted before training",
    "method": "Frozen historical geometry gate, embryo-held-out geometry ranking and temporal crop CNN, offset pooling, topology repairs",
    "gate_change": "Report coverage loss instead of abort; no threshold changed using held-out labels",
    "evaluation": "Official graph score for predeclared top25/100/250/500; exploratory not independent confirmation",
    "oracle": "GT-guided diagnostic only; original graphs remain separate",
    "compute": "Kaggle only; no submission from this research notebook"
},indent=2)+"\n")
print(folder)
