"""E0076: independently score HOCT consensus on the exact 0.947 anchor."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
anchor_dir = root / "local_runs/anchor0947"
hoct_dir = root / "local_runs/frontier_20260911/hoct"
anchor = json.loads((anchor_dir / "biohub-lf-dctta-v020.ipynb").read_text())
hoct = json.loads((hoct_dir / "biohub-lf-hoctveto-div.ipynb").read_text())
prior = json.loads((root / "research_members/naveen/experiments/E0065_kaggle_division_ranker/biohub-e0065-kaggle-division-ranker.ipynb").read_text())
parts = ["".join(anchor["cells"][i]["source"]) for i in (0, 2, 3)]
parts.append("".join(anchor["cells"][5]["source"]).split("def write_test_submission(")[0])
parts.append("".join(prior["cells"][4]["source"]))
parts.append("".join(hoct["cells"][6]["source"]).split("# ---------------------------------------------------------------- hook into write_test_submission")[0])
parts.append('''
import copy
from collections import Counter
import torch
torch.set_num_threads(4)
assert torch.cuda.is_available()
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = TRAIN_DIR
VALIDATOR_MATCH_RADIUS_UM = 7.0
cached = list(Path("/kaggle/input").rglob("ppsweep_selected.json"))
assert len(cached) == 1, cached
cached_root = cached[0].parent
selection = json.loads(cached[0].read_text())
val_stems = selection["held_out_stems"]
assert len(val_stems) >= 8, val_stems
raw_root = cached_root / "tracking_repo/predictions/unknown/unet_transformer_val/split_0"
assert raw_root.exists(), raw_root
for key, value in selection["overrides"].items():
    assert key in globals(), key
    globals()[key] = value
print("ANCHOR_SELECTION", selection, flush=True)
(WORKING_DIR / "anchor_selection.json").write_text(json.dumps(selection, indent=2))
rows = []
diagnostics = []
configs = ("anchor0947", "veto_ordinary_only", "veto_all", "veto_division_only")

def persist():
    pd.DataFrame(rows).to_csv(WORKING_DIR / "hoct_official_samples.csv", index=False)
    summaries = []
    for config in configs:
        subset = [r for r in rows if r["config"] == config]
        if not subset:
            continue
        summaries.append(dict(config=config, **_official_summarise(subset)))
        for embryo in sorted({r["stem"].split("_")[0] for r in subset}):
            embryo_rows = [r for r in subset if r["stem"].startswith(embryo + "_")]
            summaries.append(dict(config=config, embryo=embryo, **_official_summarise(embryo_rows)))
    (WORKING_DIR / "hoct_official_summary.json").write_text(json.dumps(summaries, indent=2))
    (WORKING_DIR / "hoct_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

def score(stem, nodes, edges, config, gt, meta):
    pred = _official_graph_from_processed(nodes, edges)
    result = _official_evaluate(pred, gt, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
    row = _official_per_sample_metrics(result, float((meta.extra or {})["estimated_number_of_nodes"]), _official_node_recall(pred, gt))
    row.update(stem=stem, config=config)
    rows.append(row)
    persist()
    print("OFFICIAL_RESULT", row, flush=True)

for stem in val_stems:
    raw = graph_from_geff(raw_root / f"{stem}.geff")
    nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
             "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
             for r in raw.node_attrs().iter_rows(named=True)}
    edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]), edge_prob=r.get("edge_prob"))
             for r in raw.edge_attrs().iter_rows(named=True)]
    nodes, edges, stats = filter_output_graph(nodes, edges, dataset=stem, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
    pd.DataFrame(nodes.values()).to_csv(WORKING_DIR / f"{stem}_anchor_nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(WORKING_DIR / f"{stem}_anchor_edges.csv", index=False)
    gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
    meta = _GeffMetadata.read(TRAIN_DIR / f"{stem}.geff")
    score(stem, nodes, edges, "anchor0947", gt, meta)
    pairs = _hv_hoct_pairs(stem, nodes)
    pd.DataFrame(sorted(pairs), columns=["source_id", "target_id"]).to_csv(WORKING_DIR / f"{stem}_hoct_pairs.csv", index=False)
    degree = Counter(int(e["source_id"]) for e in edges)
    for config in configs[1:]:
        if config == "veto_division_only":
            kept = [e for e in edges if degree[int(e["source_id"])] < 2 or (int(e["source_id"]), int(e["target_id"])) in pairs]
        else:
            kept, _ = _hv_apply_veto(edges, pairs, 1 if config == "veto_ordinary_only" else 2)
        assert all(int(e["target_id"]) in nodes and int(e["source_id"]) in nodes for e in kept)
        diagnostics.append(dict(stem=stem, config=config, nodes=len(nodes), edges_before=len(edges), edges_after=len(kept), hoct_pairs=len(pairs)))
        score(stem, nodes, kept, config, gt, meta)
    _HV_STATE["cache"].clear()
    torch.cuda.empty_cache()
assert len(rows) == len(val_stems) * len(configs)
persist()
print("E0076_COMPLETE", flush=True)
''')
cells = []
for i, source in enumerate(parts):
    compile(source, f"E0076:{i}", "exec")
    cells.append(dict(cell_type="code", id=f"e0076-{i:02d}", metadata={}, execution_count=None, outputs=[], source=source.splitlines(True)))
folder = root / "research_members/naveen/experiments/E0076_hoct_consensus_audit"
folder.mkdir(exist_ok=True)
slug = "biohub-e0076-hoct-consensus-audit"
(folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=anchor["metadata"], cells=cells), indent=1) + "\n")
meta = json.loads((hoct_dir / "kernel-metadata.json").read_text())
for key in ("id_no", "docker_image"):
    meta.pop(key, None)
meta.update(id=f"naveenlx111249971939/{slug}", title="Biohub E0076 HOCT Consensus Audit", code_file=f"{slug}.ipynb",
            kernel_sources=["naveenlx111249971939/biohub-lf-dctta-v020"], machine_shape="NvidiaTeslaT4", is_private=True)
meta["dataset_sources"] = sorted(set(meta["dataset_sources"] + ["dalloliogm/biohub-official-scorer-patched"]))
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
(folder / "provenance.json").write_text(json.dumps(dict(
    anchor="naveenlx111249971939/biohub-lf-dctta-v020 version 1; submission 56159060 public 0.947",
    mechanism="HOCT pretrained temporal edge consensus on frozen anchor nodes",
    source="sjlee101/biohub-lf-hoctveto-div",
    configs=["anchor0947", "veto_ordinary_only", "veto_all", "veto_division_only"],
    evaluation="Patched official scorer, same cached anchor held-out graphs and selected postprocessing; exploratory repeated holdout, not LB",
    caveat="Public HOCT validation bypasses its write_test_submission-only hook; this audit explicitly scores vetoed graphs",
    submission=False), indent=2) + "\n")
print(folder)
