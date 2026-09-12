"""E0079: isolate learned-edge preservation, without changing model weights."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ROOT / "research_members/naveen/experiments"
parent = DIRS / "E0076_hoct_consensus_audit"
nb = json.loads((parent / "biohub-e0076-hoct-consensus-audit.ipynb").read_text())
parts = ["".join(c["source"]) for c in nb["cells"][:5]]
public = json.loads((ROOT / "local_runs/frontier_20260912/lineage350/biohub-cell-tracking-lineage-submission.ipynb").read_text())
public_source = "".join(public["cells"][5]["source"])
function = next(n for n in ast.parse(public_source).body if isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph")
parts.append("anchor_filter_output_graph = filter_output_graph\n" + ast.get_source_segment(public_source, function) + "\nstrong_filter_output_graph = filter_output_graph\n")
parts.append('''
import copy
import torch
torch.set_num_threads(4)
assert torch.cuda.is_available()
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = TRAIN_DIR
reference_files = list(Path("/kaggle/input").rglob("hoct_official_samples.csv"))
selection_files = list(Path("/kaggle/input").rglob("ppsweep_selected.json"))
assert len(reference_files) == len(selection_files) == 1
expected = pd.read_csv(reference_files[0])
expected = expected[expected.config.eq("anchor0947")].set_index("stem")
selection = json.loads(selection_files[0].read_text())
stems = selection["held_out_stems"]
raw_root = selection_files[0].parent / "tracking_repo/predictions/unknown/unet_transformer_val/split_0"
for key, value in selection["overrides"].items():
    assert key in globals(), key
    globals()[key] = value
configs = {"anchor0947": None, "preserve_strong055": .55, "preserve_strong080": .80, "preserve_strong095": .95}
rows = []
stage_rows = []
def persist():
    pd.DataFrame(rows).to_csv(WORKING_DIR / "strong_edge_samples.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(WORKING_DIR / "strong_edge_stages.csv", index=False)
    summary = []
    for config in configs:
        subset = [r for r in rows if r["config"] == config]
        if not subset:
            continue
        summary.append(dict(config=config, **_official_summarise(subset)))
        for embryo in sorted({r["stem"].split("_")[0] for r in subset}):
            summary.append(dict(config=config, embryo=embryo, **_official_summarise([r for r in subset if r["stem"].startswith(embryo + "_")])))
    (WORKING_DIR / "strong_edge_summary.json").write_text(json.dumps(summary, indent=2))
for stem in stems:
    raw = graph_from_geff(raw_root / f"{stem}.geff")
    raw_nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]), "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])} for r in raw.node_attrs().iter_rows(named=True)}
    raw_edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]), edge_prob=r.get("edge_prob")) for r in raw.edge_attrs().iter_rows(named=True)]
    gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
    metadata = _GeffMetadata.read(TRAIN_DIR / f"{stem}.geff")
    for config, threshold in configs.items():
        if threshold is None:
            processor = anchor_filter_output_graph
        else:
            processor = strong_filter_output_graph
            os.environ["BIOHUB_STRONG_EDGE_THRESHOLD"] = str(threshold)
        nodes, edges, stats = processor(copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        pred = _official_graph_from_processed(nodes, edges)
        result = _official_evaluate(pred, gt, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
        if config == "anchor0947":
            for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes"):
                assert getattr(result, key) == int(expected.loc[stem, key]), (stem, key)
        row = _official_per_sample_metrics(result, float((metadata.extra or {})["estimated_number_of_nodes"]), _official_node_recall(pred, gt))
        row.update(stem=stem, config=config)
        rows.append(row)
        stage_rows.append(dict(stem=stem, config=config, **stats))
        pd.DataFrame(nodes.values()).to_csv(WORKING_DIR / f"{stem}_{config}_nodes.csv", index=False)
        pd.DataFrame(edges).to_csv(WORKING_DIR / f"{stem}_{config}_edges.csv", index=False)
        persist()
        print("STRONG_EDGE_RESULT", row, flush=True)
assert len(rows) == len(stems) * len(configs)
print("E0079_COMPLETE", flush=True)
''')
cells = []
for i, source in enumerate(parts):
    compile(source, f"E0079:{i}", "exec")
    cells.append(dict(cell_type="code", id=f"e0079-{i:02d}", metadata={}, execution_count=None, outputs=[], source=source.splitlines(True)))
nb["cells"] = cells
folder = DIRS / "E0079_strong_edge_audit"
folder.mkdir(exist_ok=True)
slug = "biohub-e0079-strong-edge-audit"
(folder / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
meta = json.loads((parent / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}", title="Biohub E0079 Strong Edge Audit", code_file=f"{slug}.ipynb")
meta["kernel_sources"] += ["naveenlx111249971939/biohub-e0076-hoct-consensus-audit"]
meta["dataset_sources"] = [s for s in meta["dataset_sources"] if "hoct" not in s]
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
(folder / "provenance.json").write_text(json.dumps(dict(
    source="tharunkumar369/biohub-cell-tracking-lineage-submission; filter_output_graph only",
    isolated_change="Preserve strong learned edges before geometric relinking; threshold .55 from public source plus predeclared .80/.95 sensitivity checks",
    frozen="Exact anchor checkpoint/detections/raw graph/tight55/DeepCenter; NO 350epoch checkpoint change",
    motivation="Current motion relinking replaces ILP edges; changing ILP division cost alone did not recover divisions",
    caveat="Reused diagnostic; no claim of public 0.95; graph/node postprocessing can change as a downstream consequence",
    compute="Kaggle T4 cached graph audit; exact official anchor replay required"), indent=2) + "\n")
print(folder)
