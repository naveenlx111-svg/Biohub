"""E0083 constrained residual matching; E0084 isolated 350epoch comparison."""
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ROOT / "research_members/naveen/experiments"
parent_dir = DIRS / "E0079_strong_edge_audit"
parent = json.loads((parent_dir / "biohub-e0079-strong-edge-audit.ipynb").read_text())
expanded_dir = DIRS / "E0082_expanded_strong_audit"
expanded = json.loads((expanded_dir / "biohub-e0082-expanded-strong-audit.ipynb").read_text())

def write(number, name, parts, meta, provenance):
    folder = DIRS / f"E{number:04d}_{name}"
    folder.mkdir(exist_ok=True)
    slug = f"biohub-e{number:04d}-{name.replace('_', '-')}"
    cells = []
    for i, source in enumerate(parts):
        compile(source, f"E{number}:{i}", "exec")
        cells.append(dict(cell_type="code", id=f"e{number:04d}-{i:02d}", metadata={}, execution_count=None, outputs=[], source=source.splitlines(True)))
    (folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=parent["metadata"]), indent=1) + "\n")
    meta.update(id=f"naveenlx111249971939/{slug}", title=f"Biohub E{number:04d} {name.replace('_', ' ').title()}", code_file=f"{slug}.ipynb")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(folder)

parts = ["".join(c["source"]) for c in parent["cells"][:6]]
graph_source = parts[3]
motion_node = next(n for n in ast.parse(graph_source).body if isinstance(n, ast.FunctionDef) and n.name == "motion_relink_edges")
motion = ast.get_source_segment(graph_source, motion_node)
motion = motion.replace("def motion_relink_edges(", "def motion_relink_edges_residual(", 1)
motion = motion.replace(") -> list[dict[str, object]]:", "    locked_edges=None,\n) -> list[dict[str, object]]:", 1)
motion = motion.replace("    learned_edge_probs = learned_edge_probs or {}", '''    learned_edge_probs = learned_edge_probs or {}
    locked_edges = locked_edges or []
    blocked_sources = {int(e["source_id"]) for e in locked_edges}
    blocked_targets = {int(e["target_id"]) for e in locked_edges}''', 1)
motion = motion.replace("    predecessor_position_um: dict[int, np.ndarray] = {}", '''    predecessor_position_um: dict[int, np.ndarray] = {
        int(e["target_id"]): position_um[int(e["source_id"])] for e in locked_edges
    }''', 1)
motion = motion.replace("        source_ids = ids_by_t.get(t, [])", "        source_ids = [n for n in ids_by_t.get(t, []) if n not in blocked_sources]", 1)
motion = motion.replace("        target_ids = ids_by_t.get(t + 1, [])", "        target_ids = [n for n in ids_by_t.get(t + 1, []) if n not in blocked_targets]", 1)
strong_node = next(n for n in ast.parse(parts[5]).body if isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph")
strong = ast.get_source_segment(parts[5], strong_node)
residual = strong.replace("def filter_output_graph(", "def residual_filter_output_graph(", 1)
call = "raw_motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)"
assert residual.count(call) == 1
residual = residual.replace(call, "raw_motion_edges = motion_relink_edges_residual(nodes_by_id, stats, learned_edge_probs, locked_edges=locked_edges)")
parts.append(motion + "\n\n" + residual)
parts.append('''
import copy
import torch
torch.set_num_threads(4)
assert torch.cuda.is_available()
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = TRAIN_DIR
MOTION_RELINK_TIGHT_UM = 5.5
os.environ["BIOHUB_STRONG_EDGE_THRESHOLD"] = "0.55"
old_selection = list(Path("/kaggle/input").rglob("ppsweep_selected.json"))
extra_selection = list(Path("/kaggle/input").rglob("expanded_panel.json"))
assert len(old_selection) == len(extra_selection) == 1
old_stems = json.loads(old_selection[0].read_text())["held_out_stems"]
extra_stems = json.loads(extra_selection[0].read_text())["selected"]
assert len(old_stems) == len(extra_stems) == 8 and not set(old_stems) & set(extra_stems)
raw_roots = {"original": old_selection[0].parent / "tracking_repo/predictions/unknown/unet_transformer_val/split_0"}
extra_roots = list((extra_selection[0].parent / "tracking_repo/predictions").glob("*/unet_transformer/split_0"))
assert len(extra_roots) == 1
raw_roots["additional"] = extra_roots[0]
reference_files = list(Path("/kaggle/input").rglob("strong_edge_samples.csv"))
assert len(reference_files) == 2, reference_files
references = pd.concat([pd.read_csv(p) for p in reference_files], ignore_index=True)
configs = {"anchor0947": anchor_filter_output_graph, "preserve_strong055": strong_filter_output_graph, "residual_strong055": residual_filter_output_graph}
rows, stage_rows = [], []
def persist():
    pd.DataFrame(rows).to_csv(WORKING_DIR / "residual_samples.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(WORKING_DIR / "residual_stages.csv", index=False)
    summaries = []
    for config in configs:
        for panel in ("original", "additional", "all16"):
            selected = [r for r in rows if r["config"] == config and (panel == "all16" or r["panel"] == panel)]
            if not selected:
                continue
            summaries.append(dict(config=config, panel=panel, **_official_summarise(selected)))
            for embryo in ("44b6", "6bba"):
                subset = [r for r in selected if r["stem"].startswith(embryo + "_")]
                if subset:
                    summaries.append(dict(config=config, panel=panel, embryo=embryo, **_official_summarise(subset)))
    (WORKING_DIR / "residual_summary.json").write_text(json.dumps(summaries, indent=2))
for panel, stems in (("original", old_stems), ("additional", extra_stems)):
    for stem in stems:
        graph = graph_from_geff(raw_roots[panel] / f"{stem}.geff")
        raw_nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]), "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])} for r in graph.node_attrs().iter_rows(named=True)}
        raw_edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]), edge_prob=r.get("edge_prob")) for r in graph.edge_attrs().iter_rows(named=True)]
        gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
        metadata = _GeffMetadata.read(TRAIN_DIR / f"{stem}.geff")
        for config, processor in configs.items():
            nodes, edges, stats = processor(copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
            pred = _official_graph_from_processed(nodes, edges)
            result = _official_evaluate(pred, gt, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
            if config != "residual_strong055":
                expected = references[(references.stem == stem) & (references.config == config)]
                assert len(expected) == 1
                for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes"):
                    assert getattr(result, key) == int(expected.iloc[0][key]), (stem, config, key)
            row = _official_per_sample_metrics(result, float((metadata.extra or {})["estimated_number_of_nodes"]), _official_node_recall(pred, gt))
            row.update(stem=stem, config=config, panel=panel)
            rows.append(row)
            stage_rows.append(dict(stem=stem, config=config, panel=panel, **stats))
            persist()
            print("RESIDUAL_RESULT", row, flush=True)
assert len(rows) == 48
print("E0083_COMPLETE", flush=True)
''')
meta = json.loads((parent_dir / "kernel-metadata.json").read_text())
meta["kernel_sources"] = ["naveenlx111249971939/biohub-lf-dctta-v020", "naveenlx111249971939/biohub-e0079-strong-edge-audit", "naveenlx111249971939/biohub-e0082-expanded-strong-audit"]
write(83, "residual_matching", parts, meta, dict(
    hypothesis="Run Hungarian motion matching only on endpoints not occupied by locked learned edges; use locked predecessor positions for velocity",
    frozen="400epoch anchor weights/detections; strong055; tight55; other postprocessing",
    evaluation="Exact original and strong055 metric replay on both existing panels; 16 movies; official per-panel and embryo summaries",
    compute="Kaggle T4 only; no submission from diagnostic notebook"))

parts = ["".join(c["source"]) for c in expanded["cells"]]
parts[0] += '\nos.environ["BIOHUB_PRIMARY_WEIGHT_OVERRIDE"] = "350ep"\nos.environ["BIOHUB_PRIMARY_WEIGHT_OVERRIDE_SHA256"] = "dfb848aa8e490bba8eda91ac927b9ad1d8b06296487ba8504e45a1037c5e36ec"\n'
source350 = json.loads((ROOT / "local_runs/frontier_20260912/lineage350/biohub-cell-tracking-lineage-submission.ipynb").read_text())
parts[3] = "".join(source350["cells"][3]["source"])
needle = '    (WORKING_DIR / "expanded_panel.json").write_text'
assert parts[4].count(needle) == 1
parts[4] = parts[4].replace(needle, '    selected = sorted(set(selected) | old_panel)\n    assert len(selected) == 16\n' + needle)
parts[4] = parts[4].replace('"excluded_original": sorted(old_panel)', '"original_panel": sorted(old_panel)')
evaluation = parts[-1]
evaluation = evaluation.replace('configs = {"anchor0947": None, "preserve_strong055": .55}', 'configs = {"checkpoint350_strong055": .55}')
evaluation = evaluation.replace('"E0082_COMPLETE"', '"E0084_COMPLETE"')
parts[-1] = evaluation
parts.append('''
# Compare new checkpoint with the existing strong055 reference on identical movies.
references = list(Path("/kaggle/input").rglob("strong_edge_samples.csv"))
assert len(references) == 2, references
old_panel = set(json.loads((WORKING_DIR / "expanded_panel.json").read_text())["original_panel"])
reference = pd.concat([pd.read_csv(p) for p in references], ignore_index=True)
reference = reference[reference.config.eq("preserve_strong055")].copy()
reference["config"] = "checkpoint400_strong055"
new_rows = pd.DataFrame(rows)
assert set(reference.stem) == set(new_rows.stem) and len(reference) == len(new_rows) == 16
comparison = pd.concat([reference, new_rows], ignore_index=True)
comparison["panel"] = comparison.stem.map(lambda s: "original" if s in old_panel else "additional")
comparison.to_csv(WORKING_DIR / "checkpoint_comparison_samples.csv", index=False)
summary = []
for config, group in comparison.groupby("config"):
    for panel in ("original", "additional", "all16"):
        selected = group if panel == "all16" else group[group.panel.eq(panel)]
        summary.append(dict(config=config, panel=panel, **_official_summarise(selected.to_dict("records"))))
        for embryo in ("44b6", "6bba"):
            subset = selected[selected.stem.str.startswith(embryo + "_")]
            summary.append(dict(config=config, panel=panel, embryo=embryo, **_official_summarise(subset.to_dict("records"))))
(WORKING_DIR / "checkpoint_comparison_summary.json").write_text(json.dumps(summary, indent=2))
print("E0084_COMPARISON_COMPLETE", flush=True)
''')
meta = json.loads((expanded_dir / "kernel-metadata.json").read_text())
meta["kernel_sources"] = ["naveenlx111249971939/biohub-e0079-strong-edge-audit", "naveenlx111249971939/biohub-e0082-expanded-strong-audit"]
meta["dataset_sources"].append("shehailrs/biohub-tracking-350ep-public-weight-snapshot")
write(84, "checkpoint350_comparison", parts, meta, dict(
    hypothesis="Compare older350epoch checkpoint against400epoch under identical strong055 processing",
    frozen="Secondary weights and blend, TTA, thresholds and strong055 postprocessing; no residual-matching change",
    weights_sha256="dfb848aa8e490bba8eda91ac927b9ad1d8b06296487ba8504e45a1037c5e36ec",
    evaluation="Same16movies; compare checkpoint400 cached official results; panel and embryo reporting; training overlap unresolved",
    compute="Kaggle T4 only; no submission from diagnostic notebook"))
