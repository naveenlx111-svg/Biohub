"""E0085: quantify production CSV export effects on both diagnostic panels."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ROOT / "research_members/naveen/experiments"
parent_dir = DIRS / "E0078_hoct_fork_fusion"
parent = json.loads((parent_dir / "biohub-e0078-hoct-fork-fusion.ipynb").read_text())
parts = ["".join(c["source"]) for c in parent["cells"][:3]]
source = (ROOT / "local_runs/metric_review_20260913/scripts/csv_to_geffs.py").read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_graph_from_rows")
parts.append(ast.get_source_segment(source, node))
parts.append('''
import io
import hashlib
import torch
torch.set_num_threads(2)
TRAIN_DIR = COMP_DIR / "train"
references = list(Path("/kaggle/input").rglob("strong_edge_samples.csv"))
assert len(references) == 2, references
rows, changes = [], []
def graph_from_geff(path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph
def persist():
    pd.DataFrame(rows).to_csv(WORKING_DIR / "export_parity_samples.csv", index=False)
    pd.DataFrame(changes).to_csv(WORKING_DIR / "export_coordinate_changes.csv", index=False)
    summaries = []
    for config in ("anchor0947", "preserve_strong055"):
        for representation in ("float_graph", "production_csv"):
            for panel in ("original", "additional", "all16"):
                selected = [r for r in rows if r["config"] == config and r["representation"] == representation and (panel == "all16" or r["panel"] == panel)]
                if selected:
                    summaries.append(dict(config=config, representation=representation, panel=panel, **_official_summarise(selected)))
    (WORKING_DIR / "export_parity_summary.json").write_text(json.dumps(summaries, indent=2))
for reference_file in references:
    expected = pd.read_csv(reference_file)
    panel = "additional" if (reference_file.parent / "expanded_panel.json").exists() else "original"
    stems = sorted(expected.stem.unique())
    assert len(stems) == 8
    for stem in stems:
        gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
        meta = _GeffMetadata.read(TRAIN_DIR / f"{stem}.geff")
        for config in ("anchor0947", "preserve_strong055"):
            node_frame = pd.read_csv(reference_file.parent / f"{stem}_{config}_nodes.csv", float_precision="round_trip")
            edges = pd.read_csv(reference_file.parent / f"{stem}_{config}_edges.csv", float_precision="round_trip").to_dict("records")
            nodes = {int(r["node_id"]): r for r in node_frame.to_dict("records")}
            # Same ordering, rounding and lower-bound clamp as production writer.
            buffer = io.StringIO()
            columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
            writer = csv.DictWriter(buffer, fieldnames=columns)
            writer.writeheader()
            for row_id, node_id in enumerate(sorted(nodes)):
                n = nodes[node_id]
                writer.writerow(dict(id=row_id, dataset=stem, row_type="node", node_id=node_id, t=int(n["t"]), z=max(0,int(round(float(n["z"])))), y=max(0,int(round(float(n["y"])))), x=max(0,int(round(float(n["x"])))), source_id=-1, target_id=-1))
            for row_id, e in enumerate(edges, start=len(nodes)):
                writer.writerow(dict(id=row_id, dataset=stem, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=int(e["source_id"]), target_id=int(e["target_id"])))
            csv_text = buffer.getvalue()
            frame = pl.read_csv(io.StringIO(csv_text))
            parsed_nodes = frame.filter(pl.col("row_type") == "node")
            parsed_edges = frame.filter(pl.col("row_type") == "edge")
            float_positions = np.array([[nodes[i][k] for k in ("z","y","x")] for i in sorted(nodes)])
            integer_positions = parsed_nodes.select("z","y","x").to_numpy()
            distances = np.linalg.norm((float_positions - integer_positions) * np.array(VOXEL_SCALE_UM), axis=1)
            changes.append(dict(stem=stem, config=config, panel=panel, nodes=len(nodes), edges=len(edges), changed_nodes=int((distances>0).sum()), max_displacement_um=float(distances.max()), mean_displacement_um=float(distances.mean()), csv_sha256=hashlib.sha256(csv_text.encode()).hexdigest()))
            for representation in ("float_graph", "production_csv"):
                pred = _official_graph_from_processed(nodes, edges) if representation == "float_graph" else build_graph_from_rows(parsed_nodes, parsed_edges)
                result = _official_evaluate(pred, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
                if representation == "float_graph":
                    ref = expected[(expected.stem==stem)&(expected.config==config)].iloc[0]
                    for key in ("edge_tp","edge_fp","edge_fn","division_tp","division_fp","division_fn","num_pred_nodes"):
                        assert getattr(result,key)==int(ref[key]),(stem,config,key)
                row = _official_per_sample_metrics(result,float((meta.extra or {})["estimated_number_of_nodes"]),_official_node_recall(pred,gt))
                row.update(stem=stem,config=config,panel=panel,representation=representation)
                rows.append(row)
                persist()
                print("EXPORT_PARITY",row,flush=True)
assert len(rows)==64
print("E0085_COMPLETE",flush=True)
''')
folder = DIRS / "E0085_export_parity_audit"
folder.mkdir(exist_ok=True)
slug = "biohub-e0085-export-parity-audit"
cells = []
for i, source in enumerate(parts):
    compile(source,f"E0085:{i}","exec")
    cells.append(dict(cell_type="code",id=f"e0085-{i:02d}",metadata={},execution_count=None,outputs=[],source=source.splitlines(True)))
(folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=parent["metadata"],cells=cells),indent=1)+"\n")
meta=json.loads((parent_dir / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}",title="Biohub E0085 Export Parity Audit",code_file=f"{slug}.ipynb",enable_gpu=False,kernel_sources=["naveenlx111249971939/biohub-e0079-strong-edge-audit","naveenlx111249971939/biohub-e0082-expanded-strong-audit"])
meta["dataset_sources"]=[s for s in meta["dataset_sources"] if "hoct" not in s]
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(folder / "provenance.json").write_text(json.dumps(dict(purpose="Investigate float-validation versus integer-production CSV parity",panels=16,policies=["anchor0947","preserve_strong055"],scorer_commit="075fc5f5a52d11077f9dc2b074644618f26939e2",csv_reader="Official scripts/csv_to_geffs.py build_graph_from_rows",controls="Exact original float metrics replay before interpreting CSV effect; frozen topology",compute="Kaggle CPU only; no submission or production changes"),indent=2)+"\n")
print(folder)
