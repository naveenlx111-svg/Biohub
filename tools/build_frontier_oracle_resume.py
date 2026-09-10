"""Rebase the existing oracle and label-free division research on E0053 graphs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
frontier = root / "local_runs/frontier_20260909/harmonic"
old = json.loads((root / "local_runs/E0047/e0047-offset-oracle.ipynb").read_text())
base = json.loads((root / "research_members/naveen/experiments/E0053_harmonic_frontier/biohub-e0053-harmonic-frontier.ipynb").read_text())
dest = root / "local_runs/E0060"
working = dest / "working"
working.mkdir(parents=True, exist_ok=True)
parts = []
parts.append("".join(base["cells"][0]["source"]) + '\nos.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n')
parts.append("".join(base["cells"][2]["source"]))
graph = "".join(base["cells"][5]["source"])
parts.append(graph[:graph.index("def write_test_submission(")])
parts.append("".join(base["cells"][-1]["source"]).split("# Audit frozen base")[0])
parts.append(f'''
import copy
torch.set_num_threads(4)
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = TRAIN_DIR
VALIDATOR_MATCH_RADIUS_UM = 7.0
VALIDATOR_ENABLE = True
val_stems = json.loads(Path({str(frontier / 'output/ppsweep_selected.json')!r}).read_text())["held_out_stems"]
raw_root = Path({str(frontier / 'output/tracking_repo/predictions/unknown/unet_transformer_val/split_0')!r})
expected = pd.read_csv(Path({str(root / 'local_runs/E0053/kaggle/frontier_official_samples.csv')!r}))
expected = expected[expected.config.eq("source_selected")].set_index("stem")
official_graph_inputs = {{}}
official_stage_stats = {{}}
official_rows = []
validator_sample_rows = []
for stem in val_stems:
    raw = graph_from_geff(raw_root / f"{{stem}}.geff")
    nodes = {{int(r["node_id"]): {{"node_id": int(r["node_id"]), "t": int(r["t"]),
               "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}}
               for r in raw.node_attrs().iter_rows(named=True)}}
    edges = [dict(source_id=int(r["source_id"]),target_id=int(r["target_id"]),edge_prob=r.get("edge_prob"))
             for r in raw.edge_attrs().iter_rows(named=True)]
    nodes, edges, stats = filter_output_graph(nodes,edges,dataset=stem,deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
    official_graph_inputs[stem] = (nodes,edges)
    official_stage_stats[stem] = stats
    pred = _official_graph_from_processed(nodes,edges)
    gt = graph_from_geff(TRAIN_DIR / f"{{stem}}.geff")
    result = _official_evaluate(pred,gt,scale=tuple(VOXEL_SCALE_UM),max_distance=7.0)
    for key in ("edge_tp","edge_fp","edge_fn","division_tp","division_fp","division_fn","num_pred_nodes"):
        assert getattr(result,key) == int(expected.loc[stem,key]), (stem,key)
    meta = _GeffMetadata.read(TRAIN_DIR / f"{{stem}}.geff")
    row = _official_per_sample_metrics(result,float((meta.extra or {{}})["estimated_number_of_nodes"]),_official_node_recall(pred,gt))
    row["stem"] = stem
    official_rows.append(row)
    pred.node_attrs().to_pandas().to_csv(WORKING_DIR / f"{{stem}}_nodes.csv",index=False)
    pred.edge_attrs().to_pandas().to_csv(WORKING_DIR / f"{{stem}}_edges.csv",index=False)
    print("BASELINE_VERIFIED",stem,flush=True)
pd.DataFrame(official_rows).to_csv(WORKING_DIR / "official_validator_results.csv",index=False)
(WORKING_DIR / "official_validator_summary.json").write_text(json.dumps(_official_summarise(official_rows),indent=2))
''')
# Reuse the established anatomy, localization, oracle, candidate ranking,
# embryo-held-out crop training and offset inference implementations.
for index in (6,7,8,10,11,12,13,14):
    parts.append("".join(old["cells"][index]["source"]))
cells = []
for i, text in enumerate(parts):
    text = text.replace("/kaggle/input",str(root / "data/kaggle_input"))
    text = text.replace("/kaggle/working",str(working))
    # Old E0047 sources were localized to EXT0006; redirect any output paths.
    text = text.replace(str(root / "local_runs/EXT0006/working"),str(working))
    compile(text,f"E0060:{i}","exec")
    cells.append(dict(cell_type="code",id=f"e0060-{i:02d}",metadata={},execution_count=None,
                      outputs=[],source=text.splitlines(True)))
nb = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
    "kernelspec":{"display_name":"Python 3","name":"python3","language":"python"}})
(dest / "e0060-frontier-oracle.ipynb").write_text(json.dumps(nb,indent=1)+"\n")
print(dest / "e0060-frontier-oracle.ipynb")
