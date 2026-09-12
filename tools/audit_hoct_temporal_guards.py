"""E0080: local CPU audit of pre-existing temporal guards on HOCT forks."""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import tracksdata as td
from hoct_graph_fusion import fuse_hoct_forks

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "local_runs/E0076/kaggle"
output = ROOT / "local_runs/E0080"
output.mkdir(exist_ok=True)
train = ROOT / "data/kaggle_input/competitions/biohub-cell-tracking-during-development/train"
VOXEL_SCALE_UM = (1.625, .40625, .40625)
nb = json.loads((ROOT / "research_members/naveen/experiments/E0065_kaggle_division_ranker/biohub-e0065-kaggle-division-ranker.ipynb").read_text())
setup = "".join(nb["cells"][4]["source"]).replace("/kaggle/input", str(ROOT / "data/kaggle_input"))
exec(compile(setup, "verified_official_setup", "exec"))
reference = pd.read_csv(ROOT / "local_runs/E0078/kaggle/fusion_official_samples.csv")
configs = ("ordinary_veto", "unguarded_forks", "persistence3", "persistence3_nonconverging", "persistence3_diverge2um")
rows, features = [], []

def persist():
    pd.DataFrame(rows).to_csv(output / "samples.csv", index=False)
    pd.DataFrame(features).to_csv(output / "fork_features.csv", index=False)
    summaries = []
    for config in configs:
        subset = [r for r in rows if r["config"] == config]
        if not subset:
            continue
        summaries.append(dict(config=config, **_official_summarise(subset)))
        for embryo in sorted({r["stem"].split("_")[0] for r in subset}):
            summaries.append(dict(config=config, embryo=embryo, **_official_summarise([r for r in subset if r["stem"].startswith(embryo + "_")])))
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))

for stem in sorted(reference.stem.unique()):
    nodes = {int(r["node_id"]): r for r in pd.read_csv(source / f"{stem}_anchor_nodes.csv", float_precision="round_trip").to_dict("records")}
    edges = pd.read_csv(source / f"{stem}_anchor_edges.csv", float_precision="round_trip").to_dict("records")
    pairs = set(map(tuple, pd.read_csv(source / f"{stem}_hoct_pairs.csv").to_numpy(dtype=int)))
    outgoing = defaultdict(set)
    for e in edges:
        outgoing[int(e["source_id"])].add(int(e["target_id"]))
    veto = [e for e in edges if len(outgoing[int(e["source_id"])]) == 2 or (int(e["source_id"]), int(e["target_id"])) in pairs]
    # Use original anchor continuations to avoid making a candidate's persistence
    # depend on its own proposed edits. Same 3-step/0um/2um guards as E0069.
    def chain(node):
        result = [node]
        for _ in range(3):
            children = outgoing[result[-1]]
            if len(children) != 1:
                break
            result.append(next(iter(children)))
        return result
    def point(node):
        return np.array([nodes[node][k] for k in ("z", "y", "x")]) * VOXEL_SCALE_UM
    proposed = defaultdict(set)
    for s, t in pairs:
        proposed[s].add(t)
    guards = {config: set() for config in configs[2:]}
    for parent, children in proposed.items():
        if len(children) != 2:
            continue
        a, b = sorted(children)
        ca, cb = chain(a), chain(b)
        persists = len(ca) == len(cb) == 4
        gain = float(np.linalg.norm(point(ca[-1]) - point(cb[-1])) - np.linalg.norm(point(a) - point(b))) if persists else None
        features.append(dict(stem=stem, parent_id=int(parent), daughter1_id=int(a), daughter2_id=int(b), persistence3=persists, separation_gain3=gain))
        if persists:
            guards["persistence3"].add(parent)
            if gain >= 0:
                guards["persistence3_nonconverging"].add(parent)
            if gain >= 2:
                guards["persistence3_diverge2um"].add(parent)
    variants = {"ordinary_veto": (veto, [])}
    variants["unguarded_forks"] = fuse_hoct_forks(nodes, veto, pairs)
    for config, allowed in guards.items():
        variants[config] = fuse_hoct_forks(nodes, veto, {(s, t) for s, t in pairs if s in allowed})
    # Annotation loading starts only after every variant is constructed.
    gt = td.graph.IndexedRXGraph.from_geff(train / f"{stem}.geff")
    if isinstance(gt, tuple):
        gt = gt[0]
    metadata = _GeffMetadata.read(train / f"{stem}.geff")
    for config, (selected, edits) in variants.items():
        pred = _official_graph_from_processed(nodes, selected)
        result = _official_evaluate(pred, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
        if config in ("ordinary_veto", "unguarded_forks"):
            expected_config = "ordinary_veto" if config == "ordinary_veto" else "veto_orphan_forks"
            expected = reference[(reference.stem == stem) & (reference.config == expected_config)].iloc[0]
            for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes"):
                assert getattr(result, key) == int(expected[key]), (stem, config, key)
        row = _official_per_sample_metrics(result, float((metadata.extra or {})["estimated_number_of_nodes"]), _official_node_recall(pred, gt))
        row.update(stem=stem, config=config, edits=len(edits))
        rows.append(row)
        persist()
        print("E0080_RESULT", row, flush=True)
assert len(rows) == reference.stem.nunique() * len(configs)
print("E0080_COMPLETE", flush=True)
