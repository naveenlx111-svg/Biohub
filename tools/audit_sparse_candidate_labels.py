"""E0058: distinguish unlabeled proposal parents from observed continuations.

Reconstruct the E0047 processed graph and require exact saved metric counts
before interpreting its internal candidate IDs. No graph edits or fitting.
"""
import json
import os
from pathlib import Path
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / "local_runs/EXT0006/working"
out = ROOT / "local_runs/E0058"
out.mkdir(exist_ok=True)
nb = json.loads((ROOT / "local_runs/E0047/e0047-offset-oracle.ipynb").read_text())
ns = {"__name__": "sparse_label_audit"}
for i in range(3):
    exec(compile("".join(nb["cells"][i]["source"]), f"E0047:{i}", "exec"), ns)
ns["WORKING_DIR"] = out
ns["TRAIN_DIR"] = ns["COMP_DIR"] / "train"
ns["TEST_DIR"] = ns["TRAIN_DIR"]
setup = "".join(nb["cells"][5]["source"]).split("official_rows = []")[0]
exec(compile(setup, "official_setup", "exec"), ns)
expected = pd.read_csv(old / "official_validator_results.csv").set_index("stem")
candidates = pd.read_csv(old / "parent_event_oof.csv")
offsets = pd.read_csv(old / "offset_pool_transfer_oof.csv")
keys = ["stem", "embryo", "fork_id", "daughter1_id", "daughter2_id"]
candidates = candidates.merge(offsets[keys + ["center_score", "offset_max_score", "offset_top2_mean_score"]],
                              on=keys, validate="one_to_one")
parts = []
for stem, group in candidates.groupby("stem", sort=True):
    started = time.monotonic()
    path = old / "tracking_repo/predictions/naveen/unet_transformer_edge20_valid_val12/split_0" / f"{stem}.geff"
    raw = ns["graph_from_geff"](path)
    nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
             "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
             for r in raw.node_attrs().iter_rows(named=True)}
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
              "edge_prob": r.get("edge_prob")} for r in raw.edge_attrs().iter_rows(named=True)]
    nodes, edges, stats = ns["filter_output_graph"](nodes, edges, dataset=stem,
                            deepcenter_bundle=ns["DEEPCENTER_VETO_DETECTOR"])
    pred = ns["_official_graph_from_processed"](nodes, edges)
    gt = ns["graph_from_geff"](ns["TRAIN_DIR"] / f"{stem}.geff")
    result = ns["_official_evaluate"](pred, gt, scale=tuple(ns["VOXEL_SCALE_UM"]), max_distance=7.0)
    for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes"):
        assert getattr(result, key) == int(expected.loc[stem, key]), (stem, key, getattr(result, key), expected.loc[stem,key])
    node_rows = pred.node_attrs().to_pandas().set_index("node_id")
    assert set(group.fork_id.astype(int)) <= set(node_rows.index), "Candidate ID namespace mismatch"
    assert np.array_equal(node_rows.loc[group.fork_id, "t"].to_numpy(), group.t.to_numpy())
    xyz = node_rows[["z", "y", "x"]]
    for daughter, distance in (("daughter1_id", "d1_um"), ("daughter2_id", "d2_um")):
        measured = np.linalg.norm((xyz.loc[group.fork_id].to_numpy() - xyz.loc[group[daughter]].to_numpy()) * np.array(ns["VOXEL_SCALE_UM"]), axis=1)
        assert np.allclose(measured, group[distance], atol=1e-5), f"Candidate geometry differs for {stem}"
    match_key = ns["td"].DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    matched_ids = node_rows.loc[group.fork_id, match_key].to_numpy()
    gt_degree = {int(n): len(gt.successors(int(n))) for n in gt.node_ids()}
    statuses = []
    for label, matched in zip(group.label, matched_ids):
        if int(label) == 1:
            status = "annotated_positive_pair"
        elif pd.isna(matched) or int(matched) < 0:
            status = "unmatched_parent_unknown"
        else:
            degree = gt_degree[int(matched)]
            status = {0: "matched_terminal_unknown", 1: "matched_continuation", 2: "matched_division_other_pair"}.get(degree, "other")
        statuses.append(status)
    group = group.copy()
    group["annotation_status"] = statuses
    parts.append(group)
    # Save the exact processed namespace for subsequent label-free edit audits.
    pred.node_attrs().to_pandas().to_csv(out / f"{stem}_nodes.csv", index=False)
    pred.edge_attrs().to_pandas().to_csv(out / f"{stem}_edges.csv", index=False)
    print(stem, group.annotation_status.value_counts().to_dict(), "seconds", round(time.monotonic()-started,1), flush=True)
frame = pd.concat(parts, ignore_index=True)
frame.to_csv(out / "candidate_annotation_audit.csv", index=False)
report = {"candidates": len(frame), "all": frame.annotation_status.value_counts().to_dict(), "rankings": []}
for score in ("center_score", "offset_max_score", "offset_top2_mean_score"):
    ranked = frame.sort_values(score, ascending=False, kind="stable")
    for k in (12, 58, 100, 250, 500):
        chosen = ranked.head(k)
        report["rankings"].append(dict(score=score, k=k, **chosen.annotation_status.value_counts().to_dict()))
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
