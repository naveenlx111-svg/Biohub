"""Check Biohub submission structure and physical bounds on local CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

parser = argparse.ArgumentParser()
parser.add_argument("submission", type=Path)
parser.add_argument("--competition", type=Path, default=Path("data/kaggle_input/competitions/biohub-cell-tracking-during-development"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
frame = pd.read_csv(args.submission)
columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
assert frame.columns.tolist() == columns, frame.columns.tolist()
assert np.array_equal(frame.id.to_numpy(), np.arange(len(frame))), "Invalid row IDs"
assert set(frame.row_type) <= {"node", "edge"}, "Unknown row type"
expected = {p.name[:-5] for p in (args.competition / "test").glob("*.zarr")}
assert expected and set(frame.dataset) == expected, "Missing or unexpected test datasets"
rows = []
for dataset, group in frame.groupby("dataset"):
    image = zarr.open(str(args.competition / "test" / f"{dataset}.zarr"), mode="r")
    if not hasattr(image, "shape"):
        image = image["0"]
    shape = image.shape
    assert len(shape) == 4, shape
    nodes = group[group.row_type.eq("node")]
    edges = group[group.row_type.eq("edge")]
    assert len(nodes) and len(edges), f"Empty graph: {dataset}"
    assert nodes.node_id.is_unique, f"Duplicate nodes: {dataset}"
    values = nodes[["node_id", "t", "z", "y", "x"]].to_numpy(float)
    assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), "Non-integer nodes"
    coords = values[:, 1:]
    assert (coords >= 0).all() and (coords < np.asarray(shape)).all(), f"Out-of-bounds: {dataset}"
    links = edges[["source_id", "target_id"]].to_numpy(float)
    assert np.isfinite(links).all() and np.equal(links, np.floor(links)).all(), "Non-integer edges"
    assert not edges.duplicated(["source_id", "target_id"]).any(), "Duplicate edges"
    times = nodes.set_index("node_id").t
    source_times = edges.source_id.map(times)
    target_times = edges.target_id.map(times)
    assert source_times.notna().all() and target_times.notna().all(), "Dangling edges"
    assert (target_times - source_times).eq(1).all(), "Non-consecutive edge"
    assert edges.target_id.value_counts().max() <= 1, "Multiple parents"
    assert edges.source_id.value_counts().max() <= 2, "More than two children"
    rows.append(dict(dataset=dataset, shape=list(shape), nodes=len(nodes), edges=len(edges),
                     divisions=int(edges.source_id.value_counts().eq(2).sum())))
report = dict(submission=str(args.submission), sha256=hashlib.sha256(args.submission.read_bytes()).hexdigest(),
              valid=True, datasets=rows, nodes=sum(r["nodes"] for r in rows), edges=sum(r["edges"] for r in rows))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
