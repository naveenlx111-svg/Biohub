"""Run public graph postprocessing and pinned official scoring on cached graphs.

Detector inference is not repeated. DeepCenter uses the requested local device;
matching and official scoring run on CPU. No downloaded files are modified.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["harmonic", "lineage"], required=True)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--configs", default="base,tight55")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "local_runs/frontier_20260909" / args.arm
output = folder / f"local_{args.device}_audit"
output.mkdir(exist_ok=True)
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
import torch
torch.set_num_threads(4)
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable")
inputs = ROOT / "data/kaggle_input"
checkpoint = inputs / "datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
namespace = {"__name__": "frontier_cached_audit"}

def run(source, label):
    source = source.replace("/kaggle/input", str(inputs)).replace("/kaggle/working", str(output))
    exec(compile(source, label, "exec"), namespace)

run((folder / "cell_00.py").read_text(), "public_config")
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(checkpoint)
run((folder / "cell_02.py").read_text(), "public_constants")
namespace["REPO_DIR"] = folder / "output/tracking_repo"
namespace["WORKING_DIR"] = output
source = (folder / "cell_05.py").read_text()
source = source[:source.index("def write_test_submission(")]
run(source, "public_graph_functions")
selection = json.loads((folder / "output/ppsweep_selected.json").read_text())
stems = selection["held_out_stems"]
namespace.update(TRAIN_DIR=namespace["COMP_DIR"] / "train", val_stems=stems,
                 VALIDATOR_ENABLE=True, VALIDATOR_MATCH_RADIUS_UM=7.0)
notebook = json.loads((ROOT / "research_members/naveen/experiments/E0033_lb0933_official_12/e0033-lb0933-official-12.ipynb").read_text())
setup = "".join(notebook["cells"][9]["source"]).split("official_rows = []")[0]
run(setup, "pinned_official_scorer")
import pandas as pd

configs = {"base": {}, "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
           "divergence30": {"MOTION_RELINK_TIGHT_UM": 5.5, "SAFE_DIV_DIVERGE_UM": 3.0},
           "divergence40": {"MOTION_RELINK_TIGHT_UM": 5.5, "SAFE_DIV_DIVERGE_UM": 4.0},
           "dc026": {"MOTION_RELINK_TIGHT_UM": 5.5, "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.26}}
requested = args.configs.split(",")
assert all(c in configs for c in requested)
samples = []
summaries = []
namespace["TEST_DIR"] = namespace["TRAIN_DIR"]
# Shared read-only heatmap cache avoids repeating identical inference across
# configurations. Values are the unmodified public function outputs.
heatmap_function = namespace["deepcenter_heatmap_for_frame"]
heatmaps = {}
def cached_heatmap(dataset, t, detector_bundle, frame_cache, heatmap_cache):
    key = (dataset, int(t))
    if key not in heatmaps:
        heatmaps[key] = heatmap_function(dataset, t, detector_bundle, frame_cache, heatmap_cache)
    return heatmaps[key]
namespace["deepcenter_heatmap_for_frame"] = cached_heatmap
original = {key: namespace[key] for config in configs.values() for key in config}
for stem in stems:
    heatmaps.clear()
    paths = list((folder / "output/tracking_repo/predictions").glob(f"*/unet_transformer_val/split_0/{stem}.geff"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one cached graph for {stem}: {paths}")
    graph = namespace["graph_from_geff"](paths[0])
    nodes = {int(row["node_id"]): {"node_id": int(row["node_id"]), "t": int(row["t"]),
              "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
             for row in graph.node_attrs().iter_rows(named=True)}
    edges = [{"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
              "edge_prob": row.get("edge_prob")} for row in graph.edge_attrs().iter_rows(named=True)]
    for label in requested:
        namespace.update(original)
        namespace.update(configs[label])
        started = time.monotonic()
        processed_nodes, processed_edges, stats = namespace["filter_output_graph"](
            copy.deepcopy(nodes), copy.deepcopy(edges), dataset=stem,
            deepcenter_bundle=namespace["DEEPCENTER_VETO_DETECTOR"])
        pred = namespace["_official_graph_from_processed"](processed_nodes, processed_edges)
        gt = namespace["graph_from_geff"](namespace["TRAIN_DIR"] / f"{stem}.geff")
        result = namespace["_official_evaluate"](pred, gt, scale=tuple(namespace["VOXEL_SCALE_UM"]), max_distance=7.0)
        recall = namespace["_official_node_recall"](pred, gt) if pred.num_nodes() and pred.num_edges() else 0.0
        meta = namespace["_GeffMetadata"].read(namespace["TRAIN_DIR"] / f"{stem}.geff")
        total = float((meta.extra or {})["estimated_number_of_nodes"])
        row = namespace["_official_per_sample_metrics"](result, total, recall)
        row.update(config=label, stem=stem, embryo=stem.split("_")[0], seconds=time.monotonic()-started)
        samples.append(row)
        pd.DataFrame(samples).to_csv(output / "samples.csv", index=False)
        print("OFFICIAL", json.dumps(row, default=str), flush=True)
for label in requested:
    group = [r for r in samples if r["config"] == label]
    summaries.append(dict(config=label, **namespace["_official_summarise"](group)))
    for embryo in sorted({r["embryo"] for r in group}):
        summaries.append(dict(config=label, embryo=embryo,
                              **namespace["_official_summarise"]([r for r in group if r["embryo"] == embryo])))
(output / "summary.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n")
print("COMPLETE", json.dumps(summaries, indent=2, default=str), flush=True)
