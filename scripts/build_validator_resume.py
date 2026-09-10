#!/usr/bin/env python3
"""Build a non-mutating scoring notebook for completed Biohub predictions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-method", required=True)
    parser.add_argument("--samples-per-prefix", type=int, default=6)
    parser.add_argument("--official-template", type=Path,
                        default=Path("local_runs/E0045/e0045-local.ipynb"))
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    official = json.loads(args.official_template.read_text())
    config_env = "".join(source["cells"][0]["source"])
    config_env += '''
# Match the production notebook's materialized DeepCenter checkpoint.
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(
    "/home/naveen/Biohub/data/kaggle_input/datasets/pilkwang/"
    "biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
)
'''
    config = "".join(source["cells"][2]["source"])

    graph_pipeline = "".join(source["cells"][5]["source"])
    cutoff = "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()"
    end = graph_pipeline.index(cutoff) + len(cutoff)
    graph_pipeline = graph_pipeline[:end]

    selection = f'''# Select the same division-aware panel used by the validated parent.
TRAIN_DIR = COMP_DIR / "train"
VALIDATOR_ENABLE = True
VALIDATOR_N_PER_TYPE = {args.samples_per_prefix}
VALIDATOR_MATCH_RADIUS_UM = 7.0
VALIDATOR_NODE_COUNT_PENALTY_A = 0.1
VALIDATOR_DIVISION_WEIGHT = 0.1
VALIDATOR_STATS_PATH = WORKING_DIR / "validator_results.csv"

def _stem_has_gt_division(stem):
    graph = graph_from_geff(TRAIN_DIR / f"{{stem}}.geff")
    out_degree = {{}}
    for row in graph.edge_attrs().iter_rows(named=True):
        source_id = int(row["source_id"])
        out_degree[source_id] = out_degree.get(source_id, 0) + 1
    return any(degree >= 2 for degree in out_degree.values())

by_prefix = {{}}
test_stems = {{path.name[:-5] for path in TEST_DIR.glob("*.zarr")}}
for path in sorted(TRAIN_DIR.glob("*.zarr")):
    stem = path.name[:-5]
    if stem in test_stems:
        continue
    by_prefix.setdefault(stem.split("_")[0], []).append(stem)
division_flags = {{stem: _stem_has_gt_division(stem) for stems in by_prefix.values() for stem in stems}}
val_stems = []
for prefix, stems in sorted(by_prefix.items()):
    ranked = sorted(stems, key=lambda stem: (not division_flags[stem], stem))
    val_stems.extend(ranked[:VALIDATOR_N_PER_TYPE])
print("Scoring held-out panel:", val_stems)
'''

    proxy = "".join(official["cells"][8]["source"])
    old_lookup = '''    val_pred_paths = {
        stem: found
        for stem in val_stems
        if (found := next((REPO_DIR / "predictions").rglob(f"{stem}.geff"), None)) is not None
    }'''
    new_lookup = f'''    _raw_prediction_dir = REPO_DIR / "predictions" / "naveen" / "{args.raw_method}" / "split_0"
    val_pred_paths = {{
        stem: candidate
        for stem in val_stems
        if (candidate := _raw_prediction_dir / f"{{stem}}.geff").exists()
    }}'''
    if proxy.count(old_lookup) != 1:
        raise RuntimeError("Official template prediction lookup changed")
    proxy = proxy.replace(old_lookup, new_lookup, 1)
    official_metric = "".join(official["cells"][9]["source"])

    notebook = {
        "cells": [code_cell(part) for part in (
            config_env, config, graph_pipeline, selection, proxy, official_metric,
        )],
        "metadata": copy.deepcopy(source.get("metadata", {})),
        "nbformat": 4, "nbformat_minor": 5,
    }
    notebook["metadata"].pop("papermill", None)
    notebook["metadata"].pop("codex", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
