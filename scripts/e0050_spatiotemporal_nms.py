#!/usr/bin/env python3
"""Greedy physical-space/time suppression of high-scoring division candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=float)
RADII = [3.25, 5.0, 7.0, 10.0, 15.0]
TIME_WINDOWS = [1, 2, 4, 8, 15]
TOP_K = [1, 2, 3, 5, 10, 20, 30, 50, 100]


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def node_positions(path: Path) -> dict[int, tuple[float, float, float]]:
    graph = graph_from_geff(path)
    return {
        int(row["node_id"]): tuple(np.array([row["z"], row["y"], row["x"]], dtype=float) * SCALE)
        for row in graph.node_attrs().iter_rows(named=True)
    }


def suppress(group: pd.DataFrame, radius: float, time_window: int, score: str) -> pd.DataFrame:
    ordered = group.sort_values(score, ascending=False, kind="mergesort")
    bins: dict[tuple[int, int, int], list[tuple[int, np.ndarray]]] = {}
    accepted = []
    for index, row in ordered.iterrows():
        if not bool(row["raw_node_matched"]):
            accepted.append(index)
            continue
        point = np.array([row["z_um"], row["y_um"], row["x_um"]], dtype=float)
        cell = tuple(np.floor(point / radius).astype(int))
        conflict = False
        for delta in itertools.product((-1, 0, 1), repeat=3):
            neighbor = tuple(cell[axis] + delta[axis] for axis in range(3))
            for previous_time, previous_point in bins.get(neighbor, ()):
                if abs(int(row["t"]) - previous_time) <= time_window and np.linalg.norm(point - previous_point) <= radius:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            accepted.append(index)
            bins.setdefault(cell, []).append((int(row["t"]), point))
    return group.loc[accepted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.working / "offset_pool_transfer_oof.csv")
    context = pd.read_csv(args.working / "parent_event_oof.csv",
                          usecols=["stem", "embryo", "fork_id", "daughter1_id", "daughter2_id", "t"])
    keys = ["stem", "embryo", "fork_id", "daughter1_id", "daughter2_id"]
    candidates = candidates.merge(context, on=keys, validate="one_to_one")
    positions = {}
    for stem in sorted(candidates["stem"].unique()):
        for node, position in node_positions(args.predictions / f"{stem}.geff").items():
            positions[(stem, node)] = position
    coords = [positions.get((stem, int(node))) for stem, node in zip(candidates.stem, candidates.fork_id)]
    candidates["raw_node_matched"] = [coord is not None for coord in coords]
    for axis, name in enumerate(("z_um", "y_um", "x_um")):
        candidates[name] = [coord[axis] if coord is not None else math.nan for coord in coords]

    rows = []
    for score in ("offset_max_score", "offset_top2_mean_score"):
        for radius in RADII:
            for time_window in TIME_WINDOWS:
                retained_parts = [
                    suppress(group, radius, time_window, score)
                    for _, group in candidates.groupby("stem", sort=False)
                ]
                retained = pd.concat(retained_parts, ignore_index=False)
                ordered = retained.sort_values(score, ascending=False, kind="mergesort")
                row = {"score": score, "radius_um": radius, "time_window": time_window,
                       "retained": len(retained), "retained_positives": int(retained.label.sum())}
                for k in TOP_K:
                    chosen = ordered.head(k); tp = int(chosen.label.sum())
                    row[f"top{k}_tp"] = tp; row[f"top{k}_fp"] = len(chosen) - tp
                rows.append(row)
    results = pd.DataFrame(rows).sort_values(
        ["top20_tp", "top20_fp", "top50_tp", "top50_fp"], ascending=[False, True, False, True]
    )
    results.to_csv(args.output / "e0050_results.csv", index=False)
    useful = []
    for row in rows:
        for k in TOP_K:
            if row[f"top{k}_tp"] >= 1 and row[f"top{k}_fp"] == 0:
                useful.append({"score": row["score"], "radius_um": row["radius_um"],
                               "time_window": row["time_window"], "k": k,
                               "tp": row[f"top{k}_tp"], "fp": 0})
    summary = {"candidates": len(candidates), "positives": int(candidates.label.sum()),
               "raw_node_match_fraction": float(candidates.raw_node_matched.mean()),
               "zero_fp_operating_points": useful,
               "best_rows": results.head(10).to_dict("records")}
    (args.output / "e0050_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.head(20).to_string(index=False))
    print("\nZero-FP operating points:", useful)


if __name__ == "__main__":
    main()
