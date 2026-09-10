#!/usr/bin/env python3
"""Suppress repeated division responses along predicted linear track segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tracksdata as td
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ["stem", "embryo", "fork_id", "daughter1_id", "daughter2_id"]
SCORES = ["offset_max_score", "offset_top2_mean_score", "center_score"]
WINDOWS = [0, 2, 4, 6, 10, 20, 1000]
TOP_K = [1, 2, 3, 5, 10, 20, 30, 50, 100]


class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def segment_map(path: Path) -> dict[int, int]:
    graph = graph_from_geff(path)
    node_ids = [int(value) for value in graph.node_attrs()["node_id"].to_list()]
    edges = [(int(row["source_id"]), int(row["target_id"]))
             for row in graph.edge_attrs().iter_rows(named=True)]
    incoming = {}
    outgoing = {}
    for source, target in edges:
        outgoing.setdefault(source, []).append(target)
        incoming.setdefault(target, []).append(source)
    union = UnionFind(node_ids)
    for source, target in edges:
        if len(outgoing.get(source, ())) == 1 and len(incoming.get(target, ())) == 1:
            union.union(source, target)
    roots = {node: union.find(node) for node in node_ids}
    stable = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    return {node: stable[root] for node, root in roots.items()}


def suppress(frame: pd.DataFrame, score: str, window: int) -> pd.DataFrame:
    if window == 0:
        return frame.copy()
    kept = []
    for _, group in frame.groupby(["stem", "track_segment"], sort=False):
        accepted_times = []
        for index, row in group.sort_values(score, ascending=False, kind="mergesort").iterrows():
            time = int(row["t"])
            if all(abs(time - previous) > window for previous in accepted_times):
                kept.append(index)
                accepted_times.append(time)
    return frame.loc[kept].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    offset = pd.read_csv(args.working / "offset_pool_transfer_oof.csv")
    context = pd.read_csv(args.working / "parent_event_oof.csv", usecols=KEYS + ["t", "label"])
    frame = offset.merge(context, on=KEYS, suffixes=("", "_context"), validate="one_to_one")
    if not frame["label"].equals(frame["label_context"]):
        raise RuntimeError("Label mismatch")
    frame = frame.drop(columns=["label_context"])

    segments = {}
    for stem in sorted(frame["stem"].unique()):
        mapping = segment_map(args.predictions / f"{stem}.geff")
        for node, segment in mapping.items():
            segments[(stem, node)] = f"raw:{segment}"
    frame["track_segment"] = [
        segments.get((stem, int(node)), f"unmatched:{int(node)}")
        for stem, node in zip(frame["stem"], frame["fork_id"])
    ]
    frame["raw_node_matched"] = [
        (stem, int(node)) in segments for stem, node in zip(frame["stem"], frame["fork_id"])
    ]

    result_rows = []
    retained_frames = []
    for score in SCORES:
        for window in WINDOWS:
            retained = suppress(frame, score, window)
            retained["nms_score"] = score
            retained["nms_window"] = window
            retained_frames.append(retained)
            ordered = retained.sort_values(score, ascending=False, kind="mergesort")
            metrics = {
                "score": score, "window": window, "retained": len(retained),
                "retained_positives": int(retained["label"].sum()),
                "average_precision": average_precision_score(retained["label"], retained[score]),
                "roc_auc": roc_auc_score(retained["label"], retained[score]),
            }
            for k in TOP_K:
                selected = ordered.head(k)
                tp = int(selected["label"].sum())
                metrics[f"top{k}_tp"] = tp
                metrics[f"top{k}_fp"] = len(selected) - tp
            result_rows.append(metrics)

    results = pd.DataFrame(result_rows).sort_values(
        ["top20_tp", "top20_fp", "average_precision"], ascending=[False, True, False]
    )
    results.to_csv(args.output / "e0049_results.csv", index=False)
    frame.to_csv(args.output / "e0049_candidates_with_segments.csv", index=False)
    useful = []
    for row in result_rows:
        for k in TOP_K:
            tp, fp = row[f"top{k}_tp"], row[f"top{k}_fp"]
            if tp >= 1 and fp == 0:
                useful.append({"score": row["score"], "window": row["window"],
                               "k": k, "tp": tp, "fp": fp})
    summary = {
        "candidates": len(frame), "positives": int(frame["label"].sum()),
        "raw_node_match_fraction": float(frame["raw_node_matched"].mean()),
        "positive_raw_node_matches": int(frame.query("label == 1")["raw_node_matched"].sum()),
        "positive_count": int(frame["label"].sum()),
        "zero_fp_operating_points": useful,
        "best_rows": results.head(10).to_dict("records"),
    }
    (args.output / "e0049_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.head(20).to_string(index=False))
    print("\nZero-FP operating points:", useful)


if __name__ == "__main__":
    main()
