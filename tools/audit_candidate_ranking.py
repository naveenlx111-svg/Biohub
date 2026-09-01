#!/usr/bin/env python3
"""Audit candidate-parent scores at exact low-false-positive cutoffs.

The E0043/E0044 notebooks report fixed score quantiles.  On a pool of more
than 100,000 proposals, even the 0.9999 quantile retains about twelve rows and
can hide useful top-1/top-2 behavior.  This script evaluates every distinct
score cutoff and the same cutoff independently within each movie.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path, score_column: str) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = [
            {
                "stem": row["stem"],
                "embryo": row.get("embryo", row["stem"].split("_", 1)[0]),
                "label": int(row["label"]),
                "score": float(row[score_column]),
            }
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def exact_cutoffs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    points: list[dict[str, object]] = []
    tp = fp = 0
    for index, row in enumerate(ranked):
        tp += int(row["label"])
        fp += 1 - int(row["label"])
        next_score = (
            float(ranked[index + 1]["score"])
            if index + 1 < len(ranked)
            else float("-inf")
        )
        score = float(row["score"])
        if next_score == score:
            continue
        points.append(
            {
                "accepted": index + 1,
                "tp": tp,
                "fp": fp,
                "threshold": score,
                "precision": tp / (index + 1),
            }
        )
    return points


def best_under_fp(points: list[dict[str, object]], fp_limit: int) -> dict[str, object]:
    eligible = [point for point in points if int(point["fp"]) <= fp_limit]
    if not eligible:
        return {"accepted": 0, "tp": 0, "fp": 0, "threshold": None, "precision": 0.0}
    return max(
        eligible,
        key=lambda point: (int(point["tp"]), -int(point["fp"]), float(point["threshold"])),
    )


def movie_top_k(rows: list[dict[str, object]], k: int) -> dict[str, object]:
    by_stem: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_stem[str(row["stem"])].append(row)
    selected = []
    for stem_rows in by_stem.values():
        selected.extend(sorted(stem_rows, key=lambda row: float(row["score"]), reverse=True)[:k])
    tp = sum(int(row["label"]) for row in selected)
    return {"k_per_movie": k, "accepted": len(selected), "tp": tp, "fp": len(selected) - tp}


def point_at_least_k(points: list[dict[str, object]], k: int) -> dict[str, object]:
    """Return the first tie-safe cutoff retaining at least k candidates."""
    return next(
        (point for point in points if int(point["accepted"]) >= k),
        points[-1],
    )


def apply_threshold(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    accepted = [row for row in rows if float(row["score"]) >= threshold]
    tp = sum(int(row["label"]) for row in accepted)
    return {
        "accepted": len(accepted),
        "tp": tp,
        "fp": len(accepted) - tp,
        "threshold": threshold,
        "precision": tp / len(accepted) if accepted else 0.0,
    }


def embryo_threshold_transfer(
    rows: list[dict[str, object]], fp_limit: int
) -> list[dict[str, object]]:
    """Choose a low-FP threshold on one embryo and freeze it on another."""
    embryos = sorted({str(row["embryo"]) for row in rows})
    transfers = []
    for train_embryo in embryos:
        train_rows = [row for row in rows if str(row["embryo"]) == train_embryo]
        selected = best_under_fp(exact_cutoffs(train_rows), fp_limit)
        if selected["threshold"] is None:
            continue
        threshold = float(selected["threshold"])
        for test_embryo in embryos:
            if test_embryo == train_embryo:
                continue
            test_rows = [row for row in rows if str(row["embryo"]) == test_embryo]
            transfers.append(
                {
                    "train_embryo": train_embryo,
                    "test_embryo": test_embryo,
                    "train": apply_threshold(train_rows, threshold),
                    "test": apply_threshold(test_rows, threshold),
                }
            )
    return transfers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oof_csv", type=Path)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.oof_csv, args.score_column)
    points = exact_cutoffs(rows)
    report = {
        "score_column": args.score_column,
        "rows": len(rows),
        "positives": sum(int(row["label"]) for row in rows),
        "best_global_at_0_fp": best_under_fp(points, 0),
        "best_global_at_1_fp": best_under_fp(points, 1),
        "embryo_threshold_transfer_at_0_fp": embryo_threshold_transfer(rows, 0),
        "embryo_threshold_transfer_at_1_fp": embryo_threshold_transfer(rows, 1),
        "global_top_k": [
            point_at_least_k(points, k)
            for k in (1, 2, 3, 5, 10)
        ],
        "movie_top_k": [movie_top_k(rows, k) for k in (1, 2, 3, 5)],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
