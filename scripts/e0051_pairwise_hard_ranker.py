#!/usr/bin/env python3
"""Embryo-held-out pairwise ranker against hardest same-movie negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


HARD_POOLS = [100, 500, 2000, 5000]
ALPHAS = [1e-5, 1e-4, 1e-3]
TOP_K = [1, 2, 3, 5, 10, 20, 30, 50, 100]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)

    feature_names = [
        "d1_um", "d2_um", "distance_sum_um", "distance_asymmetry_um", "sister_um",
        "midpoint_um", "daughter_cosine", "fork_outdegree", "daughter1_indegree",
        "daughter2_indegree", "edge1_exists", "edge2_exists", "parent_mean",
        "parent_std", "parent_max", "parent_p90", "parent_inner", "parent_contrast",
        "parent_mean_delta_prev", "parent_mean_delta_next", "parent_std_delta_prev",
        "parent_std_delta_next", "daughter_mean_average", "daughter_mean_asymmetry",
        "parent_minus_daughters", "deepcenter_score", "event_score", "center_score",
        "offset_max_score", "offset_top2_mean_score", "offset_mean_score", "offset_gain",
        "offset_support",
    ]
    feature_names = [name for name in feature_names if name in frame]
    raw = frame[feature_names].replace([np.inf, -np.inf], np.nan)
    labels = frame.label.astype(int).to_numpy()
    embryos = sorted(frame.embryo.unique())
    variants = []
    rng = np.random.default_rng(5100)

    for hard_pool in HARD_POOLS:
        for alpha in ALPHAS:
            for loss in ("hinge", "log_loss"):
                prediction = np.full(len(frame), np.nan)
                for held_out in embryos:
                    train_mask = ~frame.embryo.eq(held_out).to_numpy()
                    test_mask = ~train_mask
                    imputer = SimpleImputer(strategy="median")
                    scaler = RobustScaler()
                    train_x = scaler.fit_transform(imputer.fit_transform(raw.loc[train_mask]))
                    test_x = scaler.transform(imputer.transform(raw.loc[test_mask]))
                    train_frame = frame.loc[train_mask].reset_index(drop=True)
                    train_y = train_frame.label.astype(int).to_numpy()
                    pair_parts = []
                    for positive_index in np.flatnonzero(train_y == 1):
                        stem = train_frame.loc[positive_index, "stem"]
                        negative_indices = np.flatnonzero(
                            train_frame.stem.eq(stem).to_numpy() & (train_y == 0)
                        )
                        ordered = negative_indices[
                            np.argsort(-train_frame.loc[negative_indices, "offset_max_score"].to_numpy())
                        ]
                        hard = ordered[:hard_pool]
                        if len(negative_indices) > hard_pool:
                            random_count = min(200, len(negative_indices) - hard_pool)
                            reservoir = rng.choice(negative_indices[hard_pool:], size=random_count, replace=False)
                            chosen = np.unique(np.concatenate([hard, reservoir]))
                        else:
                            chosen = hard
                        pair_parts.append(train_x[positive_index] - train_x[chosen])
                    differences = np.vstack(pair_parts)
                    symmetric_x = np.vstack([differences, -differences])
                    symmetric_y = np.concatenate([
                        np.ones(len(differences), dtype=int), np.zeros(len(differences), dtype=int)
                    ])
                    model = SGDClassifier(loss=loss, alpha=alpha, max_iter=3000, tol=1e-5,
                                          random_state=5101, average=True)
                    model.fit(symmetric_x, symmetric_y)
                    prediction[test_mask] = model.decision_function(test_x)

                name = f"pair_{loss}_hard{hard_pool}_a{alpha:g}"
                frame[name] = prediction
                ordered = frame.sort_values(name, ascending=False, kind="mergesort")
                row = {"model": name, "hard_pool": hard_pool, "alpha": alpha, "loss": loss,
                       "average_precision": average_precision_score(labels, prediction),
                       "roc_auc": roc_auc_score(labels, prediction)}
                for k in TOP_K:
                    chosen = ordered.head(k); tp = int(chosen.label.sum())
                    row[f"top{k}_tp"] = tp; row[f"top{k}_fp"] = len(chosen) - tp
                variants.append(row)

    results = pd.DataFrame(variants).sort_values(
        ["top20_tp", "top20_fp", "average_precision"], ascending=[False, True, False]
    )
    results.to_csv(args.output / "e0051_results.csv", index=False)
    frame.to_csv(args.output / "e0051_oof.csv", index=False)
    useful = []
    for row in variants:
        for k in TOP_K:
            if row[f"top{k}_tp"] >= 1 and row[f"top{k}_fp"] == 0:
                useful.append({"model": row["model"], "k": k,
                               "tp": row[f"top{k}_tp"], "fp": 0})
    summary = {"candidates": len(frame), "positives": int(labels.sum()),
               "features": feature_names, "zero_fp_operating_points": useful,
               "best_rows": results.head(10).to_dict("records")}
    (args.output / "e0051_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.head(20).to_string(index=False))
    print("\nZero-FP operating points:", useful)


if __name__ == "__main__":
    main()
