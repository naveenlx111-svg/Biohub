#!/usr/bin/env python3
"""Embryo-held-out fusion of offset pooling and fork-context evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression


KEYS = ["stem", "embryo", "fork_id", "daughter1_id", "daughter2_id"]
TOP_K = (1, 2, 3, 5, 10, 20, 50, 100)


def rank01(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def strict_rows(frame: pd.DataFrame, score: str, model: str) -> list[dict]:
    ordered = frame.sort_values(score, ascending=False, kind="mergesort")
    rows = []
    for k in TOP_K:
        chosen = ordered.head(k)
        tp = int(chosen["label"].sum())
        rows.append({"model": model, "scope": "global", "k": k,
                     "accepted": len(chosen), "tp": tp, "fp": len(chosen) - tp})
    for k in (1, 2, 3, 5, 10):
        chosen = frame.sort_values(score, ascending=False, kind="mergesort").groupby(
            "stem", sort=False
        ).head(k)
        tp = int(chosen["label"].sum())
        rows.append({"model": model, "scope": "per_movie", "k": k,
                     "accepted": len(chosen), "tp": tp, "fp": len(chosen) - tp})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    parent = pd.read_csv(args.working / "parent_event_oof.csv")
    offset = pd.read_csv(args.working / "offset_pool_transfer_oof.csv")
    merged = parent.merge(offset, on=KEYS, suffixes=("", "_offset"), validate="one_to_one")
    if not np.array_equal(merged["label"].to_numpy(), merged["label_offset"].to_numpy()):
        raise RuntimeError("Label mismatch after offset/context join")
    merged = merged.drop(columns=["label_offset"])

    for name in ("center_score", "offset_max_score", "offset_top2_mean_score", "offset_mean_score"):
        merged[f"{name}_movie_rank"] = merged.groupby("stem")[name].transform(rank01)
        merged[f"{name}_embryo_rank"] = merged.groupby("embryo")[name].transform(rank01)
    merged["offset_gain"] = merged["offset_max_score"] - merged["center_score"]
    merged["offset_support"] = merged["offset_top2_mean_score"] / merged["offset_max_score"].clip(1e-7)

    excluded = set(KEYS + ["label", "t", "candidate_index", "rank_within_fork", "score"])
    features = [column for column in merged.columns
                if column not in excluded and pd.api.types.is_numeric_dtype(merged[column])]
    x = merged[features].replace([np.inf, -np.inf], np.nan)
    y = merged["label"].astype(int).to_numpy()

    factories = {
        "extra_trees": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=160, min_samples_leaf=2,
                                 max_features=0.7, class_weight="balanced",
                                 random_state=4801, n_jobs=-1),
        ),
        "hist_gradient": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06,
                                           max_leaf_nodes=15, min_samples_leaf=30,
                                           l2_regularization=2.0, random_state=4802),
        ),
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"), RobustScaler(),
            LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000,
                               random_state=4803),
        ),
    }

    weights = np.where(y == 1, max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)), 1.0)
    embryos = sorted(merged["embryo"].unique())
    fold_rows = []
    for name, factory in factories.items():
        prediction = np.full(len(merged), np.nan)
        for held_out in embryos:
            test = merged["embryo"].eq(held_out).to_numpy()
            train = ~test
            model = factory()
            fit_kwargs = {}
            if name == "hist_gradient":
                fit_kwargs["histgradientboostingclassifier__sample_weight"] = weights[train]
            model.fit(x.loc[train], y[train], **fit_kwargs)
            prediction[test] = model.predict_proba(x.loc[test])[:, 1]
            fold_rows.append({
                "model": name, "held_out": held_out,
                "average_precision": average_precision_score(y[test], prediction[test]),
                "roc_auc": roc_auc_score(y[test], prediction[test]),
                "positives": int(y[test].sum()), "candidates": int(test.sum()),
            })
        merged[f"score_{name}"] = prediction

    # Label-free rank fusions provide stable fallbacks if learned calibration shifts.
    merged["score_offset_geometry_rank"] = (
        0.75 * merged["offset_max_score_movie_rank"]
        + 0.15 * merged.groupby("stem")["event_score"].transform(rank01)
        + 0.10 * merged.groupby("stem")["deepcenter_score"].transform(rank01)
    )
    merged["score_offset_support_rank"] = (
        0.80 * merged["offset_max_score_movie_rank"]
        + 0.20 * merged.groupby("stem")["offset_support"].transform(rank01)
    )

    model_scores = [f"score_{name}" for name in factories] + [
        "offset_max_score", "offset_top2_mean_score",
        "score_offset_geometry_rank", "score_offset_support_rank",
    ]
    metric_rows = []
    threshold_rows = []
    for score in model_scores:
        metric_rows.append({
            "model": score, "average_precision": average_precision_score(y, merged[score]),
            "roc_auc": roc_auc_score(y, merged[score]),
        })
        threshold_rows.extend(strict_rows(merged, score, score))

    metrics = pd.DataFrame(metric_rows).sort_values("average_precision", ascending=False)
    thresholds = pd.DataFrame(threshold_rows)
    folds = pd.DataFrame(fold_rows)
    merged.to_csv(args.output / "e0048_oof.csv", index=False)
    metrics.to_csv(args.output / "e0048_metrics.csv", index=False)
    folds.to_csv(args.output / "e0048_folds.csv", index=False)
    thresholds.to_csv(args.output / "e0048_thresholds.csv", index=False)

    useful = thresholds.query("tp >= 2 and fp <= 1")
    probe = thresholds.query("tp >= 1 and fp == 0")
    summary = {
        "candidates": int(len(merged)), "positives": int(y.sum()),
        "features": features, "metrics": metrics.to_dict("records"),
        "useful_operating_points": useful.to_dict("records"),
        "zero_fp_operating_points": probe.to_dict("records"),
    }
    (args.output / "e0048_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print("\nStrict head:\n", thresholds.query("scope == 'global' and k <= 20").to_string(index=False))
    print("\nUseful operating points:\n", useful.to_string(index=False) if len(useful) else "none")


if __name__ == "__main__":
    main()
