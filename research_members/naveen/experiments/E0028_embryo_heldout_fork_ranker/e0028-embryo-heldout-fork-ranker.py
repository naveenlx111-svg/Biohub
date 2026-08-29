from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def find_candidates() -> Path:
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    found = []
    for root in roots:
        if root.exists():
            found.extend(root.rglob("fork_candidates.csv"))
    if not found:
        raise FileNotFoundError("E0027 fork_candidates.csv is not attached")
    preferred = [p for p in found if "e0027" in str(p).lower()]
    return sorted(preferred or found)[0]


FEATURES = [
    "d1_um",
    "d2_um",
    "distance_sum_um",
    "distance_asymmetry_um",
    "sister_um",
    "midpoint_um",
    "daughter_cosine",
    "fork_outdegree",
    "daughter1_indegree",
    "daughter2_indegree",
    "edge1_exists",
    "edge2_exists",
]


candidate_path = find_candidates()
print("Candidate input:", candidate_path)
df = pd.read_csv(candidate_path)
df["candidate_index"] = np.arange(len(df), dtype=np.int64)

# Every labeled positive satisfies these bounds. They remove geometrically easy
# negatives without using an event location or any ground-truth feature.
hard = df[
    (df["distance_sum_um"] <= 14.0)
    & (df["sister_um"] >= 4.0)
    & (df["midpoint_um"] <= 5.0)
    & (df["daughter_cosine"] <= -0.30)
].copy()
if int(hard["label"].sum()) != int(df["label"].sum()):
    raise RuntimeError("Hard-negative gate discarded a positive candidate")

fold_rows = []
ranked_parts = []
for heldout in sorted(hard["embryo"].unique()):
    train = hard[hard["embryo"] != heldout].copy()
    test = hard[hard["embryo"] == heldout].copy()
    positives = int(train["label"].sum())
    negatives = len(train) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError(f"Invalid training fold for {heldout}: {positives=} {negatives=}")

    sample_weight = np.ones(len(train), dtype=float)
    sample_weight[train["label"].to_numpy() == 1] = negatives / positives
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=20260830,
    )
    model.fit(train[FEATURES], train["label"], sample_weight=sample_weight)
    test["score"] = model.predict_proba(test[FEATURES])[:, 1]
    test["heldout_embryo"] = heldout

    # A deployable graph can accept at most one daughter pair per proposed fork.
    test["rank_within_fork"] = test.groupby(["stem", "fork_id"])["score"].rank(
        method="first", ascending=False
    )
    top1 = test[test["rank_within_fork"] == 1].copy()
    positive_scores = test.loc[test["label"] == 1, "score"]
    positive_ranks = test.loc[test["label"] == 1, "rank_within_fork"]
    fold_rows.append({
        "heldout_embryo": heldout,
        "train_rows": len(train),
        "train_positives": positives,
        "test_rows": len(test),
        "test_positives": int(test["label"].sum()),
        "average_precision": float(average_precision_score(test["label"], test["score"])),
        "roc_auc": float(roc_auc_score(test["label"], test["score"])),
        "positive_top1_within_fork": int((positive_ranks == 1).sum()),
        "positive_top3_within_fork": int((positive_ranks <= 3).sum()),
        "positive_score_min": float(positive_scores.min()),
        "positive_score_median": float(positive_scores.median()),
        "top1_candidates": len(top1),
        "top1_positive": int(top1["label"].sum()),
    })
    ranked_parts.append(test)

ranked = pd.concat(ranked_parts, ignore_index=True)
folds = pd.DataFrame(fold_rows)

threshold_rows = []
for threshold in sorted(set(np.quantile(ranked["score"], [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]).tolist())):
    accepted = ranked[(ranked["rank_within_fork"] == 1) & (ranked["score"] >= threshold)]
    tp = int(accepted["label"].sum())
    fp = len(accepted) - tp
    total_positive = int(ranked["label"].sum())
    threshold_rows.append({
        "threshold": float(threshold),
        "accepted": len(accepted),
        "tp_candidates": tp,
        "fp_candidates": fp,
        "candidate_precision": float(tp / len(accepted)) if len(accepted) else 0.0,
        "candidate_recall": float(tp / total_positive) if total_positive else 0.0,
    })
thresholds = pd.DataFrame(threshold_rows)

summary = {
    "input_candidates": int(len(df)),
    "input_positives": int(df["label"].sum()),
    "hard_candidates": int(len(hard)),
    "hard_positives": int(hard["label"].sum()),
    "fold_average_precision_mean": float(folds["average_precision"].mean()),
    "fold_roc_auc_mean": float(folds["roc_auc"].mean()),
    "positive_top1_within_fork": int(folds["positive_top1_within_fork"].sum()),
    "positive_top3_within_fork": int(folds["positive_top3_within_fork"].sum()),
}

out = Path("/kaggle/working")
folds.to_csv(out / "fork_ranker_folds.csv", index=False)
thresholds.to_csv(out / "fork_ranker_thresholds.csv", index=False)
ranked.to_csv(out / "fork_candidates_oof_ranked.csv", index=False)
(out / "fork_ranker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
print(folds.to_string(index=False))
print(thresholds.to_string(index=False))
