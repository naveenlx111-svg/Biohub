# E0030: independent ExtraTrees parent-event ranker on E0029 features.
from sklearn.ensemble import ExtraTreesClassifier as _ParentExtraTrees

_et_fold_rows, _et_oof_parts = [], []
for _heldout in sorted(_event["embryo"].unique()):
    _train = _event[_event["embryo"] != _heldout].copy()
    _test = _event[_event["embryo"] == _heldout].copy()
    _pos = int(_train["label"].sum())
    _neg = len(_train)-_pos
    _weights = np.ones(len(_train), dtype=float)
    _weights[_train["label"].to_numpy() == 1] = _neg / max(_pos, 1)
    _model = _ParentExtraTrees(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=20260830,
    )
    _model.fit(_train[_features], _train["label"], sample_weight=_weights)
    _test["event_score_extra_trees"] = _model.predict_proba(_test[_features])[:, 1]
    _et_fold_rows.append({
        "heldout_embryo": _heldout,
        "train_rows": len(_train), "train_positives": _pos,
        "test_rows": len(_test), "test_positives": int(_test["label"].sum()),
        "average_precision": float(_parent_ap(_test["label"], _test["event_score_extra_trees"])),
        "roc_auc": float(_parent_auc(_test["label"], _test["event_score_extra_trees"])),
        "positive_score_min": float(_test.loc[_test["label"] == 1, "event_score_extra_trees"].min()),
        "positive_score_median": float(_test.loc[_test["label"] == 1, "event_score_extra_trees"].median()),
    })
    _et_oof_parts.append(_test)

_et_oof = pd.concat(_et_oof_parts, ignore_index=True)
_et_folds = pd.DataFrame(_et_fold_rows)
_et_threshold_rows = []
for _threshold in sorted(set(np.quantile(
    _et_oof["event_score_extra_trees"], [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
).tolist())):
    _accepted = _et_oof[_et_oof["event_score_extra_trees"] >= _threshold]
    _tp = int(_accepted["label"].sum())
    _et_threshold_rows.append({
        "threshold": float(_threshold), "accepted": len(_accepted),
        "tp": _tp, "fp": len(_accepted)-_tp,
        "precision": float(_tp/len(_accepted)) if len(_accepted) else 0.0,
        "recall": float(_tp/max(int(_et_oof["label"].sum()), 1)),
    })
_et_thresholds = pd.DataFrame(_et_threshold_rows)
_et_summary = {
    "parent_candidates": int(len(_event)),
    "positive_parents": int(_event["label"].sum()),
    "mean_average_precision": float(_et_folds["average_precision"].mean()),
    "mean_roc_auc": float(_et_folds["roc_auc"].mean()),
}
_et_oof.to_csv(WORKING_DIR / "parent_event_extra_trees_oof.csv", index=False)
_et_folds.to_csv(WORKING_DIR / "parent_event_extra_trees_folds.csv", index=False)
_et_thresholds.to_csv(WORKING_DIR / "parent_event_extra_trees_thresholds.csv", index=False)
with (WORKING_DIR / "parent_event_extra_trees_summary.json").open("w") as _fh:
    json.dump(_et_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_et_summary, indent=2, sort_keys=True))
print(_et_folds.to_string(index=False))
print(_et_thresholds.to_string(index=False))
