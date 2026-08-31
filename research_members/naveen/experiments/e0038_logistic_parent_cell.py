# E0038: standardized linear parent-event model for cross-embryo calibration.
from sklearn.impute import SimpleImputer as _ParentImputer
from sklearn.linear_model import LogisticRegression as _ParentLogistic
from sklearn.pipeline import make_pipeline as _parent_pipeline
from sklearn.preprocessing import StandardScaler as _ParentScaler

_lr_fold_rows, _lr_oof_parts = [], []
for _heldout in sorted(_event["embryo"].unique()):
    _train = _event[_event["embryo"] != _heldout].copy()
    _test = _event[_event["embryo"] == _heldout].copy()
    _model = _parent_pipeline(
        _ParentImputer(strategy="median"),
        _ParentScaler(),
        _ParentLogistic(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=20260830,
        ),
    )
    _model.fit(_train[_features], _train["label"])
    _test["event_score_logistic"] = _model.predict_proba(_test[_features])[:, 1]
    _lr_fold_rows.append({
        "heldout_embryo": _heldout,
        "train_rows": len(_train),
        "train_positives": int(_train["label"].sum()),
        "test_rows": len(_test),
        "test_positives": int(_test["label"].sum()),
        "average_precision": float(_parent_ap(_test["label"], _test["event_score_logistic"])),
        "roc_auc": float(_parent_auc(_test["label"], _test["event_score_logistic"])),
        "positive_score_min": float(_test.loc[_test["label"] == 1, "event_score_logistic"].min()),
        "positive_score_median": float(_test.loc[_test["label"] == 1, "event_score_logistic"].median()),
    })
    _lr_oof_parts.append(_test)

_lr_oof = pd.concat(_lr_oof_parts, ignore_index=True)
_lr_folds = pd.DataFrame(_lr_fold_rows)
_lr_threshold_rows = []
for _threshold in sorted(set(np.quantile(
    _lr_oof["event_score_logistic"], [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
).tolist())):
    _accepted = _lr_oof[_lr_oof["event_score_logistic"] >= _threshold]
    _tp = int(_accepted["label"].sum())
    _lr_threshold_rows.append({
        "threshold": float(_threshold),
        "accepted": len(_accepted),
        "tp": _tp,
        "fp": len(_accepted)-_tp,
        "precision": float(_tp/len(_accepted)) if len(_accepted) else 0.0,
        "recall": float(_tp/max(int(_lr_oof["label"].sum()), 1)),
    })
_lr_thresholds = pd.DataFrame(_lr_threshold_rows)
_lr_summary = {
    "parent_candidates": int(len(_event)),
    "positive_parents": int(_event["label"].sum()),
    "mean_average_precision": float(_lr_folds["average_precision"].mean()),
    "mean_roc_auc": float(_lr_folds["roc_auc"].mean()),
}
_lr_oof.to_csv(WORKING_DIR / "parent_event_logistic_oof.csv", index=False)
_lr_folds.to_csv(WORKING_DIR / "parent_event_logistic_folds.csv", index=False)
_lr_thresholds.to_csv(WORKING_DIR / "parent_event_logistic_thresholds.csv", index=False)
with (WORKING_DIR / "parent_event_logistic_summary.json").open("w") as _fh:
    json.dump(_lr_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_lr_summary, indent=2, sort_keys=True))
print(_lr_folds.to_string(index=False))
print(_lr_thresholds.to_string(index=False))
