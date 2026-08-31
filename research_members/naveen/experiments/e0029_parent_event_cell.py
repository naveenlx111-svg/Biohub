# Same-run parent-event image and temporal evidence on held-out top-pair proposals.
from sklearn.ensemble import HistGradientBoostingClassifier as _ParentHGB
from sklearn.metrics import average_precision_score as _parent_ap, roc_auc_score as _parent_auc
import zarr

_pair_features = [
    "d1_um", "d2_um", "distance_sum_um", "distance_asymmetry_um",
    "sister_um", "midpoint_um", "daughter_cosine", "fork_outdegree",
    "daughter1_indegree", "daughter2_indegree", "edge1_exists", "edge2_exists",
]
_ranked = _candidate_df.copy()
_ranked["candidate_index"] = np.arange(len(_ranked), dtype=np.int64)
_ranked = _ranked[
    (_ranked["distance_sum_um"] <= 14.0)
    & (_ranked["sister_um"] >= 4.0)
    & (_ranked["midpoint_um"] <= 5.0)
    & (_ranked["daughter_cosine"] <= -0.30)
].copy()
if int(_ranked["label"].sum()) != int(_candidate_df["label"].sum()):
    raise RuntimeError("Same-run hard-negative gate discarded a positive")

_pair_oof_parts = []
for _heldout in sorted(_ranked["embryo"].unique()):
    _pair_train = _ranked[_ranked["embryo"] != _heldout].copy()
    _pair_test = _ranked[_ranked["embryo"] == _heldout].copy()
    _pair_pos = int(_pair_train["label"].sum())
    _pair_neg = len(_pair_train)-_pair_pos
    _pair_weights = np.ones(len(_pair_train), dtype=float)
    _pair_weights[_pair_train["label"].to_numpy() == 1] = _pair_neg/max(_pair_pos, 1)
    _pair_model = _ParentHGB(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=2.0, random_state=20260830,
    )
    _pair_model.fit(
        _pair_train[_pair_features], _pair_train["label"], sample_weight=_pair_weights
    )
    _pair_test["score"] = _pair_model.predict_proba(_pair_test[_pair_features])[:, 1]
    _pair_test["rank_within_fork"] = _pair_test.groupby(["stem", "fork_id"])["score"].rank(
        method="first", ascending=False
    )
    _pair_oof_parts.append(_pair_test)
_ranked = pd.concat(_pair_oof_parts, ignore_index=True)
_parents = _ranked[_ranked["rank_within_fork"] == 1].copy()
_parents = _parents.sort_values(["stem", "t", "fork_id"]).reset_index(drop=True)
print("Parent-event candidates:", len(_parents), "positives:", int(_parents["label"].sum()))


def _patch_stats(_frame, _zyx, _rz=2, _ryx=6):
    _z, _y, _x = (int(round(float(v))) for v in _zyx)
    _z0, _z1 = max(0, _z-_rz), min(_frame.shape[0], _z+_rz+1)
    _y0, _y1 = max(0, _y-_ryx), min(_frame.shape[1], _y+_ryx+1)
    _x0, _x1 = max(0, _x-_ryx), min(_frame.shape[2], _x+_ryx+1)
    _p = np.asarray(_frame[_z0:_z1, _y0:_y1, _x0:_x1], dtype=np.float32)
    if not _p.size:
        return [0.0] * 6
    _lo, _hi = np.percentile(_p, [10, 99])
    _scale = max(float(_hi-_lo), 1.0)
    _q = np.clip((_p-_lo)/_scale, -1.0, 2.0)
    _cz0, _cz1 = max(0, _z-_z0-1), min(_q.shape[0], _z-_z0+2)
    _cy0, _cy1 = max(0, _y-_y0-2), min(_q.shape[1], _y-_y0+3)
    _cx0, _cx1 = max(0, _x-_x0-2), min(_q.shape[2], _x-_x0+3)
    _inner = _q[_cz0:_cz1, _cy0:_cy1, _cx0:_cx1]
    return [float(_q.mean()), float(_q.std()), float(_q.max()),
            float(np.percentile(_q, 90)), float(_inner.mean()),
            float(_inner.mean()-_q.mean())]


_image_rows = []
_parent_event_original_test_dir = TEST_DIR
TEST_DIR = TRAIN_DIR
for _stem, _stem_rows in _parents.groupby("stem", sort=True):
    _feature_graph = _official_graph_from_processed(*official_graph_inputs[_stem])
    _nodes = {
        int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r
        for r in _feature_graph.node_attrs().iter_rows(named=True)
    }
    _feature_incoming, _feature_outgoing = {}, {}
    for _edge_row in _feature_graph.edge_attrs().iter_rows(named=True):
        _es = int(_edge_row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE])
        _et = int(_edge_row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        _feature_outgoing.setdefault(_es, set()).add(_et)
        _feature_incoming.setdefault(_et, set()).add(_es)

    def _chain_span(_start, _adjacency, _limit=12):
        _span, _current, _seen = 0, int(_start), {int(_start)}
        while _span < _limit:
            _next = list(_adjacency.get(_current, ()))
            if len(_next) != 1 or _next[0] in _seen:
                break
            _current = int(_next[0])
            _seen.add(_current)
            _span += 1
        return _span
    _array = zarr.open(TRAIN_DIR / f"{_stem}.zarr" / "0", mode="r")
    _frame_cache = {}
    _heatmap_cache = {}
    _dc_frame_cache = {}
    for _row in _stem_rows.itertuples(index=False):
        _fork = int(_row.fork_id)
        _d1, _d2 = int(_row.daughter1_id), int(_row.daughter2_id)
        if _fork not in _nodes or _d1 not in _nodes or _d2 not in _nodes:
            continue
        _t = int(_nodes[_fork]["t"])
        for _ft in (_t-1, _t, _t+1):
            if 0 <= _ft < _array.shape[0] and _ft not in _frame_cache:
                _frame_cache[_ft] = np.asarray(_array[_ft])
        # Keep only a rolling raw-frame cache because rows are time sorted.
        for _old in list(_frame_cache):
            if _old < _t-1:
                _frame_cache.pop(_old, None)
        _p = tuple(float(_nodes[_fork][k]) for k in ("z", "y", "x"))
        _pstats = _patch_stats(_frame_cache[_t], _p)
        _prev = _patch_stats(_frame_cache[_t-1], _p) if _t-1 in _frame_cache else _pstats
        _next = _patch_stats(_frame_cache[_t+1], _p) if _t+1 in _frame_cache else _pstats
        _d1p = tuple(float(_nodes[_d1][k]) for k in ("z", "y", "x"))
        _d2p = tuple(float(_nodes[_d2][k]) for k in ("z", "y", "x"))
        _d1stats = _patch_stats(_frame_cache[_t+1], _d1p)
        _d2stats = _patch_stats(_frame_cache[_t+1], _d2p)
        _parent_history = _chain_span(_fork, _feature_incoming)
        _d1_future = _chain_span(_d1, _feature_outgoing)
        _d2_future = _chain_span(_d2, _feature_outgoing)
        _predecessors = list(_feature_incoming.get(_fork, ()))
        _parent_speed = 0.0
        if len(_predecessors) == 1 and _predecessors[0] in _nodes:
            _prev_row = _nodes[_predecessors[0]]
            _prev_pos = np.asarray([_prev_row[k] for k in ("z", "y", "x")], dtype=float)
            _parent_speed = float(np.linalg.norm((np.asarray(_p)-_prev_pos)*np.asarray(VOXEL_SCALE_UM)))
        _dc = deepcenter_score_point(
            _stem, _t, _p, globals().get("DEEPCENTER_VETO_DETECTOR"),
            _dc_frame_cache, _heatmap_cache,
        )
        _image_rows.append({
            "candidate_index": int(_row.candidate_index),
            "parent_mean": _pstats[0], "parent_std": _pstats[1],
            "parent_max": _pstats[2], "parent_p90": _pstats[3],
            "parent_inner": _pstats[4], "parent_contrast": _pstats[5],
            "parent_mean_delta_prev": _pstats[0]-_prev[0],
            "parent_mean_delta_next": _pstats[0]-_next[0],
            "parent_std_delta_prev": _pstats[1]-_prev[1],
            "parent_std_delta_next": _pstats[1]-_next[1],
            "daughter_mean_average": (_d1stats[0]+_d2stats[0])/2.0,
            "daughter_mean_asymmetry": abs(_d1stats[0]-_d2stats[0]),
            "parent_minus_daughters": _pstats[0]-(_d1stats[0]+_d2stats[0])/2.0,
            "deepcenter_score": float(_dc) if _dc is not None else -1.0,
            "parent_history_span": _parent_history,
            "daughter_future_min": min(_d1_future, _d2_future),
            "daughter_future_max": max(_d1_future, _d2_future),
            "daughter_future_asymmetry": abs(_d1_future-_d2_future),
            "parent_speed_um": _parent_speed,
        })
TEST_DIR = _parent_event_original_test_dir

_image = pd.DataFrame(_image_rows)
_event = _parents.merge(_image, on="candidate_index", how="inner", validate="one_to_one")
_image_features = [c for c in _image.columns if c != "candidate_index"]
_geometry_features = [
    "d1_um", "d2_um", "distance_sum_um", "distance_asymmetry_um",
    "sister_um", "midpoint_um", "daughter_cosine", "fork_outdegree",
    "daughter1_indegree", "daughter2_indegree", "edge1_exists", "edge2_exists",
    "score",
]
_features = _geometry_features + _image_features

_fold_rows, _oof_parts = [], []
for _heldout in sorted(_event["embryo"].unique()):
    _train = _event[_event["embryo"] != _heldout].copy()
    _test = _event[_event["embryo"] == _heldout].copy()
    _pos = int(_train["label"].sum())
    _neg = len(_train)-_pos
    _weights = np.ones(len(_train), dtype=float)
    _weights[_train["label"].to_numpy() == 1] = _neg / max(_pos, 1)
    _model = _ParentHGB(
        learning_rate=0.04, max_iter=300, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=3.0, random_state=20260830,
    )
    _model.fit(_train[_features], _train["label"], sample_weight=_weights)
    _test["event_score"] = _model.predict_proba(_test[_features])[:, 1]
    _fold_rows.append({
        "heldout_embryo": _heldout,
        "train_rows": len(_train), "train_positives": _pos,
        "test_rows": len(_test), "test_positives": int(_test["label"].sum()),
        "average_precision": float(_parent_ap(_test["label"], _test["event_score"])),
        "roc_auc": float(_parent_auc(_test["label"], _test["event_score"])),
        "positive_score_min": float(_test.loc[_test["label"] == 1, "event_score"].min()),
        "positive_score_median": float(_test.loc[_test["label"] == 1, "event_score"].median()),
    })
    _oof_parts.append(_test)

_oof = pd.concat(_oof_parts, ignore_index=True)
_folds = pd.DataFrame(_fold_rows)
_threshold_rows = []
for _threshold in sorted(set(np.quantile(_oof["event_score"], [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]).tolist())):
    _accepted = _oof[_oof["event_score"] >= _threshold]
    _tp = int(_accepted["label"].sum())
    _threshold_rows.append({
        "threshold": float(_threshold), "accepted": len(_accepted),
        "tp": _tp, "fp": len(_accepted)-_tp,
        "precision": float(_tp/len(_accepted)) if len(_accepted) else 0.0,
        "recall": float(_tp/max(int(_oof["label"].sum()), 1)),
    })
_thresholds = pd.DataFrame(_threshold_rows)
_summary = {
    "parent_candidates": int(len(_event)),
    "positive_parents": int(_event["label"].sum()),
    "mean_average_precision": float(_folds["average_precision"].mean()),
    "mean_roc_auc": float(_folds["roc_auc"].mean()),
}
_event.to_csv(WORKING_DIR / "parent_event_features.csv", index=False)
_oof.to_csv(WORKING_DIR / "parent_event_oof.csv", index=False)
_folds.to_csv(WORKING_DIR / "parent_event_folds.csv", index=False)
_thresholds.to_csv(WORKING_DIR / "parent_event_thresholds.csv", index=False)
with (WORKING_DIR / "parent_event_summary.json").open("w") as _fh:
    json.dump(_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_summary, indent=2, sort_keys=True))
print(_folds.to_string(index=False))
print(_thresholds.to_string(index=False))
