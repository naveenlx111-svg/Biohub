# E0046: physical-neighborhood pooling around detected parent proposals.
_offsets_vox = [
    (0, 0, 0),
    (-2, 0, 0), (2, 0, 0),
    (0, -8, 0), (0, 8, 0),
    (0, 0, -8), (0, 0, 8),
]


def _offset_crop_from_frames(_frames, _zyx, _rz=4, _ryx=16):
    _z, _y, _x = (int(round(float(v))) for v in _zyx)
    _patches = []
    for _frame in _frames:
        _z0, _z1 = max(0, _z-_rz), min(_frame.shape[0], _z+_rz+1)
        _y0, _y1 = max(0, _y-_ryx), min(_frame.shape[1], _y+_ryx+1)
        _x0, _x1 = max(0, _x-_ryx), min(_frame.shape[2], _x+_ryx+1)
        _patch = _frame[_z0:_z1, _y0:_y1, _x0:_x1].astype(np.float32, copy=False)
        _pad = (
            (max(0, _rz-(_z-_z0)), max(0, _rz-(_z1-_z-1))),
            (max(0, _ryx-(_y-_y0)), max(0, _ryx-(_y1-_y-1))),
            (max(0, _ryx-(_x-_x0)), max(0, _ryx-(_x1-_x-1))),
        )
        _patches.append(np.pad(_patch, _pad, mode="edge"))
    _stack = np.stack(_patches)
    _lo, _hi = np.percentile(_stack, [10.0, 99.5])
    return np.clip(
        (_stack-_lo)/max(float(_hi-_lo), 1.0), -0.5, 2.0
    ).astype(np.float32)


_offset_rows = []
for _stem, _stem_parents in _parents.groupby("stem", sort=True):
    _embryo = _stem.split("_", 1)[0]
    if _embryo not in _crop_models:
        continue
    _model = _crop_models[_embryo].to(_crop_device).eval()
    _feature_graph = _official_graph_from_processed(*official_graph_inputs[_stem])
    _node_rows = {
        int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r
        for r in _feature_graph.node_attrs().iter_rows(named=True)
    }
    _array = zarr.open(TRAIN_DIR / f"{_stem}.zarr" / "0", mode="r")
    for _t, _time_rows in _stem_parents.groupby("t", sort=True):
        _t = int(_t)
        _frames = [
            np.asarray(_array[_ft])
            for _ft in (max(0, _t-1), _t, min(int(_array.shape[0])-1, _t+1))
        ]
        _pending_crops, _pending_rows = [], []

        def _flush_offset_batch():
            if not _pending_rows:
                return
            with torch.no_grad():
                _scores = torch.sigmoid(_model(
                    torch.from_numpy(np.stack(_pending_crops)).to(_crop_device)
                )).cpu().numpy().reshape(len(_pending_rows), len(_offsets_vox))
            for _row, _row_scores in zip(_pending_rows, _scores):
                _ordered = np.sort(_row_scores)
                _offset_rows.append({
                    "stem": _stem,
                    "embryo": _embryo,
                    "fork_id": int(_row.fork_id),
                    "daughter1_id": int(_row.daughter1_id),
                    "daughter2_id": int(_row.daughter2_id),
                    "label": int(_row.label),
                    "center_score": float(_row_scores[0]),
                    "offset_max_score": float(_ordered[-1]),
                    "offset_top2_mean_score": float(_ordered[-2:].mean()),
                    "offset_mean_score": float(_row_scores.mean()),
                })
            _pending_crops.clear()
            _pending_rows.clear()

        for _row in _time_rows.itertuples(index=False):
            _fork = int(_row.fork_id)
            if _fork not in _node_rows:
                continue
            _r = _node_rows[_fork]
            _center = tuple(float(_r[k]) for k in ("z", "y", "x"))
            for _dz, _dy, _dx in _offsets_vox:
                _pending_crops.append(_offset_crop_from_frames(
                    _frames,
                    (_center[0]+_dz, _center[1]+_dy, _center[2]+_dx),
                ))
            _pending_rows.append(_row)
            if len(_pending_rows) >= 32:
                _flush_offset_batch()
        _flush_offset_batch()
    _model.cpu()
    if _crop_device.type == "cuda":
        torch.cuda.empty_cache()

_offset_oof = pd.DataFrame(_offset_rows)
_offset_metric_rows, _offset_threshold_rows = [], []
for _score_name in [
    "center_score", "offset_max_score", "offset_top2_mean_score", "offset_mean_score"
]:
    _offset_metric_rows.append({
        "score": _score_name,
        "average_precision": float(_crop_ap(_offset_oof["label"], _offset_oof[_score_name])),
        "roc_auc": float(_crop_auc(_offset_oof["label"], _offset_oof[_score_name])),
    })
    for _quantile in [0.99, 0.995, 0.999, 0.9995, 0.9999]:
        _threshold = float(_offset_oof[_score_name].quantile(_quantile))
        _accepted = _offset_oof[_offset_oof[_score_name] >= _threshold]
        _tp = int(_accepted["label"].sum())
        _offset_threshold_rows.append({
            "score": _score_name, "quantile": _quantile,
            "threshold": _threshold, "accepted": len(_accepted),
            "tp": _tp, "fp": len(_accepted)-_tp,
            "precision": float(_tp/max(len(_accepted), 1)),
            "recall": float(_tp/max(int(_offset_oof["label"].sum()), 1)),
        })
_offset_metrics = pd.DataFrame(_offset_metric_rows)
_offset_thresholds = pd.DataFrame(_offset_threshold_rows)
_offset_summary = {
    "candidates": len(_offset_oof),
    "positives": int(_offset_oof["label"].sum()),
    "offsets_vox": _offsets_vox,
    "offsets_um": [[dz*1.625, dy*0.40625, dx*0.40625] for dz, dy, dx in _offsets_vox],
    "metrics": _offset_metric_rows,
}
_offset_oof.to_csv(WORKING_DIR / "offset_pool_transfer_oof.csv", index=False)
_offset_metrics.to_csv(WORKING_DIR / "offset_pool_transfer_metrics.csv", index=False)
_offset_thresholds.to_csv(WORKING_DIR / "offset_pool_transfer_thresholds.csv", index=False)
with (WORKING_DIR / "offset_pool_transfer_summary.json").open("w") as _fh:
    json.dump(_offset_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_offset_summary, indent=2, sort_keys=True))
print(_offset_thresholds.to_string(index=False))
