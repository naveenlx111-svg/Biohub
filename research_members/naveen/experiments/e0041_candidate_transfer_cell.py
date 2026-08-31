# E0041: transfer the held-out crop models to detected parent proposals.
_transfer_rows = []
for _stem, _stem_parents in _parents.groupby("stem", sort=True):
    _embryo = _stem.split("_", 1)[0]
    if _embryo not in _crop_models:
        continue
    _model = _crop_models[_embryo].to(_crop_device).eval()
    _feature_graph = _official_graph_from_processed(*official_graph_inputs[_stem])
    _nodes = {
        int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r
        for r in _feature_graph.node_attrs().iter_rows(named=True)
    }
    _array = zarr.open(TRAIN_DIR / f"{_stem}.zarr" / "0", mode="r")
    for _t, _time_rows in _stem_parents.groupby("t", sort=True):
        _t = int(_t)
        _frame_times = (max(0, _t-1), _t, min(int(_array.shape[0])-1, _t+1))
        _frames = [np.asarray(_array[ft]) for ft in _frame_times]
        _batch_crops, _batch_meta = [], []
        for _row in _time_rows.itertuples(index=False):
            _fork = int(_row.fork_id)
            if _fork not in _nodes:
                continue
            _r = _nodes[_fork]
            _z, _y, _x = (int(round(float(_r[k]))) for k in ("z", "y", "x"))
            _patches = []
            for _frame in _frames:
                _rz, _ryx = 4, 16
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
            _stack = np.clip((_stack-_lo)/max(float(_hi-_lo), 1.0), -0.5, 2.0).astype(np.float32)
            _batch_crops.append(_stack)
            _batch_meta.append(_row)
        with torch.no_grad():
            for _start in range(0, len(_batch_crops), 64):
                _x = torch.from_numpy(np.stack(_batch_crops[_start:_start+64])).to(_crop_device)
                _scores = torch.sigmoid(_model(_x)).cpu().numpy()
                for _row, _score in zip(_batch_meta[_start:_start+64], _scores):
                    _transfer_rows.append({
                        "stem": _stem,
                        "embryo": _embryo,
                        "fork_id": int(_row.fork_id),
                        "daughter1_id": int(_row.daughter1_id),
                        "daughter2_id": int(_row.daughter2_id),
                        "label": int(_row.label),
                        "crop_score": float(_score),
                    })
    _model.cpu()
    if _crop_device.type == "cuda":
        torch.cuda.empty_cache()

_transfer = pd.DataFrame(_transfer_rows)
_transfer_threshold_rows = []
for _threshold in sorted(set(np.quantile(
    _transfer["crop_score"], [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]
).tolist())):
    _accepted = _transfer[_transfer["crop_score"] >= _threshold]
    _tp = int(_accepted["label"].sum())
    _transfer_threshold_rows.append({
        "threshold": float(_threshold), "accepted": len(_accepted),
        "tp": _tp, "fp": len(_accepted)-_tp,
        "precision": float(_tp/len(_accepted)) if len(_accepted) else 0.0,
        "recall": float(_tp/max(int(_transfer["label"].sum()), 1)),
    })
_transfer_thresholds = pd.DataFrame(_transfer_threshold_rows)
_transfer_summary = {
    "detected_parent_candidates": int(len(_transfer)),
    "positive_parent_pairs": int(_transfer["label"].sum()),
    "average_precision": float(_crop_ap(_transfer["label"], _transfer["crop_score"])),
    "roc_auc": float(_crop_auc(_transfer["label"], _transfer["crop_score"])),
}
_transfer.to_csv(WORKING_DIR / "candidate_crop_transfer_oof.csv", index=False)
_transfer_thresholds.to_csv(WORKING_DIR / "candidate_crop_transfer_thresholds.csv", index=False)
with (WORKING_DIR / "candidate_crop_transfer_summary.json").open("w") as _fh:
    json.dump(_transfer_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_transfer_summary, indent=2, sort_keys=True))
print(_transfer_thresholds.to_string(index=False))
