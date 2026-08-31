# E0043/E0044: embryo-safe candidate-domain hard-negative fine-tuning.
import copy as _crop_copy
import heapq as _crop_heapq

_hn_loss_mode = os.environ.get("BIOHUB_HN_LOSS", "bce").strip().lower()
_hn_top_k = int(os.environ.get("BIOHUB_HN_TOP_K", "1024"))
_hn_random_k = int(os.environ.get("BIOHUB_HN_RANDOM_K", "512"))
_hn_epochs = int(os.environ.get("BIOHUB_HN_EPOCHS", "12"))
_hn_lr = float(os.environ.get("BIOHUB_HN_LR", "0.0002"))
if _hn_loss_mode not in {"bce", "focal"}:
    raise ValueError(_hn_loss_mode)

_hn_join_keys = ["stem", "fork_id", "daughter1_id", "daughter2_id", "label"]
_hn_parent_meta = _parents.merge(
    _transfer[_hn_join_keys], on=_hn_join_keys, how="inner", validate="one_to_one"
)
if len(_hn_parent_meta) != len(_transfer):
    raise RuntimeError("Candidate metadata join was not one-to-one")


def _hn_batches(_rows, _batch_size=64):
    """Yield candidate crops without retaining the dense proposal pool in RAM."""
    for _stem, _stem_rows in _rows.groupby("stem", sort=True):
        _feature_graph = _official_graph_from_processed(*official_graph_inputs[_stem])
        _node_rows = {
            int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r
            for r in _feature_graph.node_attrs().iter_rows(named=True)
        }
        _array = zarr.open(TRAIN_DIR / f"{_stem}.zarr" / "0", mode="r")
        for _t, _time_rows in _stem_rows.groupby("t", sort=True):
            _t = int(_t)
            _frames = [
                np.asarray(_array[_ft])
                for _ft in (max(0, _t-1), _t, min(int(_array.shape[0])-1, _t+1))
            ]
            _crops, _meta = [], []
            for _row in _time_rows.itertuples(index=False):
                _fork = int(_row.fork_id)
                if _fork not in _node_rows:
                    continue
                _r = _node_rows[_fork]
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
                _stack = np.clip(
                    (_stack-_lo)/max(float(_hi-_lo), 1.0), -0.5, 2.0
                ).astype(np.float16)
                _crops.append(_stack)
                _meta.append(_row)
                if len(_crops) == _batch_size:
                    yield np.stack(_crops), _meta
                    _crops, _meta = [], []
            if _crops:
                yield np.stack(_crops), _meta


def _hn_focal_loss(_logits, _targets, _gamma=2.0):
    _base = nn.functional.binary_cross_entropy_with_logits(_logits, _targets, reduction="none")
    _prob = torch.sigmoid(_logits)
    _pt = torch.where(_targets > 0.5, _prob, 1.0-_prob)
    return (((1.0-_pt) ** _gamma) * _base).mean()


_hn_rng = np.random.default_rng(_crop_seed + (1 if _hn_loss_mode == "focal" else 0))
_hn_rows, _hn_fold_rows = [], []
for _heldout in sorted(_hn_parent_meta["embryo"].unique()):
    _train_meta = _hn_parent_meta[_hn_parent_meta["embryo"] != _heldout].copy()
    _test_meta = _hn_parent_meta[_hn_parent_meta["embryo"] == _heldout].copy()
    _base_model = _crop_models[_heldout].to(_crop_device).eval()

    # Mine hard negatives with the fold's legal base model. It has never seen the
    # held-out embryo and scores only the opposite embryo for training selection.
    _top_heap, _positive_crops = [], []
    _random_negative_crops = []
    _negative_seen = 0
    with torch.no_grad():
        for _crops, _meta in _hn_batches(_train_meta):
            _scores = torch.sigmoid(
                _base_model(torch.from_numpy(_crops.astype(np.float32)).to(_crop_device))
            ).cpu().numpy()
            for _crop, _row, _score in zip(_crops, _meta, _scores):
                if int(_row.label) == 1:
                    _positive_crops.append(_crop.copy())
                    continue
                _negative_seen += 1
                _item = (float(_score), _negative_seen, _crop.copy())
                if len(_top_heap) < _hn_top_k:
                    _crop_heapq.heappush(_top_heap, _item)
                elif _score > _top_heap[0][0]:
                    _crop_heapq.heapreplace(_top_heap, _item)
                # Reservoir sample adds background diversity without label leakage.
                if len(_random_negative_crops) < _hn_random_k:
                    _random_negative_crops.append(_crop.copy())
                else:
                    _replace = int(_hn_rng.integers(0, _negative_seen))
                    if _replace < _hn_random_k:
                        _random_negative_crops[_replace] = _crop.copy()

    _hard_negative_crops = [x[2] for x in sorted(_top_heap, reverse=True)]
    if not _positive_crops or not _hard_negative_crops:
        raise RuntimeError(f"Fold {_heldout} lacks candidate positives or negatives")

    # Retain annotated training positives as a stabilizer while adapting to the
    # detector-centered candidate distribution.
    _gt_positive_crops = [
        x["crop"] for x in _crop_examples
        if x["embryo"] != _heldout and int(x["label"]) == 1
    ]
    _positive_pool = _positive_crops + _gt_positive_crops
    _negative_pool = _hard_negative_crops + _random_negative_crops
    _model = _crop_copy.deepcopy(_base_model).to(_crop_device).train()
    _optimizer = torch.optim.AdamW(_model.parameters(), lr=_hn_lr, weight_decay=2e-3)
    _steps = max(32, int(np.ceil(len(_negative_pool)/24)))
    for _epoch in range(_hn_epochs):
        for _ in range(_steps):
            _pos_idx = _hn_rng.choice(len(_positive_pool), 8, replace=True)
            _neg_idx = _hn_rng.choice(len(_negative_pool), 24, replace=True)
            _x = np.stack(
                [_positive_pool[int(i)] for i in _pos_idx]
                + [_negative_pool[int(i)] for i in _neg_idx]
            ).astype(np.float32)
            _y = np.concatenate([np.ones(8), np.zeros(24)]).astype(np.float32)
            _order = _hn_rng.permutation(len(_y))
            _x, _y = _x[_order], _y[_order]
            _optimizer.zero_grad(set_to_none=True)
            _logits = _model(torch.from_numpy(_x).to(_crop_device))
            _targets = torch.from_numpy(_y).to(_crop_device)
            _loss = (
                _hn_focal_loss(_logits, _targets)
                if _hn_loss_mode == "focal"
                else nn.functional.binary_cross_entropy_with_logits(_logits, _targets)
            )
            _loss.backward()
            _optimizer.step()

    _model.eval()
    _fold_scores, _fold_truth = [], []
    with torch.no_grad():
        for _crops, _meta in _hn_batches(_test_meta):
            _scores = torch.sigmoid(
                _model(torch.from_numpy(_crops.astype(np.float32)).to(_crop_device))
            ).cpu().numpy()
            for _row, _score in zip(_meta, _scores):
                _truth = int(_row.label)
                _fold_scores.append(float(_score))
                _fold_truth.append(_truth)
                _hn_rows.append({
                    "stem": _row.stem, "embryo": _heldout,
                    "fork_id": int(_row.fork_id), "label": _truth,
                    "score": float(_score),
                })
    _hn_fold_rows.append({
        "heldout_embryo": _heldout,
        "train_candidates": len(_train_meta),
        "train_candidate_positives": len(_positive_crops),
        "hard_negatives": len(_hard_negative_crops),
        "random_negatives": len(_random_negative_crops),
        "test_candidates": len(_fold_truth),
        "test_positives": int(np.sum(_fold_truth)),
        "average_precision": float(_crop_ap(_fold_truth, _fold_scores)),
        "roc_auc": float(_crop_auc(_fold_truth, _fold_scores)),
    })
    _base_model.cpu()
    _model.cpu()
    if _crop_device.type == "cuda":
        torch.cuda.empty_cache()

_hn_oof = pd.DataFrame(_hn_rows)
_hn_folds = pd.DataFrame(_hn_fold_rows)
_hn_threshold_rows = []
for _quantile in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]:
    _threshold = float(_hn_oof["score"].quantile(_quantile))
    _accepted = _hn_oof[_hn_oof["score"] >= _threshold]
    _tp = int(_accepted["label"].sum())
    _hn_threshold_rows.append({
        "quantile": _quantile, "threshold": _threshold, "accepted": len(_accepted),
        "tp": _tp, "fp": len(_accepted)-_tp,
        "precision": float(_tp/max(len(_accepted), 1)),
        "recall": float(_tp/max(int(_hn_oof["label"].sum()), 1)),
    })
_hn_thresholds = pd.DataFrame(_hn_threshold_rows)
_hn_summary = {
    "loss": _hn_loss_mode,
    "top_hard_negatives_per_fold": _hn_top_k,
    "random_negatives_per_fold": _hn_random_k,
    "epochs": _hn_epochs,
    "candidate_rows": len(_hn_oof),
    "positive_rows": int(_hn_oof["label"].sum()),
    "average_precision": float(_crop_ap(_hn_oof["label"], _hn_oof["score"])),
    "roc_auc": float(_crop_auc(_hn_oof["label"], _hn_oof["score"])),
}
_hn_oof.to_csv(WORKING_DIR / "candidate_hard_negative_oof.csv", index=False)
_hn_folds.to_csv(WORKING_DIR / "candidate_hard_negative_folds.csv", index=False)
_hn_thresholds.to_csv(WORKING_DIR / "candidate_hard_negative_thresholds.csv", index=False)
with (WORKING_DIR / "candidate_hard_negative_summary.json").open("w") as _fh:
    json.dump(_hn_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_hn_summary, indent=2, sort_keys=True))
print(_hn_folds.to_string(index=False))
print(_hn_thresholds.to_string(index=False))
