# Temporal 3D crop classifier for division-parent identification.
import random as _crop_random
import zarr
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score as _crop_ap, roc_auc_score as _crop_auc

_crop_mode = os.environ.get("BIOHUB_TEMPORAL_CROP_MODE", "raw").strip()
_crop_translation_aug = os.environ.get("BIOHUB_CROP_TRANSLATION_AUG", "0") != "0"
if _crop_mode not in {"raw", "raw_diff"}:
    raise ValueError(_crop_mode)
_crop_seed = 20260830
_crop_rng = np.random.default_rng(_crop_seed)
_crop_examples = []


def _extract_temporal_crop(_array, _t, _zyx, _rz=4, _ryx=16):
    _z, _y, _x = (int(round(float(v))) for v in _zyx)
    _frames = []
    for _ft in (max(0, _t-1), _t, min(int(_array.shape[0])-1, _t+1)):
        _frame = np.asarray(_array[_ft])
        _z0, _z1 = max(0, _z-_rz), min(_frame.shape[0], _z+_rz+1)
        _y0, _y1 = max(0, _y-_ryx), min(_frame.shape[1], _y+_ryx+1)
        _x0, _x1 = max(0, _x-_ryx), min(_frame.shape[2], _x+_ryx+1)
        _patch = _frame[_z0:_z1, _y0:_y1, _x0:_x1].astype(np.float32, copy=False)
        _pad = (
            (max(0, _rz-(_z-_z0)), max(0, _rz-(_z1-_z-1))),
            (max(0, _ryx-(_y-_y0)), max(0, _ryx-(_y1-_y-1))),
            (max(0, _ryx-(_x-_x0)), max(0, _ryx-(_x1-_x-1))),
        )
        _frames.append(np.pad(_patch, _pad, mode="edge"))
    _stack = np.stack(_frames)
    _lo, _hi = np.percentile(_stack, [10.0, 99.5])
    _stack = np.clip((_stack-_lo)/max(float(_hi-_lo), 1.0), -0.5, 2.0)
    if _crop_mode == "raw_diff":
        _stack = np.concatenate([_stack, _stack[1:2]-_stack[0:1], _stack[2:3]-_stack[1:2]], axis=0)
    return _stack.astype(np.float16)


_train_geffs = sorted(TRAIN_DIR.glob("*.geff"))
for _geff_path in _train_geffs:
    _stem = _geff_path.name[:-5]
    _image_path = TRAIN_DIR / f"{_stem}.zarr" / "0"
    if not _image_path.exists():
        continue
    _graph = graph_from_geff(_geff_path)
    _rows = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r for r in _graph.node_attrs().iter_rows(named=True)}
    _positive_ids = [nid for nid in _rows if _graph.out_degree(nid) >= 2]
    _negative_ids = [nid for nid in _rows if _graph.in_degree(nid) == 1 and _graph.out_degree(nid) == 1]
    _negative_cap = min(len(_negative_ids), max(20, 30*len(_positive_ids)))
    if _negative_cap < len(_negative_ids):
        _negative_ids = list(_crop_rng.choice(_negative_ids, size=_negative_cap, replace=False))
    _array = zarr.open(_image_path, mode="r")
    for _label, _ids in ((1, _positive_ids), (0, _negative_ids)):
        for _nid in _ids:
            _r = _rows[int(_nid)]
            _crop_examples.append({
                "stem": _stem,
                "embryo": _stem.split("_", 1)[0],
                "node_id": int(_nid),
                "label": _label,
                "crop": _extract_temporal_crop(
                    _array, int(_r[td.DEFAULT_ATTR_KEYS.T]),
                    tuple(float(_r[k]) for k in ("z", "y", "x")),
                ),
            })

if not _crop_examples:
    raise RuntimeError("No temporal crop examples were extracted")
_crop_labels = np.asarray([x["label"] for x in _crop_examples], dtype=np.int64)
_crop_embryos = np.asarray([x["embryo"] for x in _crop_examples])
print("Temporal crops:", len(_crop_examples), "positives:", int(_crop_labels.sum()),
      "embryos:", sorted(set(_crop_embryos)), "mode:", _crop_mode)


class _DivisionCropNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(channels, 16, 3, padding=1), nn.BatchNorm3d(16), nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.SiLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 48, 3, padding=1), nn.BatchNorm3d(48), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(48, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)


_crop_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_fold_rows, _prediction_rows = [], []
_crop_models = {}
for _heldout in sorted(set(_crop_embryos)):
    _train_idx = np.flatnonzero(_crop_embryos != _heldout)
    _test_idx = np.flatnonzero(_crop_embryos == _heldout)
    _train_pos = _train_idx[_crop_labels[_train_idx] == 1]
    _train_neg = _train_idx[_crop_labels[_train_idx] == 0]
    if not len(_train_pos) or not len(_train_neg) or not _crop_labels[_test_idx].sum():
        continue
    torch.manual_seed(_crop_seed)
    _model = _DivisionCropNet(_crop_examples[0]["crop"].shape[0]).to(_crop_device)
    _optimizer = torch.optim.AdamW(_model.parameters(), lr=2e-3, weight_decay=1e-3)
    _loss_fn = nn.BCEWithLogitsLoss()
    _steps_per_epoch = max(20, int(np.ceil(len(_train_neg)/32)))
    _sampler_rng = np.random.default_rng(_crop_seed)
    _model.train()
    for _epoch in range(25):
        for _ in range(_steps_per_epoch):
            _npos = 8
            _nneg = 24
            _indices = np.concatenate([
                _sampler_rng.choice(_train_pos, _npos, replace=True),
                _sampler_rng.choice(_train_neg, _nneg, replace=len(_train_neg) < _nneg),
            ])
            _sampler_rng.shuffle(_indices)
            _x = np.stack([_crop_examples[i]["crop"] for i in _indices]).astype(np.float32)
            # Spatial jitter and flips preserve the temporal channel order.
            if _crop_rng.random() < 0.5:
                _x = _x[..., ::-1].copy()
            if _crop_rng.random() < 0.5:
                _x = _x[..., ::-1, :].copy()
            if _crop_translation_aug:
                for _bi in range(len(_x)):
                    _shift = (
                        int(_crop_rng.integers(-1, 2)),
                        int(_crop_rng.integers(-4, 5)),
                        int(_crop_rng.integers(-4, 5)),
                    )
                    _x[_bi] = np.roll(_x[_bi], _shift, axis=(-3, -2, -1))
            _y = _crop_labels[_indices].astype(np.float32)
            _optimizer.zero_grad(set_to_none=True)
            _logits = _model(torch.from_numpy(_x).to(_crop_device))
            _loss = _loss_fn(_logits, torch.from_numpy(_y).to(_crop_device))
            _loss.backward()
            _optimizer.step()

    _model.eval()
    _scores = []
    with torch.no_grad():
        for _start in range(0, len(_test_idx), 32):
            _batch_idx = _test_idx[_start:_start+32]
            _x = np.stack([_crop_examples[i]["crop"] for i in _batch_idx]).astype(np.float32)
            _scores.extend(torch.sigmoid(_model(torch.from_numpy(_x).to(_crop_device))).cpu().numpy().tolist())
    _truth = _crop_labels[_test_idx]
    _fold_rows.append({
        "heldout_embryo": _heldout,
        "train_examples": len(_train_idx),
        "train_positives": len(_train_pos),
        "test_examples": len(_test_idx),
        "test_positives": int(_truth.sum()),
        "average_precision": float(_crop_ap(_truth, _scores)),
        "roc_auc": float(_crop_auc(_truth, _scores)),
    })
    _crop_models[_heldout] = _model.cpu().eval()
    if _crop_device.type == "cuda":
        torch.cuda.empty_cache()
    for _idx, _score in zip(_test_idx, _scores):
        _ex = _crop_examples[int(_idx)]
        _prediction_rows.append({
            "stem": _ex["stem"], "embryo": _heldout, "node_id": _ex["node_id"],
            "label": _ex["label"], "score": float(_score),
        })

_folds = pd.DataFrame(_fold_rows)
_predictions = pd.DataFrame(_prediction_rows)
_threshold_rows = []
for _threshold in sorted(set(np.quantile(_predictions["score"], [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]).tolist())):
    _accepted = _predictions[_predictions["score"] >= _threshold]
    _tp = int(_accepted["label"].sum())
    _threshold_rows.append({
        "threshold": float(_threshold), "accepted": len(_accepted),
        "tp": _tp, "fp": len(_accepted)-_tp,
        "precision": float(_tp/len(_accepted)) if len(_accepted) else 0.0,
        "recall": float(_tp/max(int(_predictions["label"].sum()), 1)),
    })
_thresholds = pd.DataFrame(_threshold_rows)
_summary = {
    "mode": _crop_mode,
    "translation_augmentation": _crop_translation_aug,
    "examples": len(_crop_examples),
    "positives": int(_crop_labels.sum()),
    "mean_average_precision": float(_folds["average_precision"].mean()),
    "mean_roc_auc": float(_folds["roc_auc"].mean()),
}
_folds.to_csv(WORKING_DIR / "temporal_crop_folds.csv", index=False)
_predictions.to_csv(WORKING_DIR / "temporal_crop_oof.csv", index=False)
_thresholds.to_csv(WORKING_DIR / "temporal_crop_thresholds.csv", index=False)
with (WORKING_DIR / "temporal_crop_summary.json").open("w") as _fh:
    json.dump(_summary, _fh, indent=2, sort_keys=True)
print(json.dumps(_summary, indent=2, sort_keys=True))
print(_folds.to_string(index=False))
print(_thresholds.to_string(index=False))
