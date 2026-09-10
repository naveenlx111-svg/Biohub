# E0052: candidate-domain seven-offset hard-negative fine-tuning.
import copy as _e52_copy
import zarr as _e52_zarr
import torch as _e52_torch

_e52_offsets = [(0, 0, 0), (-2, 0, 0), (2, 0, 0), (0, -8, 0),
                (0, 8, 0), (0, 0, -8), (0, 0, 8)]
_e52_rng = np.random.default_rng(5200)
_e52_node_cache = {}

def _e52_crop(frames, center, rz=4, ryx=16):
    zc, yc, xc = (int(round(float(v))) for v in center)
    out = []
    for frame in frames:
        z0, z1 = max(0, zc-rz), min(frame.shape[0], zc+rz+1)
        y0, y1 = max(0, yc-ryx), min(frame.shape[1], yc+ryx+1)
        x0, x1 = max(0, xc-ryx), min(frame.shape[2], xc+ryx+1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
        pad = ((max(0, rz-(zc-z0)), max(0, rz-(z1-zc-1))),
               (max(0, ryx-(yc-y0)), max(0, ryx-(y1-yc-1))),
               (max(0, ryx-(xc-x0)), max(0, ryx-(x1-xc-1))))
        out.append(np.pad(patch, pad, mode="edge"))
    stack = np.stack(out)
    lo, hi = np.percentile(stack, [10.0, 99.5])
    return np.clip((stack-lo)/max(float(hi-lo), 1.0), -0.5, 2.0).astype(np.float32)

_e52_rows = []
for _e52_cache_stem in _parents["stem"].unique():
    _e52_cache_graph = _official_graph_from_processed(*official_graph_inputs[_e52_cache_stem])
    _e52_node_cache[_e52_cache_stem] = {
        int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r
        for r in _e52_cache_graph.node_attrs().iter_rows(named=True)
    }
for _e52_stem, _e52_group in _parents.groupby("stem", sort=True):
    _e52_emb = _e52_stem.split("_", 1)[0]
    if _e52_emb not in _crop_models:
        continue
    _e52_graph = _official_graph_from_processed(*official_graph_inputs[_e52_stem])
    _e52_nodes = _e52_node_cache[_e52_stem]
    _e52_array = _e52_zarr.open(TRAIN_DIR / f"{_e52_stem}.zarr" / "0", mode="r")
    _e52_model = _e52_copy.deepcopy(_crop_models[_e52_emb]).to(_crop_device)
    _e52_train = _parents[_parents["embryo"].ne(_e52_emb)].copy()
    _e52_pos = _e52_train[_e52_train.label.eq(1)]
    _e52_neg = _e52_train[_e52_train.label.eq(0)].sort_values("score", ascending=False)
    _e52_neg = _e52_neg.groupby("stem", sort=False).head(512)
    _e52_train_rows = pd.concat([_e52_pos, _e52_neg], ignore_index=True)
    # Bound the exploratory run: reopening large Zarr volumes per row makes a
    # full candidate sweep impractical on CPU. Keep all positives and a
    # reproducible hard-negative subset.
    _e52_max_train = min(len(_e52_train_rows), 512)
    if len(_e52_train_rows) > _e52_max_train:
        _e52_pos_keep = _e52_pos
        _e52_neg_keep = _e52_neg.head(max(0, _e52_max_train - len(_e52_pos_keep)))
        _e52_train_rows = pd.concat([_e52_pos_keep, _e52_neg_keep], ignore_index=True)
    _e52_model.train()
    _e52_opt = _e52_torch.optim.AdamW(_e52_model.parameters(), lr=3e-4, weight_decay=1e-3)
    _e52_loss = _e52_torch.nn.BCEWithLogitsLoss()
    # Keep the CPU fallback bounded; the full eight-epoch sweep exhausted the
    # notebook kernel before producing an artifact on this machine.
    for _e52_epoch in range(1):
        _e52_order = _e52_rng.permutation(len(_e52_train_rows))
        for _e52_start in range(0, len(_e52_order), 16):
            _e52_batch = _e52_train_rows.iloc[_e52_order[_e52_start:_e52_start+16]]
            _e52_x, _e52_y = [], []
            for _e52_row in _e52_batch.itertuples(index=False):
                _e52_path = TRAIN_DIR / f"{_e52_row.stem}.zarr" / "0"
                _e52_a = _e52_zarr.open(_e52_path, mode="r")
                _e52_t = int(_e52_row.t)
                _e52_frames = [np.asarray(_e52_a[ft]) for ft in
                               (max(0, _e52_t-1), _e52_t, min(int(_e52_a.shape[0])-1, _e52_t+1))]
                _e52_node = _e52_node_cache.get(_e52_row.stem, {}).get(int(_e52_row.fork_id))
                if _e52_node is None:
                    continue
                _e52_center = tuple(float(_e52_node[k]) for k in ("z", "y", "x"))
                _e52_bag = np.stack([_e52_crop(_e52_frames, tuple(_e52_center[i] + off[i] for i in range(3)))
                                      for off in _e52_offsets])
                _e52_x.append(_e52_bag)
                _e52_y.append(float(_e52_row.label))
            if not _e52_x:
                continue
            _e52_tensor = _e52_torch.from_numpy(np.stack(_e52_x)).to(_crop_device)
            _e52_shape = _e52_tensor.shape
            _e52_logits = _e52_model(_e52_tensor.reshape(-1, *_e52_shape[2:]))
            _e52_bag_logits = _e52_logits.reshape(_e52_shape[0], _e52_shape[1]).max(dim=1).values
            _e52_target = _e52_torch.tensor(_e52_y, dtype=_e52_bag_logits.dtype, device=_crop_device)
            _e52_opt.zero_grad(set_to_none=True)
            _e52_loss(_e52_bag_logits, _e52_target).backward()
            _e52_opt.step()
    _e52_model.eval()
    _e52_eval = _parents[_parents["embryo"].eq(_e52_emb)]
    for _e52_start in range(0, len(_e52_eval), 32):
        _e52_batch = _e52_eval.iloc[_e52_start:_e52_start+32]
        _e52_x, _e52_meta = [], []
        for _e52_row in _e52_batch.itertuples(index=False):
            _e52_path = TRAIN_DIR / f"{_e52_row.stem}.zarr" / "0"
            _e52_a = _e52_zarr.open(_e52_path, mode="r")
            _e52_t = int(_e52_row.t)
            _e52_frames = [np.asarray(_e52_a[ft]) for ft in
                           (max(0, _e52_t-1), _e52_t, min(int(_e52_a.shape[0])-1, _e52_t+1))]
            _e52_node = _e52_node_cache.get(_e52_row.stem, {}).get(int(_e52_row.fork_id))
            if _e52_node is None:
                continue
            _e52_center = tuple(float(_e52_node[k]) for k in ("z", "y", "x"))
            _e52_x.append(np.stack([_e52_crop(_e52_frames, tuple(_e52_center[i] + off[i] for i in range(3)))
                                    for off in _e52_offsets]))
            _e52_meta.append(_e52_row)
        if not _e52_x:
            continue
        with _e52_torch.no_grad():
            _e52_tensor = _e52_torch.from_numpy(np.stack(_e52_x)).to(_crop_device)
            _e52_shape = _e52_tensor.shape
            _e52_logits = _e52_model(_e52_tensor.reshape(-1, *_e52_shape[2:]))
            _e52_scores = _e52_logits.reshape(_e52_shape[0], _e52_shape[1]).max(dim=1).values.sigmoid().cpu().numpy()
        for _e52_row, _e52_score in zip(_e52_meta, _e52_scores):
            _e52_rows.append({"stem": _e52_row.stem, "embryo": _e52_emb,
                              "fork_id": int(_e52_row.fork_id),
                              "daughter1_id": int(_e52_row.daughter1_id),
                              "daughter2_id": int(_e52_row.daughter2_id),
                              "label": int(_e52_row.label), "score": float(_e52_score)})
    _e52_model.cpu()
    if _crop_device.type == "cuda":
        _e52_torch.cuda.empty_cache()

_e52_out = pd.DataFrame(_e52_rows)
_e52_out.to_csv(WORKING_DIR / "e0052_candidate_mil_oof.csv", index=False)
_e52_metrics = []
for _e52_k in (1, 2, 3, 5, 10, 20, 50, 100):
    _e52_top = _e52_out.sort_values("score", ascending=False).head(_e52_k)
    _e52_metrics.append({"k": _e52_k, "tp": int(_e52_top.label.sum()),
                         "fp": int(len(_e52_top)-_e52_top.label.sum())})
_e52_summary = {"candidates": len(_e52_out), "positives": int(_e52_out.label.sum()),
                "top": _e52_metrics}
(WORKING_DIR / "e0052_summary.json").write_text(json.dumps(_e52_summary, indent=2) + "\n")
print(json.dumps(_e52_summary, indent=2))
