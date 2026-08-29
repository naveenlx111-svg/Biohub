# E0026: global label-free fork proposal audit against oracle-repairable events.
from itertools import combinations as _candidate_combinations

_proposal_radius_um = float(os.environ.get("BIOHUB_FORK_PROPOSAL_RADIUS_UM", "7.0"))
_proposal_k = int(os.environ.get("BIOHUB_FORK_PROPOSAL_K", "8"))
_proposal_sister_max_um = float(os.environ.get("BIOHUB_FORK_SISTER_MAX_UM", "12.0"))
fork_candidate_rows = []
fork_event_recall_rows = []

for _stem in val_stems:
    _base = _official_graph_from_processed(*official_graph_inputs[_stem])
    _gt = graph_from_geff(TRAIN_DIR / f"{_stem}.geff")
    _gt_divisions = _extract_divisions(_gt)
    _matched = _match_divisions(
        _base, _gt, scale=tuple(VOXEL_SCALE_UM), max_distance=VALIDATOR_MATCH_RADIUS_UM
    )
    _node_rows = {
        int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r
        for r in _base.node_attrs().iter_rows(named=True)
    }
    _positions = {
        nid: np.asarray([r["z"], r["y"], r["x"]], dtype=float) * np.asarray(VOXEL_SCALE_UM)
        for nid, r in _node_rows.items()
    }
    _times = {nid: int(r[td.DEFAULT_ATTR_KEYS.T]) for nid, r in _node_rows.items()}
    _base_pairs = {
        (int(e[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]), int(e[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]))
        for e in _base.edge_attrs().iter_rows(named=True)
    }
    _incoming, _outgoing = {}, {}
    for _s, _d in _base_pairs:
        _incoming.setdefault(_d, set()).add(_s)
        _outgoing.setdefault(_s, set()).add(_d)

    _nodes_at_time = {}
    for _nid, _time in _times.items():
        _nodes_at_time.setdefault(_time, []).append(_nid)

    # Positive triples are used only after label-free proposals have been generated.
    _positive_triples = set()
    _event_positive_triples = {}
    for _divider, _gt_div in _gt_divisions.items():
        _roles = _matched_division_nodes(
            _matched_node_attrs(_matched[_divider]), _gt_div, _divider
        )
        _event_key = (_stem, int(_divider))
        _event_positive_triples[_event_key] = set()
        if _roles is None:
            continue
        _parent_ids, _daughter_sets = _roles
        _forks = set(int(x) for x in _parent_ids)
        for _parent in _parent_ids:
            _forks.update(int(x) for x in _matched[_divider].successors(_parent))
        for _fork in _forks:
            if _fork not in _times:
                continue
            _next_t = _times[_fork] + 1
            _dsets = [
                {int(x) for x in ds if int(x) in _times and _times[int(x)] == _next_t}
                for ds in _daughter_sets[:2]
            ]
            if len(_dsets) < 2:
                continue
            for _d1 in _dsets[0]:
                for _d2 in _dsets[1]:
                    if _d1 == _d2:
                        continue
                    _triple = (_fork, *_sorted_pair(_d1, _d2)) if "_sorted_pair" in globals() else (_fork, *sorted((_d1, _d2)))
                    _positive_triples.add(_triple)
                    _event_positive_triples[_event_key].add(_triple)

    _proposed_triples = set()
    for _fork, _t in _times.items():
        _next_nodes = _nodes_at_time.get(_t + 1, [])
        if len(_next_nodes) < 2:
            continue
        _ordered = sorted(
            ((float(np.linalg.norm(_positions[_daughter] - _positions[_fork])), _daughter)
             for _daughter in _next_nodes),
            key=lambda item: (item[0], item[1]),
        )
        _near = [(dist, nid) for dist, nid in _ordered if dist <= _proposal_radius_um][:_proposal_k]
        for (_d1_dist, _d1), (_d2_dist, _d2) in _candidate_combinations(_near, 2):
            _sister = float(np.linalg.norm(_positions[_d1] - _positions[_d2]))
            if _sister > _proposal_sister_max_um:
                continue
            _d1, _d2 = sorted((_d1, _d2))
            _d1_dist = float(np.linalg.norm(_positions[_d1] - _positions[_fork]))
            _d2_dist = float(np.linalg.norm(_positions[_d2] - _positions[_fork]))
            _v1 = _positions[_d1] - _positions[_fork]
            _v2 = _positions[_d2] - _positions[_fork]
            _denom = max(float(np.linalg.norm(_v1) * np.linalg.norm(_v2)), 1e-8)
            _cosine = float(np.dot(_v1, _v2) / _denom)
            _midpoint = float(np.linalg.norm((_positions[_d1] + _positions[_d2]) / 2.0 - _positions[_fork]))
            _triple = (_fork, _d1, _d2)
            _proposed_triples.add(_triple)
            fork_candidate_rows.append({
                "stem": _stem,
                "embryo": _stem.split("_", 1)[0],
                "fork_id": _fork,
                "t": _t,
                "daughter1_id": _d1,
                "daughter2_id": _d2,
                "label": int(_triple in _positive_triples),
                "d1_um": _d1_dist,
                "d2_um": _d2_dist,
                "distance_sum_um": _d1_dist + _d2_dist,
                "distance_asymmetry_um": abs(_d1_dist - _d2_dist),
                "sister_um": _sister,
                "midpoint_um": _midpoint,
                "daughter_cosine": _cosine,
                "fork_outdegree": len(_outgoing.get(_fork, ())),
                "daughter1_indegree": len(_incoming.get(_d1, ())),
                "daughter2_indegree": len(_incoming.get(_d2, ())),
                "edge1_exists": int((_fork, _d1) in _base_pairs),
                "edge2_exists": int((_fork, _d2) in _base_pairs),
            })

    for (_event_stem, _divider), _triples in _event_positive_triples.items():
        fork_event_recall_rows.append({
            "stem": _event_stem,
            "embryo": _event_stem.split("_", 1)[0],
            "gt_division_id": _divider,
            "has_matched_role_triple": int(bool(_triples)),
            "proposed": int(bool(_triples & _proposed_triples)),
            "positive_triples": len(_triples),
            "proposed_positive_triples": len(_triples & _proposed_triples),
        })

_candidate_df = pd.DataFrame(fork_candidate_rows)
_event_recall_df = pd.DataFrame(fork_event_recall_rows)
_candidate_df.to_csv(WORKING_DIR / "fork_candidates.csv", index=False)
_event_recall_df.to_csv(WORKING_DIR / "fork_candidate_event_recall.csv", index=False)

_audit_summary = {
    "proposal_radius_um": _proposal_radius_um,
    "proposal_k": _proposal_k,
    "sister_max_um": _proposal_sister_max_um,
    "candidates": int(len(_candidate_df)),
    "positive_candidates": int(_candidate_df["label"].sum()) if not _candidate_df.empty else 0,
    "annotated_divisions": int(len(_event_recall_df)),
    "events_with_matched_role_triple": int(_event_recall_df["has_matched_role_triple"].sum()),
    "events_proposed": int(_event_recall_df["proposed"].sum()),
    "event_proposal_recall": float(_event_recall_df["proposed"].mean()),
    "conditional_proposal_recall": float(
        _event_recall_df.loc[_event_recall_df["has_matched_role_triple"] == 1, "proposed"].mean()
    ),
}
with (WORKING_DIR / "fork_candidate_audit_summary.json").open("w") as _fh:
    json.dump(_audit_summary, _fh, indent=2, sort_keys=True)
print("=" * 78)
print("GLOBAL LABEL-FREE FORK CANDIDATE AUDIT")
print(json.dumps(_audit_summary, indent=2, sort_keys=True))
print(_event_recall_df.groupby(["embryo", "has_matched_role_triple", "proposed"]).size().to_string())
print("=" * 78)
