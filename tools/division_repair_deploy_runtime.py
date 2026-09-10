"""Embedded Kaggle cell: replay official diagnostics and export a conservative repair."""
def repaired_links(nodes, baseline, selected):
    links = set(baseline)
    incoming, outgoing = {}, {}
    for s,d in links:
        incoming[d] = s
        outgoing.setdefault(s,set()).add(d)
    applied = []
    for row in selected.itertuples():
        p,a,b = int(row.fork_id),int(row.daughter1_id),int(row.daughter2_id)
        if any(x not in nodes for x in (p,a,b)) or a==b:
            continue
        if nodes[a]["t"] != nodes[p]["t"]+1 or nodes[b]["t"] != nodes[p]["t"]+1:
            continue
        current = outgoing.get(p,set())
        if p not in incoming or len(current)!=1 or not current <= {a,b}:
            continue
        if not outgoing.get(a) or not outgoing.get(b):
            continue
        child = b if a in current else a
        old = incoming.get(child)
        if old is not None:
            if len(outgoing.get(old,set())) != 1:
                continue
            links.remove((old,child))
            outgoing[old].remove(child)
        links.add((p,child))
        outgoing[p].add(child)
        incoming[child] = p
        applied.append(dict(parent=p,child=child,old_parent=old))
    return links,applied

oof = pd.read_csv(study / "offset_pool_transfer_oof.csv")
ranked = oof.sort_values("offset_max_score",ascending=False,kind="stable")
chosen = ranked.head(100)
threshold = float(chosen.offset_max_score.min())
replay_rows = []
for stem in sorted(oof.stem.unique()):
    nf = pd.read_csv(study / f"{stem}_nodes.csv")
    ef = pd.read_csv(study / f"{stem}_edges.csv")
    nodes = {int(r.node_id):dict(node_id=int(r.node_id),t=int(r.t),z=float(r.z),y=float(r.y),x=float(r.x)) for r in nf.itertuples()}
    links, edits = repaired_links(nodes,{(int(r.source_id),int(r.target_id)) for r in ef.itertuples()},chosen[chosen.stem.eq(stem)])
    pred = _official_graph_from_processed(nodes,[dict(source_id=s,target_id=d) for s,d in sorted(links)])
    gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
    result = _official_evaluate(pred,gt,scale=tuple(VOXEL_SCALE_UM),max_distance=7.0)
    total = float((_GeffMetadata.read(TRAIN_DIR / f"{stem}.geff").extra or {})["estimated_number_of_nodes"])
    row = _official_per_sample_metrics(result,total,_official_node_recall(pred,gt))
    row.update(stem=stem,edits=len(edits))
    replay_rows.append(row)
summary = _official_summarise(replay_rows)
assert abs(summary["score"]-0.9499248688395877)<1e-10, summary
assert sum(r["edits"] for r in replay_rows)==75
(WORKING_DIR / "diagnostic_replay.json").write_text(json.dumps(summary,indent=2))

# Freeze the validation-derived score threshold, with a candidate-density cap.
# On the diagnostic table this rule selects exactly the original top 100.
# This is exploratory calibration, not an independent generalisation estimate.
test_candidates = _offset_oof.sort_values("offset_max_score",ascending=False,kind="stable")
budget = max(1,int(np.ceil(len(test_candidates)*100/len(oof))))
selected = test_candidates[test_candidates.offset_max_score.ge(threshold)].head(budget)
selected.to_csv(WORKING_DIR / "test_selected_repairs.csv",index=False)
out_rows,edit_rows = [],[]
for stem in test_stems:
    graph = _official_graph_from_processed(*official_graph_inputs[stem])
    nodes = {int(r["node_id"]):dict(node_id=int(r["node_id"]),t=int(r["t"]),z=float(r["z"]),y=float(r["y"]),x=float(r["x"])) for r in graph.node_attrs().iter_rows(named=True)}
    baseline = {(int(r["source_id"]),int(r["target_id"])) for r in graph.edge_attrs().iter_rows(named=True)}
    links,edits = repaired_links(nodes,baseline,selected[selected.stem.eq(stem)])
    for nid,r in sorted(nodes.items()):
        out_rows.append(dict(dataset=stem,row_type="node",**r))
    for s,d in sorted(links):
        out_rows.append(dict(dataset=stem,row_type="edge",source_id=s,target_id=d))
    edit_rows.extend(dict(dataset=stem,**r) for r in edits)
frame = pd.DataFrame(out_rows)
frame.insert(0,"id",np.arange(len(frame)))
frame = frame.reindex(columns=["id","dataset","row_type","node_id","t","z","y","x","source_id","target_id"])
for col in ["id","node_id","t","z","y","x","source_id","target_id"]:
    frame[col] = frame[col].astype("Int64")
frame.to_csv(WORKING_DIR / "submission.csv",index=False)
pd.DataFrame(edit_rows).to_csv(WORKING_DIR / "applied_test_edits.csv",index=False)
(WORKING_DIR / "deployment_summary.json").write_text(json.dumps(dict(
    diagnostic_score=summary["score"],threshold=threshold,budget=budget,
    test_candidates=len(test_candidates),selected=len(selected),applied=len(edit_rows),
    public_score=None,baseline_public=0.946,policy="OOF score threshold plus candidate-density cap; no test GT"),indent=2))
