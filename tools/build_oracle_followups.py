"""E0069/E0070: temporal repair guards and feasible daughter-pair selection."""
import copy
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parent=root / "research_members/naveen/experiments/E0068_repair_selection_audit"
nb=json.loads((parent / "biohub-e0068-repair-selection-audit.ipynb").read_text())
body="".join(nb["cells"][-1]["source"])
setup=body.split("# Static topology gate")[0]
evaluate=body[body.index("rows=[]\n"):]
common='''
def topology(stem):
    nodes,links=graphs[stem]
    incoming,outgoing={},{}
    for s,d in links:
        incoming[d]=s
        outgoing.setdefault(s,set()).add(d)
    return nodes,incoming,outgoing
def feasible(r, nodes, incoming, outgoing):
    p,a,b=int(r.fork_id),int(r.daughter1_id),int(r.daughter2_id)
    if any(x not in nodes for x in (p,a,b)) or a==b: return False
    if nodes[a]["t"]!=nodes[p]["t"]+1 or nodes[b]["t"]!=nodes[p]["t"]+1: return False
    children=outgoing.get(p,set())
    if p not in incoming or len(children)!=1 or not children <= {a,b}: return False
    if not outgoing.get(a) or not outgoing.get(b): return False
    new=b if a in children else a
    old=incoming.get(new)
    return old is None or len(outgoing.get(old,set()))==1
ranked=candidate.sort_values("offset_max_score",ascending=False,kind="stable")
configs={"base":candidate.iloc[:0],"replay_global_top100":ranked.head(100)}
'''
temporal='''
feature_rows=[]
for stem,group in candidate.groupby("stem",sort=True):
    nodes,incoming,outgoing=topology(stem)
    def chain(n,steps):
        result=[n]
        for _ in range(steps):
            children=outgoing.get(result[-1],set())
            if len(children)!=1: break
            result.append(next(iter(children)))
        return result
    def point(n):
        return np.array([nodes[n][k] for k in ("z","y","x")])*np.array(VOXEL_SCALE_UM)
    for r in group.itertuples():
        p,a,b=int(r.fork_id),int(r.daughter1_id),int(r.daughter2_id)
        ac,bc=chain(a,3),chain(b,3)
        persisted=len(ac)==4 and len(bc)==4
        sep0=float(np.linalg.norm(point(a)-point(b)))
        sep3=float(np.linalg.norm(point(ac[-1])-point(bc[-1]))) if persisted else float("nan")
        feature_rows.append(dict(index=r.Index,persistence3=persisted,
            separation_gain3=sep3-sep0,feasible=feasible(r,nodes,incoming,outgoing)))
features=pd.DataFrame(feature_rows).set_index("index")
candidate=candidate.join(features)
candidate.to_csv(WORKING_DIR / "temporal_candidate_features.csv",index=False)
ranked=candidate.sort_values("offset_max_score",ascending=False,kind="stable")
for k in (100,250):
    top=ranked.head(k)
    configs[f"unfiltered_top{k}"]=top
    configs[f"persistence3_top{k}"]=top[top.persistence3]
    configs[f"persistence3_nonconverging_top{k}"]=top[top.persistence3 & top.separation_gain3.ge(0)]
    configs[f"persistence3_diverge2um_top{k}"]=top[top.persistence3 & top.separation_gain3.ge(2)]
'''
pairs='''
from sklearn.ensemble import HistGradientBoostingClassifier
pair_features=["d1_um","d2_um","distance_sum_um","distance_asymmetry_um",
    "sister_um","midpoint_um","daughter_cosine","fork_outdegree",
    "daughter1_indegree","daughter2_indegree","edge1_exists","edge2_exists"]
all_pairs=pd.read_csv(study / "fork_candidates.csv",float_precision="round_trip")
all_pairs=all_pairs[(all_pairs.distance_sum_um<=14)&(all_pairs.sister_um>=4)
    &(all_pairs.midpoint_um<=5)&(all_pairs.daughter_cosine<=-0.30)].copy()
saved=pd.read_csv(study / "frozen_gate_parents.csv")
scored=[]
for heldout in sorted(all_pairs.embryo.unique()):
    tr=all_pairs[all_pairs.embryo.ne(heldout)]
    te=all_pairs[all_pairs.embryo.eq(heldout)].copy()
    w=np.ones(len(tr));w[tr.label.to_numpy()==1]=int((tr.label==0).sum())/max(int(tr.label.sum()),1)
    model=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=250,max_leaf_nodes=15,
        min_samples_leaf=20,l2_regularization=2.0,random_state=20260830)
    model.fit(tr[pair_features],tr.label,sample_weight=w)
    te["pair_score"]=model.predict_proba(te[pair_features])[:,1]
    replay=te[te.groupby(["stem","fork_id"]).pair_score.rank(method="first",ascending=False).eq(1)]
    cols=["stem","fork_id","daughter1_id","daughter2_id"]
    expected=saved[saved.embryo.eq(heldout)]
    assert set(map(tuple,replay[cols].to_numpy()))==set(map(tuple,expected[cols].to_numpy()))
    scored.append(te.drop(columns=["label"]))
scored=pd.concat(scored,ignore_index=True)
keep=[]
for stem,group in scored.groupby("stem",sort=True):
    nodes,incoming,outgoing=topology(stem)
    keep.extend(r.Index for r in group.itertuples() if feasible(r,nodes,incoming,outgoing))
legal=scored.loc[keep].copy()
legal=legal[legal.groupby(["stem","fork_id"]).pair_score.rank(method="first",ascending=False).eq(1)]
parent_scores=candidate[["stem","fork_id","offset_max_score"]]
assert not parent_scores.duplicated(["stem","fork_id"]).any()
legal=legal.merge(parent_scores,on=["stem","fork_id"],how="inner",validate="one_to_one")
legal=legal.sort_values(["stem","t","fork_id"],kind="stable")
legal.to_csv(WORKING_DIR / "feasible_pair_candidates.csv",index=False)
reranked=legal.sort_values("offset_max_score",ascending=False,kind="stable")
for k in (25,100,250):
    configs[f"feasible_pair_first_top{k}"]=reranked.head(k)
'''
for exp,slug,title,extra in [
    ("E0069","biohub-e0069-temporal-repair-guards","Biohub E0069 Temporal Repair Guards",temporal),
    ("E0070","biohub-e0070-feasible-pair-ranking","Biohub E0070 Feasible Pair Ranking",pairs)]:
    out=copy.deepcopy(nb)
    out["cells"][-1]["source"]=(setup+common+extra+evaluate).splitlines(True)
    for i,c in enumerate(out["cells"]):
        c.update(id=f"{exp.lower()}-{i:02d}",execution_count=None,outputs=[])
        compile("".join(c["source"]),f"{exp}:{i}","exec")
    folder=root / "research_members/naveen/experiments" / f"{exp}_{slug.split(exp.lower()+'-')[1].replace('-','_')}"
    folder.mkdir(exist_ok=True)
    (folder / f"{slug}.ipynb").write_text(json.dumps(out,indent=1)+"\n")
    meta=json.loads((parent / "kernel-metadata.json").read_text())
    meta.update(id=f"naveenlx111249971939/{slug}",title=title,code_file=f"{slug}.ipynb")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    (folder / "provenance.json").write_text(json.dumps(dict(parent="E0065 cached OOF scores and graphs",
        hypothesis=title,compute="Kaggle only",validation="Official baseline and top100 replay; exploratory panel; no submission",
        frozen="CNN scores; detector graphs; historical geometry gate; no test GT"),indent=2)+"\n")
    print(folder)
