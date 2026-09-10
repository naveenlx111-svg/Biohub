"""E0068: predeclared label-free repair selection comparisons on Kaggle."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
parent = root / "research_members/naveen/experiments/E0065_kaggle_division_ranker"
n = json.loads((parent / "biohub-e0065-kaggle-division-ranker.ipynb").read_text())
parts = ["".join(c["source"]) for c in n["cells"][:5]]
runtime = (root / "tools/division_repair_deploy_runtime.py").read_text().split('oof = pd.read_csv')[0]
parts.append(runtime + '''
paths = list(Path("/kaggle/input").rglob("offset_pool_transfer_oof.csv"))
assert len(paths)==1, paths
study = paths[0].parent
candidate = pd.read_csv(paths[0],usecols=["stem","embryo","fork_id","daughter1_id","daughter2_id","offset_max_score"])
train = COMP_DIR / "train"
graphs = {}
for stem in sorted(candidate.stem.unique()):
    nf = pd.read_csv(study / f"{stem}_nodes.csv")
    ef = pd.read_csv(study / f"{stem}_edges.csv")
    nodes = {int(r.node_id):dict(node_id=int(r.node_id),t=int(r.t),z=float(r.z),y=float(r.y),x=float(r.x)) for r in nf.itertuples()}
    links = {(int(r.source_id),int(r.target_id)) for r in ef.itertuples()}
    graphs[stem] = (nodes,links)
# Static topology gate is equivalent to one isolated repair's feasibility.
eligible = []
for stem, group in candidate.groupby("stem",sort=True):
    nodes, links = graphs[stem]
    incoming,outgoing = {},{}
    for s,d in links:
        incoming[d]=s
        outgoing.setdefault(s,set()).add(d)
    for r in group.itertuples():
        p,a,b = int(r.fork_id),int(r.daughter1_id),int(r.daughter2_id)
        if any(x not in nodes for x in (p,a,b)) or a==b: continue
        if nodes[a]["t"]!=nodes[p]["t"]+1 or nodes[b]["t"]!=nodes[p]["t"]+1: continue
        children=outgoing.get(p,set())
        if p not in incoming or len(children)!=1 or not children <= {a,b}: continue
        if not outgoing.get(a) or not outgoing.get(b): continue
        new = b if a in children else a
        old = incoming.get(new)
        if old is not None and len(outgoing.get(old,set()))!=1: continue
        eligible.append(r.Index)
filtered = candidate.loc[eligible]
configs = {"base":candidate.iloc[:0]}
configs["replay_global_top100"] = candidate.sort_values("offset_max_score",ascending=False,kind="stable").head(100)
for name,pool in [("all",candidate),("topology_first",filtered)]:
    ranked=pool.sort_values("offset_max_score",ascending=False,kind="stable")
    if name=="topology_first":
        configs["topology_first_global100"] = ranked.head(100)
    for k in (25,50,100):
        configs[f"{name}_per_embryo{k}"] = ranked.groupby("embryo",sort=False).head(k)
rows=[]
for label,selected in configs.items():
    for stem,(nodes,links) in graphs.items():
        edited,applied=repaired_links(nodes,links,selected[selected.stem.eq(stem)])
        pred=_official_graph_from_processed(nodes,[dict(source_id=s,target_id=d) for s,d in sorted(edited)])
        gt=graph_from_geff(train / f"{stem}.geff")
        result=_official_evaluate(pred,gt,scale=tuple(VOXEL_SCALE_UM),max_distance=7.0)
        total=float((_GeffMetadata.read(train / f"{stem}.geff").extra or {})["estimated_number_of_nodes"])
        row=_official_per_sample_metrics(result,total,_official_node_recall(pred,gt))
        row.update(config=label,stem=stem,embryo=stem.split("_")[0],edits=len(applied))
        rows.append(row)
    group=[r for r in rows if r["config"]==label]
    score=_official_summarise(group)["score"]
    if label=="base": assert abs(score-0.9435550596933744)<1e-10
    if label=="replay_global_top100": assert abs(score-0.9499248688395877)<1e-10
    print("SCORED",label,score,flush=True)
    pd.DataFrame(rows).to_csv(WORKING_DIR / "selection_samples.csv",index=False)
summaries=[]
for label in configs:
    group=[r for r in rows if r["config"]==label]
    summaries.append(dict(config=label,edits=sum(r["edits"] for r in group),**_official_summarise(group)))
    for embryo in sorted(candidate.embryo.unique()):
        subset=[r for r in group if r["embryo"]==embryo]
        summaries.append(dict(config=label,embryo=embryo,edits=sum(r["edits"] for r in subset),**_official_summarise(subset)))
(WORKING_DIR / "selection_summary.json").write_text(json.dumps(summaries,indent=2))
''')
folder=root / "research_members/naveen/experiments/E0068_repair_selection_audit"
folder.mkdir(exist_ok=True)
cells=[]
for i,s in enumerate(parts):
    compile(s,f"E0068:{i}","exec")
    cells.append(dict(cell_type="code",id=f"e0068-{i:02d}",metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
slug="biohub-e0068-repair-selection-audit"
(folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=n["metadata"],cells=cells),indent=1)+"\n")
meta=json.loads((parent / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}",title="Biohub E0068 Repair Selection Audit",code_file=f"{slug}.ipynb")
meta["kernel_sources"]=["naveenlx111249971939/biohub-e0065-kaggle-division-ranker"]
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
print(folder)
