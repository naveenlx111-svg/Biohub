"""E0072: event-level diagnostic funnel on exact E0065 graph namespace."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","4")
os.environ.setdefault("OMP_NUM_THREADS","4")
import json
from pathlib import Path
import pandas as pd
import numpy as np
import tracksdata as td

root=Path(__file__).resolve().parents[1]
source=root / "local_runs/E0065/analysis_inputs"
graphs=root / "local_runs/E0060/working"
train=root / "data/kaggle_input/competitions/biohub-cell-tracking-during-development/train"
out=root / "local_runs/E0072";out.mkdir(exist_ok=True)
n=json.loads((root / "research_members/naveen/experiments/E0065_kaggle_division_ranker/biohub-e0065-kaggle-division-ranker.ipynb").read_text())
setup="".join(n["cells"][4]["source"]).replace('/kaggle/input',str(root / 'data/kaggle_input'))
exec(compile(setup,"official_setup","exec"))
from tracking_cellmot.division_metrics import extract_divisions,match_divisions,score_divisions,_matched_node_attrs,_matched_division_nodes
runtime=(root / "tools/division_repair_deploy_runtime.py").read_text().split('oof = pd.read_csv')[0]
exec(compile(runtime,"repair_function","exec"))
candidate=pd.read_csv(source / "fork_candidates.csv",float_precision="round_trip")
parents=pd.read_csv(source / "frozen_gate_parents.csv")
scores=pd.read_csv(source / "offset_pool_transfer_oof.csv")
scores=scores.sort_values("offset_max_score",ascending=False,kind="stable").reset_index(drop=True)
scores["global_rank"]=np.arange(1,len(scores)+1)
selected=scores.head(100)
rows=[];base_metrics=[];repair_metrics=[]
def triples(df):
    return set(map(tuple,df[["fork_id","daughter1_id","daughter2_id"]].astype(int).to_numpy()))
for stem,group in candidate.groupby("stem",sort=True):
    nf=pd.read_csv(graphs / f"{stem}_nodes.csv")
    ef=pd.read_csv(graphs / f"{stem}_edges.csv")
    nodes={int(r.node_id):dict(node_id=int(r.node_id),t=int(r.t),z=float(r.z),y=float(r.y),x=float(r.x)) for r in nf.itertuples()}
    links={(int(r.source_id),int(r.target_id)) for r in ef.itertuples()}
    pred=_official_graph_from_processed(nodes,[dict(source_id=s,target_id=d) for s,d in sorted(links)])
    gt=td.graph.IndexedRXGraph.from_geff(train / f"{stem}.geff")
    if isinstance(gt,tuple):gt=gt[0]
    edited,applied=repaired_links(nodes,links,selected[selected.stem.eq(stem)])
    repaired=_official_graph_from_processed(nodes,[dict(source_id=s,target_id=d) for s,d in sorted(edited)])
    total=float((_GeffMetadata.read(train / f"{stem}.geff").extra or {})["estimated_number_of_nodes"])
    for graph,collection in [(pred,base_metrics),(repaired,repair_metrics)]:
        result=_official_evaluate(graph,gt,scale=(1.625,.40625,.40625),max_distance=7.0)
        collection.append(_official_per_sample_metrics(result,total,_official_node_recall(graph,gt)))
    matched=match_divisions(pred,gt,scale=(1.625,.40625,.40625),max_distance=7.0)
    before=score_divisions(pred,gt,scale=(1.625,.40625,.40625),max_distance=7.0)
    after=score_divisions(repaired,gt,scale=(1.625,.40625,.40625),max_distance=7.0)
    times={int(r["node_id"]):int(r["t"]) for r in pred.node_attrs().iter_rows(named=True)}
    proposed=triples(group)
    gated=triples(group[(group.distance_sum_um<=14)&(group.sister_um>=4)&(group.midpoint_um<=5)&(group.daughter_cosine<=-.30)])
    kept=triples(parents[parents.stem.eq(stem)])
    stem_scores=scores[scores.stem.eq(stem)]
    rank={ (int(r.fork_id),int(r.daughter1_id),int(r.daughter2_id)):int(r.global_rank) for r in stem_scores.itertuples()}
    for divider,event in extract_divisions(gt).items():
        roles=_matched_division_nodes(_matched_node_attrs(matched[divider]),event,divider)
        possible=set()
        if roles is not None:
            pids,dsets=roles
            forks=set(map(int,pids))
            for p in pids: forks.update(map(int,matched[divider].successors(p)))
            for p in forks:
                if p not in times or len(dsets)<2:continue
                for a in dsets[0]:
                    for b in dsets[1]:
                        a,b=int(a),int(b)
                        if a!=b and times.get(a)==times[p]+1 and times.get(b)==times[p]+1:
                            possible.add((p,*sorted((a,b))))
        ranks=[rank[t] for t in possible if t in rank]
        tp0=int(before.scores.get(divider,0));tp1=int(after.scores.get(divider,0))
        if tp0:stage="already_recovered"
        elif tp1:stage="recovered_top100"
        elif not possible:stage="missing_matchable_role_triple"
        elif not possible & proposed:stage="proposal_generation"
        elif not possible & gated:stage="geometry_gate"
        elif not possible & kept:stage="pair_selection"
        elif not ranks or min(ranks)>100:stage="crop_ranking"
        else:stage="repair_application_or_event_connectivity"
        rows.append(dict(stem=stem,embryo=stem.split('_')[0],gt_division_id=int(divider),baseline_tp=tp0,
            repaired_tp=tp1,possible_triples=len(possible),proposed_triples=len(possible & proposed),
            gated_triples=len(possible & gated),selected_pair_triples=len(possible & kept),
            best_global_rank=min(ranks) if ranks else None,first_loss_stage=stage))
    print("AUDITED",stem,flush=True)
assert abs(_official_summarise(base_metrics)["score"]-.9435550596933744)<1e-10
assert abs(_official_summarise(repair_metrics)["score"]-.9499248688395877)<1e-10
df=pd.DataFrame(rows);df.to_csv(out / "event_funnel.csv",index=False)
summary=dict(events=len(df),counts=df.first_loss_stage.value_counts().to_dict(),
    baseline=_official_summarise(base_metrics),repaired=_official_summarise(repair_metrics),
    note="GT used for diagnostics only; no inference rules selected using event identities")
(out / "summary.json").write_text(json.dumps(summary,indent=2))
print(df.to_string(index=False));print(json.dumps(summary,indent=2))
