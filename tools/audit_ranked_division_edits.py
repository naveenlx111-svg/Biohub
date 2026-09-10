"""E0059: score label-free ranked fork edits with the official graph metric.

Uses the exact E0047 namespace verified and saved by E0058. Candidate labels,
annotation status, and GT matching are excluded from selection. Diagnostics
only; this does not produce a competition submission or fit a new model.
"""
import copy
import json
import os
from pathlib import Path
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import pandas as pd
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "local_runs/E0058"
output = ROOT / "local_runs/E0059"
output.mkdir(exist_ok=True)
train = ROOT / "data/kaggle_input/competitions/biohub-cell-tracking-during-development/train"
nb = json.loads((ROOT / "research_members/naveen/experiments/E0033_lb0933_official_12/e0033-lb0933-official-12.ipynb").read_text())
setup = "".join(nb["cells"][9]["source"]).split("official_rows = []")[0]
setup = setup.replace("/kaggle/input", str(ROOT / "data/kaggle_input"))
ns = dict(Path=Path, td=td)
exec(compile(setup, "official_setup", "exec"), ns)
build = ns["_official_graph_from_processed"]
evaluate = ns["_official_evaluate"]
per_sample = ns["_official_per_sample_metrics"]
summarise = ns["_official_summarise"]
candidate = pd.read_csv(source / "candidate_annotation_audit.csv", usecols=[
    "stem", "fork_id", "daughter1_id", "daughter2_id", "offset_max_score", "offset_top2_mean_score"])
stems = sorted(candidate.stem.unique())
expected = pd.read_csv(ROOT / "local_runs/EXT0006/working/official_validator_results.csv").set_index("stem")
rows = []
configs = [("base", None, 0)] + [(f"{score}_top{k}", score, k)
    for score in ("offset_max_score", "offset_top2_mean_score") for k in (25, 100, 250, 500)]
selected = {label: (candidate.iloc[:0] if not score else candidate.sort_values(score, ascending=False, kind="stable").head(k))
            for label, score, k in configs}
for stem in stems:
    node_frame = pd.read_csv(source / f"{stem}_nodes.csv")
    edge_frame = pd.read_csv(source / f"{stem}_edges.csv")
    nodes = {int(r.node_id): dict(node_id=int(r.node_id), t=int(r.t), z=float(r.z), y=float(r.y), x=float(r.x))
             for r in node_frame.itertuples()}
    baseline = {(int(r.source_id), int(r.target_id)) for r in edge_frame.itertuples()}
    gt = td.graph.IndexedRXGraph.from_geff(train / f"{stem}.geff")
    gt = gt[0] if isinstance(gt, tuple) else gt
    total = float((ns["_GeffMetadata"].read(train / f"{stem}.geff").extra or {})["estimated_number_of_nodes"])
    for label, score, k in configs:
        links = set(baseline)
        incoming = {}
        outgoing = {}
        for s, d in links:
            incoming[d] = s
            outgoing.setdefault(s, set()).add(d)
        applied = 0
        for row in selected[label].loc[selected[label].stem.eq(stem)].itertuples():
            p, a, b = int(row.fork_id), int(row.daughter1_id), int(row.daughter2_id)
            if p not in nodes or a not in nodes or b not in nodes or a == b:
                continue
            if nodes[a]["t"] != nodes[p]["t"]+1 or nodes[b]["t"] != nodes[p]["t"]+1:
                continue
            current = outgoing.get(p, set())
            # Preserve an anchored existing branch and two persistent daughters.
            if p not in incoming or len(current) != 1 or not current <= {a,b}:
                continue
            if not outgoing.get(a) or not outgoing.get(b):
                continue
            new_child = b if a in current else a
            old_parent = incoming.get(new_child)
            if old_parent is not None:
                if len(outgoing.get(old_parent,set())) != 1:
                    continue
                links.remove((old_parent,new_child))
                outgoing[old_parent].remove(new_child)
            links.add((p,new_child))
            outgoing[p].add(new_child)
            incoming[new_child] = p
            applied += 1
        pred = build(nodes, [dict(source_id=s,target_id=d) for s,d in sorted(links)])
        result = evaluate(pred, gt, scale=(1.625,0.40625,0.40625), max_distance=7.0)
        if label == "base":
            for key in ("edge_tp","edge_fp","edge_fn","division_tp","division_fp","division_fn","num_pred_nodes"):
                assert getattr(result,key) == int(expected.loc[stem,key]), (stem,key)
        row = per_sample(result,total,ns["_official_node_recall"](pred,gt))
        row.update(stem=stem,embryo=stem.split("_")[0],config=label,edits=applied)
        rows.append(row)
        print("SCORED",stem,label,"edits",applied,"edge",result.edge_tp,result.edge_fp,result.edge_fn,
              "division",result.division_tp,result.division_fp,result.division_fn,flush=True)
    pd.DataFrame(rows).to_csv(output / "samples.csv",index=False)
summaries = []
for label, score, k in configs:
    group = [r for r in rows if r["config"]==label]
    summaries.append(dict(config=label,edits=sum(r["edits"] for r in group),**summarise(group)))
    for embryo in sorted({r["embryo"] for r in group}):
        subset = [r for r in group if r["embryo"]==embryo]
        summaries.append(dict(config=label,embryo=embryo,edits=sum(r["edits"] for r in subset),**summarise(subset)))
(output / "summary.json").write_text(json.dumps(summaries,indent=2) + "\n")
print(json.dumps([s for s in summaries if "embryo" not in s],indent=2),flush=True)
