"""E0067: deploy E0065 frozen OOF crop ranking on E0053 test graphs on Kaggle."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = root / "research_members/naveen/experiments/E0065_kaggle_division_ranker"
n = json.loads((parent / "biohub-e0065-kaggle-division-ranker.ipynb").read_text())
parts = ["".join(c["source"]) for c in n["cells"][:5]]
parts.append('''
import torch, torch.nn as nn, zarr
from sklearn.ensemble import HistGradientBoostingClassifier
def unique_input(name):
    paths = list(Path("/kaggle/input").rglob(name))
    assert len(paths) == 1, (name, paths)
    return paths[0]
study = unique_input("offset_pool_transfer_oof.csv").parent
production_csv = unique_input("submission.csv")
baseline_frame = pd.read_csv(production_csv)
test_stems = sorted(baseline_frame.dataset.unique())
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = COMP_DIR / "test"
official_graph_inputs = {}
for stem, group in baseline_frame.groupby("dataset", sort=True):
    nodes = {int(r.node_id): dict(node_id=int(r.node_id),t=int(r.t),z=float(r.z),y=float(r.y),x=float(r.x))
             for r in group[group.row_type.eq("node")].itertuples()}
    edges = [dict(source_id=int(r.source_id),target_id=int(r.target_id))
             for r in group[group.row_type.eq("edge")].itertuples()]
    official_graph_inputs[stem] = (nodes,edges)
''')
# Reuse the proposal enumeration without accessing test GT or matching labels.
proposal = "".join(n["cells"][9]["source"])
a = proposal.index('    _gt = graph_from_geff(')
b = proposal.index('    _node_rows =', a)
proposal = proposal[:a] + proposal[b:]
a = proposal.index('    # Positive triples')
b = proposal.index('    _proposed_triples =', a)
proposal = proposal[:a] + '    _positive_triples = set()\n' + proposal[b:]
proposal = proposal[:proposal.index('    for (_event_stem, _divider)')]
proposal = proposal.replace('for _stem in val_stems:', 'for _stem in test_stems:')
proposal += '\n_candidate_df = pd.DataFrame(fork_candidate_rows)\n'
parts.append(proposal)
parts.append('''
pair_features = ["d1_um","d2_um","distance_sum_um","distance_asymmetry_um",
    "sister_um","midpoint_um","daughter_cosine","fork_outdegree",
    "daughter1_indegree","daughter2_indegree","edge1_exists","edge2_exists"]
def gate(df):
    return df[(df.distance_sum_um <= 14.0) & (df.sister_um >= 4.0)
              & (df.midpoint_um <= 5.0) & (df.daughter_cosine <= -0.30)].copy()
# Preserve the original float64 geometry through the CSV round trip. The default
# parser can perturb values at histogram bin boundaries in the pair classifier.
training = gate(pd.read_csv(study / "fork_candidates.csv",float_precision="round_trip"))
saved_parents = pd.read_csv(study / "frozen_gate_parents.csv")
test_pairs = gate(_candidate_df)
parent_parts = []
for heldout in sorted(test_pairs.embryo.unique()):
    train = training[training.embryo.ne(heldout)]
    weights = np.ones(len(train))
    weights[train.label.to_numpy() == 1] = int((train.label==0).sum()) / max(int(train.label.sum()),1)
    model = HistGradientBoostingClassifier(learning_rate=0.05,max_iter=250,
        max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=2.0,random_state=20260830)
    model.fit(train[pair_features],train.label,sample_weight=weights)
    # Replay pair selection against the saved held-out diagnostic rows.
    replay = training[training.embryo.eq(heldout)].copy()
    replay["score"] = model.predict_proba(replay[pair_features])[:,1]
    replay = replay[replay.groupby(["stem","fork_id"]).score.rank(method="first",ascending=False).eq(1)]
    cols = ["stem","fork_id","daughter1_id","daughter2_id"]
    want = saved_parents[saved_parents.embryo.eq(heldout)]
    got_set = set(map(tuple,replay[cols].to_numpy()))
    want_set = set(map(tuple,want[cols].to_numpy()))
    report = dict(embryo=heldout,replayed=len(got_set),expected=len(want_set),
                  only_replayed=len(got_set-want_set),only_expected=len(want_set-got_set))
    (WORKING_DIR / f"pair_replay_{heldout}.json").write_text(json.dumps(report,indent=2))
    print("PAIR_REPLAY",report,flush=True)
    assert got_set == want_set, report
    test = test_pairs[test_pairs.embryo.eq(heldout)].copy()
    test["score"] = model.predict_proba(test[pair_features])[:,1]
    test = test[test.groupby(["stem","fork_id"]).score.rank(method="first",ascending=False).eq(1)]
    parent_parts.append(test)
_parents = pd.concat(parent_parts).sort_values(["stem","t","fork_id"]).reset_index(drop=True)
_parents.to_csv(WORKING_DIR / "test_parents.csv",index=False)
''')
crop = "".join(n["cells"][11]["source"])
network = crop[crop.index('class _DivisionCropNet'):crop.index('_crop_models = {}')]
parts.append(network + '''
_crop_models = {}
for embryo in sorted(_parents.embryo.unique()):
    checkpoint = torch.load(study / f"division_crop_heldout_{embryo}.pt",map_location="cpu",weights_only=False)
    assert checkpoint["heldout_embryo"] == embryo
    assert embryo not in checkpoint["training_embryos"]
    assert checkpoint["crop_mode"] == "raw"
    model = _DivisionCropNet(3)
    model.load_state_dict(checkpoint["model_state"])
    _crop_models[embryo] = model.eval()
''')
offset = "".join(n["cells"][13]["source"])
offset = offset[:offset.index('_offset_metric_rows')]
# Cut at the metrics setup, retaining complete inference and dataframe creation.
offset = offset.replace('TRAIN_DIR /', 'TEST_DIR /')
parts.append(offset)
runtime = (root / "tools/division_repair_deploy_runtime.py").read_text()
parts.append(runtime)
audit = (root / "tools/audit_submission.py").read_text()
parts.append('import sys\nsys.argv = ["audit_submission",str(WORKING_DIR / "submission.csv"),"--competition",str(COMP_DIR),"--output",str(WORKING_DIR / "submission_audit.json")]\n' + audit)
folder = root / "research_members/naveen/experiments/E0067_division_repair_submission"
folder.mkdir(exist_ok=True)
cells = []
for i, source in enumerate(parts):
    compile(source,f"E0067:{i}","exec")
    cells.append(dict(cell_type="code",id=f"e0067-{i:02d}",metadata={},execution_count=None,outputs=[],source=source.splitlines(True)))
nb = dict(nbformat=4,nbformat_minor=5,metadata=n["metadata"],cells=cells)
slug = "biohub-e0067-division-repair-submission"
(folder / f"{slug}.ipynb").write_text(json.dumps(nb,indent=1)+"\n")
meta = json.loads((parent / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}",title="Biohub E0067 Division Repair Submission",code_file=f"{slug}.ipynb")
meta["kernel_sources"].append("naveenlx111249971939/biohub-e0065-kaggle-division-ranker")
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
print(folder)
