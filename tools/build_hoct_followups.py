"""Build E0077 live HOCT candidate and E0078 cached, label-free fork fusion audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ROOT / "research_members/naveen/experiments"
anchor = json.loads((ROOT / "local_runs/anchor0947/biohub-lf-dctta-v020.ipynb").read_text())
prior_dir = DIRS / "E0076_hoct_consensus_audit"
prior = json.loads((prior_dir / "biohub-e0076-hoct-consensus-audit.ipynb").read_text())
hoct = json.loads((ROOT / "local_runs/frontier_20260911/hoct/biohub-lf-hoctveto-div.ipynb").read_text())
base_meta = json.loads((prior_dir / "kernel-metadata.json").read_text())


def write(number, name, title, parts, meta, provenance):
    folder = DIRS / f"E{number:04d}_{name}"
    folder.mkdir(exist_ok=True)
    slug = f"biohub-e{number:04d}-{name.replace('_', '-')}"
    cells = []
    for i, source in enumerate(parts):
        compile(source, f"E{number}:{i}", "exec")
        cells.append(dict(cell_type="code", id=f"e{number:04d}-{i:02d}", metadata={}, execution_count=None, outputs=[], source=source.splitlines(True)))
    (folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=anchor["metadata"], cells=cells), indent=1) + "\n")
    meta.update(id=f"naveenlx111249971939/{slug}", title=title, code_file=f"{slug}.ipynb")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(folder)


# Live anchor inference, frozen tight55 selected by the already-scored parent.
# No cached public-test graphs and no new validation selection in production.
parts = ["".join(anchor["cells"][i]["source"]) for i in range(6)]
parts[0] += '\nos.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\nos.environ["BIOHUB_HOCT_VETO"] = "1"\n'
parts.append('''
import hashlib
import shutil
shutil.copy2(SUBMISSION_PATH, WORKING_DIR / "anchor_submission.csv")
print("LIVE_ANCHOR_SHA256", hashlib.sha256(SUBMISSION_PATH.read_bytes()).hexdigest(), flush=True)
''')
parts.append("".join(hoct["cells"][6]["source"]))
audit = (ROOT / "tools/audit_submission.py").read_text()
audit = audit[:audit.index("parser = argparse.ArgumentParser()")] + '''
from types import SimpleNamespace
args = SimpleNamespace(submission=SUBMISSION_PATH, competition=COMP_DIR, output=WORKING_DIR / "submission_audit.json")
''' + audit[audit.index("frame = pd.read_csv(args.submission)"):]
parts.append(audit + '''
baseline = pd.read_csv(WORKING_DIR / "anchor_submission.csv")
columns = ["dataset", "node_id", "t", "z", "y", "x"]
assert frame[frame.row_type.eq("node")][columns].reset_index(drop=True).equals(baseline[baseline.row_type.eq("node")][columns].reset_index(drop=True))
before = set(map(tuple, baseline[baseline.row_type.eq("edge")][["dataset", "source_id", "target_id"]].to_numpy()))
after = set(map(tuple, frame[frame.row_type.eq("edge")][["dataset", "source_id", "target_id"]].to_numpy()))
assert after <= before
print("E0077_VALIDATED", dict(removed_edges=len(before-after), nodes_unchanged=True), flush=True)
''')
meta = dict(base_meta)
meta["kernel_sources"] = []
meta["dataset_sources"] = [s for s in meta["dataset_sources"] if "official-scorer" not in s]
write(77, "live_hoct_submission", "Biohub E0077 Live HOCT Submission", parts, meta, dict(
    parent="0.947 public anchor submission56159060; tight55 frozen",
    change="HOCT ordinary-edge consensus veto only; protect all existing division edges",
    evidence="E0076 official reused holdout 0.943402106 to 0.944912830; embryo44b6 -0.000073306 and embryo6bba +0.002075393",
    risk="Small aggregate gain with minor one-embryo regression; public improvement unproven; baseline preserved",
    production="Actual mounted test inference; no cached test outputs; no GT-dependent decisions; GPU Kaggle T4",
    submission="Candidate only; require successful run and graph audit before submitting"))

# Reuse the independently generated HOCT predictions; no new GPU inference.
parts = ["".join(prior["cells"][i]["source"]) for i in (1, 2)]
parts.append('import tracksdata as td\nimport numpy as np\nVOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n' + "".join(prior["cells"][4]["source"]))
parts.append((ROOT / "tools/hoct_graph_fusion.py").read_text())
parts.append('''
import torch
torch.set_num_threads(4)
TRAIN_DIR = COMP_DIR / "train"
cached = list(Path("/kaggle/input").rglob("hoct_official_samples.csv"))
assert len(cached) == 1, cached
source_root = cached[0].parent
expected = pd.read_csv(cached[0])
stems = sorted(expected.stem.unique())
rows, edit_rows = [], []
configs = ("anchor0947", "ordinary_veto", "hoct_full_graph", "veto_orphan_forks", "veto_reparent_forks")
def graph_from_geff(path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result
def persist():
    pd.DataFrame(rows).to_csv(WORKING_DIR / "fusion_official_samples.csv", index=False)
    pd.DataFrame(edit_rows).to_csv(WORKING_DIR / "fusion_edits.csv", index=False)
    summaries = []
    for config in configs:
        subset = [r for r in rows if r["config"] == config]
        if not subset:
            continue
        summaries.append(dict(config=config, **_official_summarise(subset)))
        for embryo in sorted({r["stem"].split("_")[0] for r in subset}):
            subset_embryo = [r for r in subset if r["stem"].startswith(embryo + "_")]
            summaries.append(dict(config=config, embryo=embryo, **_official_summarise(subset_embryo)))
    (WORKING_DIR / "fusion_official_summary.json").write_text(json.dumps(summaries, indent=2))
for stem in stems:
    records = pd.read_csv(source_root / f"{stem}_anchor_nodes.csv", float_precision="round_trip").to_dict("records")
    nodes = {int(r["node_id"]): r for r in records}
    edges = pd.read_csv(source_root / f"{stem}_anchor_edges.csv", float_precision="round_trip").to_dict("records")
    pairs = set(map(tuple, pd.read_csv(source_root / f"{stem}_hoct_pairs.csv").to_numpy(dtype=int)))
    degree = Counter(int(e["source_id"]) for e in edges)
    veto = [e for e in edges if degree[int(e["source_id"])] == 2 or (int(e["source_id"]), int(e["target_id"])) in pairs]
    variants = {"anchor0947": edges, "ordinary_veto": veto, "hoct_full_graph": [dict(source_id=s, target_id=t) for s,t in sorted(pairs)]}
    for config, allow in (("veto_orphan_forks", False), ("veto_reparent_forks", True)):
        variants[config], edits = fuse_hoct_forks(nodes, veto, pairs, allow_reparent=allow)
        edit_rows.extend(dict(stem=stem, config=config, **e) for e in edits)
    gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
    metadata = _GeffMetadata.read(TRAIN_DIR / f"{stem}.geff")
    for config, selected in variants.items():
        assert max(Counter(int(e["target_id"]) for e in selected).values(), default=0) <= 1
        assert max(Counter(int(e["source_id"]) for e in selected).values(), default=0) <= 2
        pred = _official_graph_from_processed(nodes, selected)
        result = _official_evaluate(pred, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
        if config in ("anchor0947", "ordinary_veto"):
            label = "anchor0947" if config == "anchor0947" else "veto_ordinary_only"
            reference = expected[(expected.stem == stem) & (expected.config == label)].iloc[0]
            for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes"):
                assert getattr(result, key) == int(reference[key]), (stem, config, key)
        row = _official_per_sample_metrics(result, float((metadata.extra or {})["estimated_number_of_nodes"]), _official_node_recall(pred, gt))
        row.update(stem=stem, config=config, num_edges=len(selected))
        rows.append(row)
        persist()
        print("FUSION_RESULT", row, flush=True)
assert len(rows) == len(stems) * len(configs)
print("E0078_COMPLETE", flush=True)
''')
meta = dict(base_meta)
meta.update(kernel_sources=["naveenlx111249971939/biohub-e0076-hoct-consensus-audit"], enable_gpu=False)
meta.pop("machine_shape", None)
write(78, "hoct_fork_fusion", "Biohub E0078 HOCT Fork Fusion", parts, meta, dict(
    parent="E0076 cached final anchor nodes and independent HOCT edges",
    configs=["anchor0947", "ordinary_veto", "hoct_full_graph", "veto_orphan_forks", "veto_reparent_forks"],
    checks="Exact official anchor and veto per-sample count replay; same node set; one parent and at most two children",
    mechanism="Use HOCT-selected divisions; require one existing agreed daughter; either orphan-only or reparent an ordinary incoming link",
    compute="Kaggle CPU cached graph scoring; no new local inference",
    caveat="Reused eight-movie diagnostic; not independent CV or leaderboard; no submission"))
