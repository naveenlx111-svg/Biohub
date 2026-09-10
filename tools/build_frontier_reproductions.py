"""Build private, source-preserving frontier reproductions with an official audit."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "naveenlx111249971939"
FRONTIER = ROOT / "local_runs/frontier_20260909"
old = json.loads((ROOT / "research_members/naveen/experiments/E0033_lb0933_official_12/e0033-lb0933-official-12.ipynb").read_text())
official = "".join(old["cells"][9]["source"])
setup = official.split("official_rows = []")[0]
audit = '''
# Audit frozen base and the source-selected configuration independently.
# The public notebook's legacy sweep is preserved for exact reproduction.
# Its proxy score is not used as official validation evidence.
import copy
all_official_rows = []
all_official_summaries = []
for audit_label, audit_config in [("base", {}), ("source_selected", selected_config)]:
    official_rows = []
    saved = pp_apply(audit_config)
    real_test_dir = TEST_DIR
    globals()["TEST_DIR"] = TRAIN_DIR
    try:
        for stem in val_stems:
            raw_nodes, raw_edges = VAL_RAW_GRAPHS[stem]
            nodes, edges, stats = filter_output_graph(
                copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            pred = _official_graph_from_processed(nodes, edges)
            gt = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
            result = _official_evaluate(pred, gt, scale=tuple(VOXEL_SCALE_UM),
                                       max_distance=VALIDATOR_MATCH_RADIUS_UM)
            recall = _official_node_recall(pred, gt) if pred.num_nodes() and pred.num_edges() else 0.0
            meta = _GeffMetadata.read(TRAIN_DIR / f"{stem}.geff")
            total = float((meta.extra or {})["estimated_number_of_nodes"])
            row = _official_per_sample_metrics(result, total, recall)
            row.update(stem=stem, config=audit_label, embryo=stem.split("_")[0])
            official_rows.append(row)
            print("OFFICIAL_SAMPLE", audit_label, stem, row, flush=True)
        summary = _official_summarise(official_rows)
        all_official_summaries.append(dict(config=audit_label, **summary))
        for embryo in sorted({r["embryo"] for r in official_rows}):
            group = [r for r in official_rows if r["embryo"] == embryo]
            all_official_summaries.append(dict(config=audit_label, embryo=embryo,
                                              **_official_summarise(group)))
        all_official_rows.extend(official_rows)
        print("OFFICIAL_SUMMARY", audit_label, json.dumps(summary), flush=True)
        pd.DataFrame(all_official_rows).to_csv(WORKING_DIR / "frontier_official_samples.csv", index=False)
        (WORKING_DIR / "frontier_official_summary.json").write_text(
            json.dumps(all_official_summaries, indent=2, default=str))
    finally:
        globals()["TEST_DIR"] = real_test_dir
        pp_restore(saved)
print("Audit panel is diagnostic; detector training independence is not established.")
'''
for label, exp, slug in [
    ("harmonic", "E0053", "biohub-e0053-harmonic-frontier"),
    ("lineage", "E0054", "biohub-e0054-lineage-frontier"),
]:
    folder = FRONTIER / label
    source = next(folder.glob("*.ipynb"))
    nb = json.loads(source.read_text())
    nb = copy.deepcopy(nb)
    for cell in nb["cells"]:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nb["cells"].append(dict(cell_type="code", metadata={}, execution_count=None,
                            outputs=[], source=(setup + audit).splitlines(True)))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{exp}:cell{i}", "exec")
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    nb["nbformat_minor"] = 5
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"{exp.lower()}-{i:02d}"
    dest = ROOT / "research_members/naveen/experiments" / f"{exp}_{label}_frontier"
    dest.mkdir(parents=True, exist_ok=True)
    code_file = slug + ".ipynb"
    (dest / code_file).write_text(json.dumps(nb, indent=1) + "\n")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    meta = {k: v for k, v in meta.items() if k not in ("id_no", "docker_image", "machine_shape")}
    meta.update(id=f"{OWNER}/{slug}-audit", title=f"Biohub {exp} {label.title()} Frontier Audit",
                code_file=code_file, is_private=True)
    meta["dataset_sources"].append("dalloliogm/biohub-official-scorer-patched")
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (dest / "provenance.json").write_text(json.dumps({
        "source": json.loads((folder / "kernel-metadata.json").read_text())["id"],
        "download_date": "2026-09-09", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "changes": "Original executable cells preserved; appended hash-verified official scorer audit.",
        "validation": "Same eight diagnostic movies selected by public notebook; not independent detector OOF."
    }, indent=2) + "\n")
    print(dest.relative_to(ROOT))
