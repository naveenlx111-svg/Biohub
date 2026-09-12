"""E0081 live strong-edge candidate; E0082 eight additional diagnostic movies."""
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ROOT / "research_members/naveen/experiments"
anchor = json.loads((ROOT / "local_runs/anchor0947/biohub-lf-dctta-v020.ipynb").read_text())
study_dir = DIRS / "E0079_strong_edge_audit"
study = json.loads((study_dir / "biohub-e0079-strong-edge-audit.ipynb").read_text())
strong_defs = "".join(study["cells"][5]["source"])
strong_fn = next(n for n in ast.parse(strong_defs).body if isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph")
strong_source = ast.get_source_segment(strong_defs, strong_fn)
base_parts = ["".join(c["source"]) for c in anchor["cells"][:6]]
base_parts[0] += '\nos.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\nos.environ["BIOHUB_STRONG_EDGE_THRESHOLD"] = "0.55"\n'
meta_template = json.loads((study_dir / "kernel-metadata.json").read_text())

def write(number, name, title, parts, diagnostic):
    folder = DIRS / f"E{number:04d}_{name}"
    folder.mkdir(exist_ok=True)
    slug = f"biohub-e{number:04d}-{name.replace('_', '-')}"
    cells = []
    for i, source in enumerate(parts):
        compile(source, f"E{number}:{i}", "exec")
        cells.append(dict(cell_type="code", id=f"e{number:04d}-{i:02d}", metadata={}, execution_count=None, outputs=[], source=source.splitlines(True)))
    (folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=anchor["metadata"]), indent=1) + "\n")
    meta = copy.deepcopy(meta_template)
    meta.update(id=f"naveenlx111249971939/{slug}", title=title, code_file=f"{slug}.ipynb", kernel_sources=[])
    if not diagnostic:
        meta["dataset_sources"] = [s for s in meta["dataset_sources"] if "official-scorer" not in s]
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "provenance.json").write_text(json.dumps(dict(
        parent="0.947 public anchor and E0079 preserve_strong055",
        evidence="E0079 diagnostic 0.943402106 to 0.971036663; both embryos improve; division4TP1FP8FN",
        change="Freeze strong-edge preservation threshold0.55 and tight55; original anchor weights and inference",
        compute="Kaggle T4 only",
        purpose="Eight additional movies chosen without division labels; diagnostic training overlap unresolved" if diagnostic else "Live actual-test inference; no cached test CSV or GT-based production selection",
        public_score="Unverified; diagnostic0.971 is not a public0.97 result"), indent=2) + "\n")
    print(folder)

# Production uses the same filter that was evaluated, then the original serializer.
parts = base_parts.copy()
old_fn = next(n for n in ast.parse(parts[5]).body if isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph")
old_source = ast.get_source_segment(parts[5], old_fn)
assert parts[5].count(old_source) == 1
parts[5] = parts[5].replace(old_source, strong_source)
audit = (ROOT / "tools/audit_submission.py").read_text()
audit = audit[:audit.index("parser = argparse.ArgumentParser()")] + '''
from types import SimpleNamespace
args = SimpleNamespace(submission=SUBMISSION_PATH, competition=COMP_DIR, output=WORKING_DIR / "submission_audit.json")
''' + audit[audit.index("frame = pd.read_csv(args.submission)"):]
parts.append(audit + '\nprint("E0081_VALIDATED", flush=True)\n')
write(81, "live_strong_submission", "Biohub E0081 Live Strong Submission", parts, False)

# Additional panel: run unchanged detector once, score two postprocessors.
parts = base_parts.copy()
parts[4] = 'ACTUAL_TEST_DIR = TEST_DIR\nTEST_DIR = COMP_DIR / "train"\n' + parts[4]
exclusions = json.loads((ROOT / "local_runs/anchor0947/output/ppsweep_selected.json").read_text())["held_out_stems"]
selector = '''    import hashlib
    old_panel = set(EXCLUSIONS)
    actual_test = {p.name[:-5] for p in ACTUAL_TEST_DIR.glob("*.zarr")}
    candidates = [s for s in stems if s not in old_panel and s not in actual_test and (TEST_DIR / f"{s}.geff").exists()]
    selected = []
    for embryo in ("44b6", "6bba"):
        group = [s for s in candidates if s.startswith(embryo + "_")]
        group.sort(key=lambda s: hashlib.sha256(("E0082-fixed-panel:" + s).encode()).hexdigest())
        assert len(group) >= 4, (embryo, len(group))
        selected.extend(group[:4])
    assert len(selected) == 8 and not set(selected) & old_panel
    (WORKING_DIR / "expanded_panel.json").write_text(json.dumps({"selected": selected, "excluded_original": sorted(old_panel), "selection": "Fixed SHA256 order; no GT contents inspected"}, indent=2))
    return sorted(selected)'''.replace("EXCLUSIONS", repr(exclusions))
assert parts[4].count("    return stems\n") == 1
parts[4] = parts[4].replace("    return stems\n", selector + "\n")
parts[5] = parts[5].split("def write_test_submission(")[0]
parts.append("".join(study["cells"][4]["source"]))
parts.append(strong_defs)
evaluation = "".join(study["cells"][-1]["source"])
a = evaluation.index("reference_files =")
b = evaluation.index("configs =", a)
evaluation = evaluation[:a] + '''stems = test_stems
matches = list((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0"))
assert len(matches) == 1, matches
raw_root = matches[0]
''' + evaluation[b:]
evaluation = evaluation.replace('configs = {"anchor0947": None, "preserve_strong055": .55, "preserve_strong080": .80, "preserve_strong095": .95}', 'configs = {"anchor0947": None, "preserve_strong055": .55}')
a = evaluation.index('        if config == "anchor0947":')
b = evaluation.index('        row = _official_per_sample_metrics', a)
evaluation = evaluation[:a] + evaluation[b:]
evaluation = evaluation.replace("E0079_COMPLETE", "E0082_COMPLETE")
parts.append(evaluation)
write(82, "expanded_strong_audit", "Biohub E0082 Expanded Strong Audit", parts, True)
