"""E0071: live competition inference followed by the frozen E0067 repair policy."""
import copy,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dirs=root / "research_members/naveen/experiments"
base=json.loads((dirs / "E0053_harmonic_frontier/biohub-e0053-harmonic-frontier.ipynb").read_text())
repair=json.loads((dirs / "E0067_division_repair_submission/biohub-e0067-division-repair-submission.ipynb").read_text())
parts=["".join(c["source"]) for c in base["cells"][:6]]
parts[0]+='\nos.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"]="5.5"\n'
parts.append("".join(repair["cells"][4]["source"]))
setup="".join(repair["cells"][5]["source"])
setup=setup.replace('production_csv = unique_input("submission.csv")','production_csv = WORKING_DIR / "submission.csv"')
setup=setup.replace('test_stems = sorted(baseline_frame.dataset.unique())','''all_test_stems = sorted(baseline_frame.dataset.unique())
known_embryos = {p.stem.removeprefix("division_crop_heldout_") for p in study.glob("division_crop_heldout_*.pt")}
test_stems = [s for s in all_test_stems if s.split("_")[0] in known_embryos]
unsupported_stems = sorted(set(all_test_stems)-set(test_stems))
assert set(all_test_stems)=={p.name[:-5] for p in (COMP_DIR / "test").glob("*.zarr")}
print("LIVE_INPUT_ROUTING",dict(repair=test_stems,baseline_only=unsupported_stems),flush=True)
''')
parts.append(setup)
work=["".join(c["source"]) for c in repair["cells"][6:11]]
runtime=work[-1]
a=runtime.index('replay_rows = []')
b=runtime.index('# Freeze the validation-derived',a)
runtime=runtime[:a]+'''# Diagnostic replay was completed in E0067; production never requires train GT.
saved_summaries=json.loads((study / "ranked_repairs/summary.json").read_text())
summary=next(r for r in saved_summaries if r["config"]=="offset_max_score_top100" and "embryo" not in r)
assert abs(summary["score"]-0.9499248688395877)<1e-10
''' +runtime[b:]
runtime=runtime.replace('frame = pd.DataFrame(out_rows)','''# Unknown embryo identities retain live baseline predictions, not cached CSVs.
for r in baseline_frame[baseline_frame.dataset.isin(unsupported_stems)].drop(columns=["id"]).to_dict("records"):
    out_rows.append(r)
frame = pd.DataFrame(out_rows)''')
work[-1]=runtime
for source in work:
    # With no supported embryos the already-generated live baseline is retained.
    parts.append('if test_stems:\n'+''.join('    '+line+'\n' for line in source.splitlines()))
parts.append("".join(repair["cells"][11]["source"]))
folder=dirs / "E0071_live_repair_submission";folder.mkdir(exist_ok=True)
slug="biohub-e0071-live-repair-submission"
cells=[]
for i,s in enumerate(parts):
    compile(s,f"E0071:{i}","exec")
    cells.append(dict(cell_type="code",id=f"e0071-{i:02d}",metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
(folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=base["metadata"],cells=cells),indent=1)+"\n")
meta=json.loads((dirs / "E0067_division_repair_submission/kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}",title="Biohub E0071 Live Repair Submission",code_file=f"{slug}.ipynb")
meta["kernel_sources"]=["naveenlx111249971939/biohub-e0065-kaggle-division-ranker"]
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(folder / "provenance.json").write_text(json.dumps(dict(parent="E0053 live inference plus E0067 frozen repairs",
    fix="Never load cached test submission; discover actual test inputs; unsupported embryo identities retain live baseline",
    validation="Kaggle public full run plus actual-input graph audit; hidden rerun remains unverified until submission",
    compute="Kaggle T4 only"),indent=2)+"\n")
print(folder)
