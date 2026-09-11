"""Public frontier reproduction plus an independent local GPU gate study."""
import copy,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dirs=root / "research_members/naveen/experiments"
source=root / "local_runs/frontier_20260911/trackcells"
nb=json.loads(next(source.glob("*.ipynb")).read_text())
base=json.loads((dirs / "E0053_harmonic_frontier/biohub-e0053-harmonic-frontier.ipynb").read_text())
nb["cells"].append(copy.deepcopy(base["cells"][-1]))
slug="biohub-e0073-public-trackcells-audit"
folder=dirs / "E0073_public_trackcells_audit";folder.mkdir(exist_ok=True)
for i,c in enumerate(nb["cells"]):
    c["id"]=f"e0073-{i:02d}";c["metadata"]={}
    if c["cell_type"]=="code":
        c.update(execution_count=None,outputs=[])
        compile("".join(c["source"]),f"E0073:{i}","exec")
nb["nbformat_minor"]=5
(folder / f"{slug}.ipynb").write_text(json.dumps(nb,indent=1)+"\n")
meta=json.loads((source / "kernel-metadata.json").read_text())
meta.pop("id_no",None) # Never address the source author's existing kernel ID.
meta.pop("docker_image",None) # Use the supported runtime for our own kernel.
meta.update(id=f"naveenlx111249971939/{slug}",title="Biohub E0073 Public Trackcells Audit",code_file=f"{slug}.ipynb",is_private=True,machine_shape="NvidiaTeslaT4")
scorer="dalloliogm/biohub-official-scorer-patched"
if scorer not in meta["dataset_sources"]: meta["dataset_sources"].append(scorer)
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(folder / "provenance.json").write_text(json.dumps(dict(source="anhadmahajan06/biohub-track-your-cells",
    score_claim="0.947-0.950+ in source markdown; not independently verified",
    method="Original cells preserved; appended pinned official audit",compute="Kaggle T4"),indent=2)+"\n")
print(folder)

local=json.loads((dirs / "E0066_cached_deepcenter_tta/biohub-e0066-cached-deepcenter-tta.ipynb").read_text())
local["cells"].pop(2) # Dependencies already installed locally; no environment mutation.
working=root / "local_runs/E0074/working";working.mkdir(parents=True,exist_ok=True)
for i,c in enumerate(local["cells"]):
    s="".join(c["source"])
    s=s.replace('/kaggle/input',str(root / "data/kaggle_input")).replace('/kaggle/working',str(working))
    old=f'_cached = list(Path("{root / "data/kaggle_input"}").rglob("frontier_official_samples.csv"))'
    s=s.replace(old,f'_cached = [Path("{root / "local_runs/E0053/kaggle/frontier_official_samples.csv"}")]')
    s=s.replace('cached_root = _cached[0].parent',f'cached_root = Path("{root / "local_runs/frontier_20260909/harmonic/output"}")')
    if 'os.environ["BIOHUB_DEEPCENTER_TTA"] = "1"' in s:
        s='DEEPCENTER_SAFE_DIV_THRESHOLD = 0.20\n'+s
        s=s.replace('public_eight_view','public_eight_view_threshold020').replace('true_d4_eight_view','true_d4_eight_view_threshold020')
    c.update(id=f"e0074-{i:02d}",source=s.splitlines(True),outputs=[],execution_count=None)
    compile(s,f"E0074:{i}","exec")
path=root / "local_runs/E0074/e0074-local-deepcenter020.ipynb"
path.write_text(json.dumps(local,indent=1)+"\n")
print(path)
