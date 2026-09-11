"""E0075: train-only spatial position jitter, keeping inference and selection fixed."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parent=root / "research_members/naveen/experiments/E0065_kaggle_division_ranker"
n=json.loads((parent / "biohub-e0065-kaggle-division-ranker.ipynb").read_text())
parts=["".join(c["source"]) for c in n["cells"][:5]]
parts.append('''
import copy
torch.set_num_threads(4)
TRAIN_DIR=COMP_DIR / "train"
paths=list(Path("/kaggle/input").rglob("frozen_gate_parents.csv"))
assert len(paths)==1,paths
study=paths[0].parent
_parents=pd.read_csv(paths[0],float_precision="round_trip")
val_stems=sorted(_parents.stem.unique())
official_graph_inputs={}
for stem in val_stems:
    nf=pd.read_csv(study / f"{stem}_nodes.csv")
    ef=pd.read_csv(study / f"{stem}_edges.csv")
    nodes={int(r.node_id):dict(node_id=int(r.node_id),t=int(r.t),z=float(r.z),y=float(r.y),x=float(r.x)) for r in nf.itertuples()}
    edges=[dict(source_id=int(r.source_id),target_id=int(r.target_id)) for r in ef.itertuples()]
    official_graph_inputs[stem]=(nodes,edges)
    nf.to_csv(WORKING_DIR / f"{stem}_nodes.csv",index=False)
    ef.to_csv(WORKING_DIR / f"{stem}_edges.csv",index=False)
pd.read_csv(study / "official_validator_results.csv").to_csv(WORKING_DIR / "official_validator_results.csv",index=False)
''')
crop="".join(n["cells"][11]["source"])
needle='            # Spatial jitter and flips preserve the temporal channel order.'
assert crop.count(needle)==1
crop=crop.replace('_crop_seed = 20260830','_crop_seed = 20260830\n_jitter_rng = np.random.default_rng(20260911)')
crop=crop.replace(needle,'''            # Shared spatial shift across all temporal channels; train rows only.
            padded=np.pad(_x,((0,0),(0,0),(2,2),(8,8),(8,8)),mode="edge")
            for ji in range(len(_x)):
                dz,dy,dx=int(_jitter_rng.integers(-2,3)),int(_jitter_rng.integers(-8,9)),int(_jitter_rng.integers(-8,9))
                _x[ji]=padded[ji,:,2+dz:2+dz+_x.shape[2],8+dy:8+dy+_x.shape[3],8+dx:8+dx+_x.shape[4]]
            # Existing flips and sample RNG remain unchanged.''')
crop=crop.replace('"crop_radius_zyx": [4,16,16], "epochs": 25','"crop_radius_zyx": [4,16,16], "epochs": 25, "train_jitter_zyx": [2,8,8], "jitter_seed": 20260911')
parts.append(crop)
parts.extend("".join(n["cells"][i]["source"]) for i in [13,14])
folder=root / "research_members/naveen/experiments/E0075_jitter_crop_ranker"
folder.mkdir(exist_ok=True);slug="biohub-e0075-jitter-crop-ranker"
cells=[]
for i,s in enumerate(parts):
    compile(s,f"E0075:{i}","exec")
    cells.append(dict(cell_type="code",id=f"e0075-{i:02d}",metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
(folder / f"{slug}.ipynb").write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=n["metadata"],cells=cells),indent=1)+"\n")
meta=json.loads((parent / "kernel-metadata.json").read_text())
meta.update(id=f"naveenlx111249971939/{slug}",title="Biohub E0075 Jitter Crop Ranker",code_file=f"{slug}.ipynb")
meta["kernel_sources"]=["naveenlx111249971939/biohub-e0065-kaggle-division-ranker"]
(folder / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(folder / "provenance.json").write_text(json.dumps(dict(parent="E0065",change="Train-only common temporal-channel shift +/-2 z and +/-8 y/x voxels; edge padding",
    frozen="Training examples and folds; CNN architecture; 25 epochs; optimizer; pair choices; 7-offset inference; repair cutoffs",
    hypothesis="Improve robustness to off-centre detector proposals without using held-out event labels for augmentation",
    validation="Official graph scores and per-embryo comparison; no leaderboard claim"),indent=2)+"\n")
print(folder)
