"""Build Kaggle-only native-fork experiments from the verified 0.946 parent."""
import copy
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = root / "research_members/naveen/experiments/E0053_harmonic_frontier"
original = json.loads((parent / "biohub-e0053-harmonic-frontier.ipynb").read_text())
for exp, weight in [("E0063", 0.95), ("E0064", 0.80)]:
    nb = copy.deepcopy(original)
    nb["cells"][0]["source"] = ("".join(nb["cells"][0]["source"]) + f'''
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "{weight}"
# Fail early on an incompatible allocation, before expensive inference.
import subprocess, sys
subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.ones(4,device='cuda'); print((x+x).cpu()); torch.cuda.synchronize()"], check=True)
''').splitlines(True)
    nb["cells"][10]["source"] = '''
selected_label = "frozen_tight55"
selected_config = {}
(WORKING_DIR / "ppsweep_selected.json").write_text(json.dumps({
    "selected": selected_label, "overrides": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "held_out_stems": val_stems, "selection": "Frozen; no proxy-based selection"
}, indent=2))
'''.splitlines(True)
    audit = "".join(nb["cells"][-1]["source"])
    audit = audit.replace('[("base", {}), ("source_selected", selected_config)]', '[("frozen_tight55", {})]')
    nb["cells"][-1]["source"] = audit.splitlines(True)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"{exp.lower()}-{i:02d}"
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            compile("".join(cell["source"]), f"{exp}:{i}", "exec")
    slug = f"biohub-{exp.lower()}-native-division-{str(weight).replace('.', '-')}"
    folder = root / "research_members/naveen/experiments" / f"{exp}_native_division_{int(weight*100)}"
    folder.mkdir(exist_ok=True)
    (folder / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1)+"\n")
    meta = json.loads((parent / "kernel-metadata.json").read_text())
    meta.update(id=f"naveenlx111249971939/{slug}", title=f"Biohub {exp} Native Division {weight}",
                code_file=f"{slug}.ipynb", machine_shape="NvidiaTeslaT4")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    (folder / "provenance.json").write_text(json.dumps({
        "parent": "E0053 v1 / public 0.946 / frozen tight55",
        "single_algorithm_change": {"ILP_DIVISION_WEIGHT": weight},
        "hypothesis": "Allow high-confidence native forks suppressed by division cost 1.2 with appearance cost zero.",
        "risk": "False forks and changed components may outweigh recovered divisions.",
        "validation": "Pinned official aggregate and per-embryo metrics; eight diagnostic movies, not independent model validation.",
        "compute": "Kaggle T4 only; no local experiment execution"
    }, indent=2)+"\n")
    print(folder)
