"""Build E0055 as a single-mechanism change to E0053."""
import copy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = root / "research_members/naveen/experiments/E0053_harmonic_frontier"
nb = json.loads((parent / "biohub-e0053-harmonic-frontier.ipynb").read_text())
patch = (root / "tools/true_d4_patch.py").read_text().split('if __name__ == "__main__":')[0]
patch += '''
_d4_source = _ps.read_text()
_ps.write_text(patch_true_d4(_d4_source))
print("TRUE_D4: two forward and three inverse transforms corrected; eight unique views")
'''
cell = "".join(nb["cells"][4]["source"])
anchor = "def list_test_stems() -> list[str]:"
assert cell.count(anchor) == 1
nb["cells"][4]["source"] = cell.replace(anchor, patch + "\n" + anchor, 1).splitlines(True)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"E0055:{i}", "exec")
        c["outputs"] = []
        c["execution_count"] = None
dest = root / "research_members/naveen/experiments/E0055_true_d4"
dest.mkdir(exist_ok=True)
(dest / "biohub-e0055-true-d4.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
meta = json.loads((parent / "kernel-metadata.json").read_text())
meta.update(id="naveenlx111249971939/biohub-e0055-true-d4", title="Biohub E0055 True D4",
            code_file="biohub-e0055-true-d4.ipynb")
(dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
(dest / "provenance.json").write_text(json.dumps({
    "parent": "E0053", "hypothesis": "Recover the missing anti-diagonal reflection instead of duplicating horizontal flip.",
    "change": "Exactly two forward and three inverse spatial transform expressions in materialized predictor.",
    "validation": "CPU transform uniqueness and round-trip; eight-movie official diagnostic panel; public score pending."
}, indent=2) + "\n")
print(dest)
