"""Inventory downloaded public notebooks without executing their source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
downloads = ROOT / "local_runs/frontier_20260909"
inventory = {}
for path in sorted(downloads.glob("*/*.ipynb")):
    notebook = json.loads(path.read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    normalized = []
    for i, source in enumerate(sources):
        tree = ast.parse(source)
        compile(tree, f"{path.name}:cell{i}", "exec")
        # Ignore documentation strings as well as comments when comparing code.
        tree.body = [s for s in tree.body if not (
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str)
        )]
        normalized.append(ast.dump(tree, include_attributes=False))
        (path.parent / f"cell_{i:02d}.py").write_text(source)
    inventory[path.parent.name] = {
        "source": str(path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256("\n".join(normalized).encode()).hexdigest(),
        "cells": len(sources),
    }
for name in ("tta", "lineage"):
    a = downloads / "harmonic"
    b = downloads / name
    diff = []
    for source in sorted(a.glob("cell_*.py")):
        other = b / source.name
        if other.exists():
            diff.extend(difflib.unified_diff(source.read_text().splitlines(True),
                other.read_text().splitlines(True), fromfile=str(source), tofile=str(other)))
    (downloads / f"harmonic_vs_{name}.diff").write_text("".join(diff))
(downloads / "source_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
print(json.dumps(inventory, indent=2))
