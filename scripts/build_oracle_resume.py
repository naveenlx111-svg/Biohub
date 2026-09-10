#!/usr/bin/env python3
"""Append oracle-to-detected-domain diagnostics to a safe scoring notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scorer", type=Path)
    parser.add_argument("oracle_template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-cell", type=int, default=10)
    parser.add_argument("--last-cell", type=int, default=18)
    args = parser.parse_args()

    notebook = json.loads(args.scorer.read_text())
    oracle = json.loads(args.oracle_template.read_text())
    selected = oracle["cells"][args.first_cell : args.last_cell + 1]
    if len(selected) != args.last_cell - args.first_cell + 1:
        raise RuntimeError("Oracle template does not contain the requested cell range")

    for original in selected:
        cell = {
            "cell_type": original["cell_type"],
            "metadata": {},
            "source": original.get("source", []),
        }
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        notebook["cells"].append(cell)

    notebook.get("metadata", {}).pop("papermill", None)
    notebook.get("metadata", {}).pop("codex", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
