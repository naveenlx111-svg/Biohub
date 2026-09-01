#!/usr/bin/env python3
"""Create a local-path copy of a Kaggle notebook without changing its logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/kaggle_input"),
        help="Local mirror replacing /kaggle/input",
    )
    parser.add_argument(
        "--working-dir", type=Path, default=None,
        help="Local directory replacing /kaggle/working",
    )
    parser.add_argument(
        "--skip-create-working-dir",
        action="store_true",
        help="Rewrite a remote working path without creating it on this machine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    input_root = args.input_root.resolve()
    working_dir = (
        args.working_dir.resolve()
        if args.working_dir is not None
        else (Path("local_runs") / source.stem).resolve()
    )
    notebook = json.loads(source.read_text())
    input_hits = working_hits = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        input_hits += text.count("/kaggle/input")
        working_hits += text.count("/kaggle/working")
        text = text.replace("/kaggle/input", str(input_root))
        text = text.replace("/kaggle/working", str(working_dir))
        compile(text, f"{source.name}:cell{index}", "exec")
        cell["source"] = text.splitlines(keepends=True)
        cell["execution_count"] = None
        cell["outputs"] = []
    if input_hits == 0 or working_hits == 0:
        raise RuntimeError(
            f"Expected Kaggle paths were not found: input={input_hits}, working={working_hits}"
        )
    notebook.setdefault("metadata", {}).pop("kaggle", None)
    if not args.skip_create_working_dir:
        working_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"source={source}")
    print(f"output={output}")
    print(f"input_root={input_root} replacements={input_hits}")
    print(f"working_dir={working_dir} replacements={working_hits}")


if __name__ == "__main__":
    main()
