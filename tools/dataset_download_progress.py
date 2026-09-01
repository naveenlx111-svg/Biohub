#!/usr/bin/env python3
"""Show progress for the resumable Biohub competition archive download."""

from pathlib import Path


ARCHIVE = Path(
    "/home/naveen/Biohub/data/biohub-cell-tracking-during-development/"
    "biohub-cell-tracking-during-development.zip"
)
TOTAL_BYTES = 87_393_127_165


size = ARCHIVE.stat().st_size
print(f"Downloaded: {size / 1e9:.2f} GB / {TOTAL_BYTES / 1e9:.2f} GB ({100 * size / TOTAL_BYTES:.1f}%)")
print(f"Remaining:  {(TOTAL_BYTES - size) / 1e9:.2f} GB")
