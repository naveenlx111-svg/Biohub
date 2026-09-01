from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "research_members" / "naveen" / "experiments"


def _write_variant(source_dir, source_name, target_dir, slug, title, notebook_transform=None, append_cell=None):
    source = EXPERIMENTS / source_dir / source_name
    notebook = json.loads(source.read_text())
    if notebook_transform is not None:
        notebook_transform(notebook)
    if append_cell is not None:
        notebook["cells"].append({
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": append_cell.splitlines(keepends=True),
        })
    destination = EXPERIMENTS / target_dir
    destination.mkdir(parents=True, exist_ok=True)
    notebook_path = destination / f"{slug}.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = json.loads((EXPERIMENTS / source_dir / "kernel-metadata.json").read_text())
    metadata.update({
        "id": f"naveenlx111249971939/biohub-{slug}",
        "title": title,
        "code_file": notebook_path.name,
    })
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _wide_nonwrapping_jitter(notebook):
    old_config = '_crop_translation_aug = os.environ.get("BIOHUB_CROP_TRANSLATION_AUG", "0") != "0"\n'
    new_config = old_config + (
        '_crop_translation_max_z = int(os.environ.get("BIOHUB_CROP_TRANSLATION_MAX_Z", "2"))\n'
        '_crop_translation_max_yx = int(os.environ.get("BIOHUB_CROP_TRANSLATION_MAX_YX", "9"))\n'
    )
    old_shift = '''            if _crop_translation_aug:
                for _bi in range(len(_x)):
                    _shift = (
                        int(_crop_rng.integers(-1, 2)),
                        int(_crop_rng.integers(-4, 5)),
                        int(_crop_rng.integers(-4, 5)),
                    )
                    _x[_bi] = np.roll(_x[_bi], _shift, axis=(-3, -2, -1))
'''
    new_shift = '''            if _crop_translation_aug:
                for _bi in range(len(_x)):
                    _shift = (
                        int(_crop_rng.integers(-_crop_translation_max_z, _crop_translation_max_z+1)),
                        int(_crop_rng.integers(-_crop_translation_max_yx, _crop_translation_max_yx+1)),
                        int(_crop_rng.integers(-_crop_translation_max_yx, _crop_translation_max_yx+1)),
                    )
                    _source = _x[_bi]
                    _padded = np.pad(
                        _source,
                        ((0, 0), (_crop_translation_max_z, _crop_translation_max_z),
                         (_crop_translation_max_yx, _crop_translation_max_yx),
                         (_crop_translation_max_yx, _crop_translation_max_yx)),
                        mode="edge",
                    )
                    _z0 = _crop_translation_max_z + _shift[0]
                    _y0 = _crop_translation_max_yx + _shift[1]
                    _x0 = _crop_translation_max_yx + _shift[2]
                    _x[_bi] = _padded[
                        :, _z0:_z0+_source.shape[-3],
                        _y0:_y0+_source.shape[-2], _x0:_x0+_source.shape[-1]
                    ]
'''
    config_hits = shift_hits = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if old_config in source:
            source = source.replace(old_config, new_config, 1)
            config_hits += 1
        if old_shift in source:
            source = source.replace(old_shift, new_shift, 1)
            shift_hits += 1
        cell["source"] = source.splitlines(keepends=True)
    if (config_hits, shift_hits) != (1, 1):
        raise RuntimeError(f"Wide-jitter patch mismatch: config={config_hits}, shift={shift_hits}")


_write_variant(
    "E0042_jittered_crop_transfer", "e0042-jittered-crop-transfer.ipynb",
    "E0045_wide_nonwrapping_jitter", "e0045-wide-nonwrapping-jitter",
    "Biohub E0045 Wide Nonwrapping Jitter", notebook_transform=_wide_nonwrapping_jitter,
)
_write_variant(
    "E0041_candidate_crop_transfer", "e0041-candidate-crop-transfer.ipynb",
    "E0046_offset_pool_transfer", "e0046-offset-pool-transfer",
    "Biohub E0046 Offset Pool Transfer",
    append_cell=(EXPERIMENTS / "e0046_offset_pool_transfer_cell.py").read_text(),
)
