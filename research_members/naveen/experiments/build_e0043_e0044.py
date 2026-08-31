from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "research_members" / "naveen" / "experiments"
SOURCE = EXPERIMENTS / "E0041_candidate_crop_transfer" / "e0041-candidate-crop-transfer.ipynb"
CELL = (EXPERIMENTS / "e0043_candidate_hard_negative_cell.py").read_text()

VARIANTS = {
    "E0043_candidate_hard_negative_finetune": {
        "slug": "e0043-candidate-hard-negative-finetune",
        "title": "Biohub E0043 Candidate Hard Negative Finetune",
        "env": {
            "BIOHUB_HN_LOSS": "bce", "BIOHUB_HN_TOP_K": "1024",
            "BIOHUB_HN_RANDOM_K": "512", "BIOHUB_HN_EPOCHS": "12",
            "BIOHUB_HN_LR": "0.0002",
        },
    },
    "E0044_online_hard_negative_focal": {
        "slug": "e0044-online-hard-negative-focal",
        "title": "Biohub E0044 Online Hard Negative Focal",
        "env": {
            "BIOHUB_HN_LOSS": "focal", "BIOHUB_HN_TOP_K": "2048",
            "BIOHUB_HN_RANDOM_K": "256", "BIOHUB_HN_EPOCHS": "16",
            "BIOHUB_HN_LR": "0.00015",
        },
    },
}


notebook = json.loads(SOURCE.read_text())
for directory, config in VARIANTS.items():
    target_dir = EXPERIMENTS / directory
    target_dir.mkdir(parents=True, exist_ok=True)
    variant = json.loads(json.dumps(notebook))
    env_source = [f'os.environ["{key}"] = "{value}"\n' for key, value in config["env"].items()]
    variant["cells"].append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": env_source + ["\n"] + CELL.splitlines(keepends=True),
    })
    notebook_path = target_dir / f'{config["slug"]}.ipynb'
    notebook_path.write_text(json.dumps(variant, indent=1) + "\n")
    metadata = json.loads((EXPERIMENTS / "E0041_candidate_crop_transfer" / "kernel-metadata.json").read_text())
    metadata.update({
        "id": f'naveenlx111249971939/biohub-{config["slug"]}',
        "title": config["title"],
        "code_file": notebook_path.name,
    })
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
