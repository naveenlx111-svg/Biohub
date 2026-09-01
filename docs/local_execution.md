# Local execution

Local runs use the ignored `.venv`, `data/`, and `local_runs/` directories. Do not modify Arch's system Python and do not commit downloaded competition data, model artifacts, generated notebooks, predictions, or outputs.

## Directory layout

Mirror Kaggle inputs below `data/kaggle_input/`:

```text
data/kaggle_input/
├── competitions/biohub-cell-tracking-during-development/
├── datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/
├── datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/
├── datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/
└── datasets/dalloliogm/biohub-official-scorer-patched/
```

The model-loading code also checks short Kaggle mount forms. Add local symlinks inside `data/kaggle_input/` only if a run reports a missing short-form artifact path.

## Environment

The repository has an ignored Python 3.12 virtual environment at `.venv`. After support artifacts are downloaded, install their exact cp312 wheel set into this environment and install a CUDA build of PyTorch compatible with the RTX 5060 and current NVIDIA driver. Verify `torch.cuda.is_available()` before running volumetric inference.

## Localize a notebook

Canonical experiment notebooks remain Kaggle-compatible. Generate a path-rewritten local copy with:

```bash
.venv/bin/python tools/localize_kaggle_notebook.py \
  research_members/naveen/experiments/E0043_candidate_hard_negative_finetune/e0043-candidate-hard-negative-finetune.ipynb \
  local_runs/E0043/e0043-local.ipynb \
  --input-root data/kaggle_input \
  --working-dir local_runs/E0043/working
```

The utility replaces `/kaggle/input` and `/kaggle/working`, clears old outputs, removes Kaggle-only metadata, and compiles every code cell. It does not change algorithm parameters.

## Audit candidate rankings

For dense division-parent experiments, inspect every distinct low-false-positive
cutoff rather than relying only on the notebook's coarse quantiles:

```bash
tools/audit_candidate_ranking.py \
  local_runs/E0043/working/candidate_hard_negative_oof.csv
```

Use `--score-column` for outputs with multiple scoring rules, such as
E0046's `center_score`, `offset_max_score`, `offset_top2_mean_score`, and
`offset_mean_score`. The report separates same-pool oracle cutoffs from frozen
embryo-to-embryo threshold transfer. Only the latter is evidence that an
absolute score threshold may generalize; neither result alone authorizes a
submission until the selected edits pass the patched official graph scorer.
