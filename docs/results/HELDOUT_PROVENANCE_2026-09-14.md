# Held-out checkpoint investigation

## Decision

No verified, strictly held-out replacement for the full pipeline was found in
the artifacts inspected. This is a scoped search result, not a claim that no
such checkpoint exists anywhere. Existing best remains 0.947 public. No model
training, inference experiment, production modification, or submission was
performed during this investigation. Downloads and metadata/source inspection
were local; experiment compute remains Kaggle only.

## Existing components

- Secondary seed314159: hash-linked manifest confirms training on all199 movies.
  Every original/additional diagnostic movie is in training.
- Primary400: packaged checkpoint lacks training membership. The author's
  [working note, section3.1](https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  explicitly describes UNET300 and UNET400 as all-train models evaluated on the
  same199 training movies. This is author-reported provenance consistent with
  our concern, not a newly discovered split manifest bound to our exact hash.
- DeepCenter: train71 movies from44b6; validation128 from6bba. Validation labels
  select the best checkpoint and gate calibration. Thus even its gradient-held-out
  embryo is not a pristine final test of the selected/calibrated component.
- Bundled temporal training code selects `edge_predictor_best.pth` by
  `test_acc * test_recall`, not official combined graph score. Stored values such
  as0.98349 do not establish leaderboard performance.

## Public checkpoint search

Queried Kaggle dataset search pages1–2 for `biohub`, plus targeted `biohub oof`,
`biohub twofold`, and `271828`. Inspected relevant file inventories and downloaded
small provenance files. Local evidence directory:
`local_runs/validation_provenance_20260913/` (ignored by git).

### Fold0: meaningful lead, not a strict final test

Source: [mihiryanamandra/biohub-artifact-fold0](https://www.kaggle.com/datasets/mihiryanamandra/biohub-artifact-fold0).

`artifact/BUILD.json` reports two embryo folds, only fold0 trained,25 epochs,
seed0, batch4, elapsed10.504 hours. This is the author's reported runtime, not a
forecast for our configuration. `artifact/splits.json` contains:

| Fold | Training movies | Listed test movies | Intersection |
| --- | --- | --- | ---: |
| 0 | 128 from6bba | 71 from44b6 | 0 |
| 1 | 71 from44b6 | 128 from6bba | 0 |

Only `weights/fold_0/edge_predictor_best.pth` is supplied. The bundled trainer
selects that file by accuracy × recall on the listed test set. There is no
fixed-epoch last checkpoint in the full file inventory. Safe metadata loading
shows a136-key state dictionary, no epoch/split/training metadata. The source
also accepts optional pretrained UNet weights, but BUILD does not record whether
they were supplied. Public notebook search for this owner's Biohub source
returned `Not found`; initialization/run-command provenance was not recovered.

Hashes of downloaded files:

- Weight: `f3e4fefbb8e93c47e57f373f2ea53cada8ab90c24cab81d0576135b9ff66ae00`.
- BUILD: `b638249c7e106df137c8d725ea975ed42eb4a96052df67a3f2913b521ab260ed`.
- Splits: `d7c8f3780abafcd0ae81214246ab502e687fe5bdcf7b7da66e1a2320bebb46a3`.

Every one of its71 listed holdout movies is in DeepCenter's training set.
Therefore adding the existing DeepCenter or all-train secondary destroys the
full-pipeline holdout. A standalone fold0 diagnostic might be exploratory, but
cannot be reported as strict untouched validation given selection and provenance
limitations. It is not a verified substitute for a fresh fixed-epoch model.

### Seed271828 is not the hoped-for OOF checkpoint

Source: [xstargate/biohub-edge-seed271828-pilot-output](https://www.kaggle.com/datasets/xstargate/biohub-edge-seed271828-pilot-output).
Downloaded retraining manifest and split file independently show train199,
monitor40, intersection40. Manifest reports14 completed epochs of30 configured,
best epoch11 zero-based, and intended use as an all-train secondary replacement.
It is not the twofold OOF run described in the original author's note; sharing
a seed does not establish shared provenance.

### Other inspected leads

- `pilkwang/biohub-tracking-support-pack-v1`: original weight hash
  `347915de9c33883cb2ee69832a8e4552c88b1ec692d0fbfe956422467d3d4235`;
  inspected manifest contains no training membership. Earlier350 snapshot also
  supplies no verified held-out provenance.
- `muhanqiu/biohub-ft-weights-v1` and
  `xiaoleilian/biohub-unet3d-weights-v2models`: file inventories supply models but
  no separate split/training manifest; embedded provenance was not established.
- `samuelx1a/biohub-aa65e90-oof-validation-toolkit-v1`: full inventory contains
  validation code/catalog/splits, not trained fold weights. A split file alone
  does not turn all-train predictions into OOF predictions.
- `josephadamski91/biohub-v1327-w3-real-model`: receipt describes256 detector
  updates from V1274 with transformer frozen, but provides no held-out membership
  or full initialization lineage. This is a model-development lead, not verified
  clean validation. No inference was run.
- `yaolinghan/biohub-tracking-bundle`: initial inventory includes a best weight
  and dependencies; provenance not established. Inventory was paginated, so this
  investigation does not assert that the entire bundle lacks metadata.

## Proposed next experiment — not launched

1. Train a minimal temporal detector/linker from scratch on Kaggle, using two
   explicit embryo-disjoint folds. Do not initialize from the all-train primary
   or secondary; fine-tuning cannot undo earlier exposure to holdout labels.
2. Begin with a bounded throughput/checkpoint smoke run, not an immediate
   hundreds-of-epochs commitment. Pin code/dependencies/seeds and save split
   hashes, exact initialization, epoch, optimizer state, and fixed-epoch weights.
   Set the diagnostic training budget before examining outer-holdout outcomes.
3. Disable secondary blending and DeepCenter in this first clean diagnostic.
   This tests transfer of graph policies on a simplified system, not the exact
   0.947 ensemble. Fold-specific auxiliaries are required before evaluating the
   full ensemble's generalization.
4. Freeze baseline versus strong055 policies; score exported-and-reloaded CSV
   graphs with the verified official metric. Report both embryo directions and
   runtime. Do not promote based solely on one fold or an absolute score.
5. Separate subsequent policy calibration from final evaluation. With only two
   embryos, cross-embryo results have limited independent replication; movie
   counts do not manufacture additional independent embryos. Any finer split
   requires checking spatial/temporal sample overlap before calling it clean.

The original author likewise distinguishes all-train ensemble weights from
fold-held-out policy evaluation and warns against selecting a best epoch on the
same outer holdout used for reporting:
[working note1, section12](https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/).

This plan repairs experimental evidence. It does not guarantee0.97 or prove the
exact cause of E0081's0.946 public result. E0085 already showed that rounding
does not reverse the candidate ranking on our16 in-sample diagnostic movies.
