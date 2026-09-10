# September 10 checkpoint

Final target: 0.97 public leaderboard. Compute restriction today: Kaggle only;
local machine reserved for the user's other project. Lightweight CLI and record
management only; do not start local model or scoring jobs.

Authenticated Kaggle results:

- E0053 / 56127787: COMPLETE, public **0.946**, new team anchor.
- E0057 / 56128925: COMPLETE, public **0.944**. Its diagnostic 0.953756 did not
  translate to a leaderboard gain. Keep diagnostic and public scores separate.

E0061/E0062 v1 failed after roughly five minutes, before producing scored results.
Both logs report P100 capability sm_60 unsupported by the installed PyTorch build,
followed by `CUDA error: no kernel image is available for execution on the device`.
Logs are under `local_runs/E0061/kaggle` and `local_runs/E0062/kaggle`.

Retry both unchanged notebook algorithms with explicit CLI accelerator
`--accelerator NvidiaTeslaT4`; inspect version 2 outputs before interpreting either
experiment. E0061 uses 25% averaged primary features; E0062 uses native primary
features. Detection TTA and tight55 graph settings remain fixed.

Next: compare completed official diagnostics and per-embryo results, inspect actual
allocated GPU in logs, and avoid treating this eight-movie panel as a reliable
standalone leaderboard selector. Division recovery remains a separate research
lead; GT-guided oracle scores must never be presented as deployable performance.

## Persistent target work

User explicitly requested continued experimentation until the 0.97 public target.
An active goal now tracks this objective, with Kaggle-only compute.
E0061 v2 completed at diagnostic 0.952452833; E0062 v2 at 0.953797067.
Neither is submitted; E0062 is only 0.000041458 above E0057 with mixed embryo effects.

E0063/E0064 v1 now test native ILP division penalties 0.95 and 0.80 respectively,
instead of E0053's 1.2. Full primary feature averaging and frozen tight55 retained.
Explicit T4 allocation and early CUDA smoke test. Actual Kaggle slugs (the title
determined new-kernel URL): `biohub-e0063-native-division-0-95` and
`biohub-e0064-native-division-0-8`, owner `naveenlx111249971939`.
Compare both false divisions and edge disruption before promoting a result.

E0065 is the Kaggle-only continuation of E0060's stopped crop-ranking study.
It mounts E0053's notebook outputs, asserts the reconstructed baseline counts,
and reuses embryo-held-out geometry and temporal crop models. The historical
geometry gate is unchanged; loss of candidate-positive coverage is reported rather
than used to abort execution. The unused second-stage image-feature classifier is
omitted, while geometry pair selection and crop/offset inference are retained.
Predeclared top25/100/250/500 label-free repairs are scored with the official metric.
This is exploratory diagnostic work and produces no competition submission.
Kernel: `naveenlx111249971939/biohub-e0065-kaggle-division-ranker`.
Launch attempt was rejected by Kaggle's maximum **two simultaneous batch GPU
sessions**. E0063/E0064 were rechecked and remain RUNNING. E0065 is prepared but
not running; retry its push when either existing run reaches a terminal state.

## New public source review while GPU jobs run

Latest-run listing refreshed September 10 using the authenticated competition code
endpoint. Downloaded (source only; no local inference) into
`local_runs/frontier_20260910`:

- `sjlee101/biohub-lf-dctta`: compared against downloaded Lineage Forge. Adds one
  environment switch and eight-view logit averaging inside the DeepCenter repair
  gate, leaving its threshold at 0.25. The purported anti-diagonal transform again
  duplicates a horizontal flip, so this is not eight unique D4 views. This is a
  distinct candidate experiment, not a verified improvement. Any test should
  isolate this change on the public 0.946 Harmonic parent, not simultaneously adopt
  Lineage's secondary feature averaging. Threshold score claims in source comments
  have not been independently verified.
- `backtracking/biohub-focus3d-silver-hedge-v1`: alternate 1.1B-parameter instance
  segmentation detector with physical-only linking. Its notebook reports 98.5%
  sparse node recall and 95.2% annotated-edge recall on one movie, not the full
  official score or a verified current leaderboard result. Runtime/model dependency
  `qiweiyin/focus3d-nuclei-runtime`; potentially useful complementary detector,
  but not evidence to replace the current production graph.
- `sushanthtiruvaipati/biohub-xiaoleilian-divaug-fork-v1`: two centre-heatmap U-Nets
  with distinct preprocessing and physical Hungarian linking. No verified public
  score established in this source review.

E0063/E0064 were polled again and both remain RUNNING; do not restart live jobs.

E0066 is prepared (not launched) for the second free slot after E0065: reprocess
E0053 cached raw graphs with baseline single-view DeepCenter, the public eight-view
implementation, and corrected true D4. No detector or edge inference is repeated;
the baseline must reproduce recorded metric counts before variant scoring. The
repair threshold remains 0.25 and tight55 is frozen. Generated notebook syntax
checked locally without executing any experiment. Builder:
`tools/build_cached_deepcenter_tta.py`.

## Completed native-division tests and next launches

E0063 v1 completed at diagnostic **0.9435550596933744**, exactly matching E0053.
E0064 v1 completed at **0.9435542316866237**, about 0.000000828 below E0053.
Both retain identical aggregate edge TP/FP/FN-derived Jaccard, node recall and
division counts (2TP/1FP/10FN). The tiny E0064 difference is in the adjusted-edge
term. Lowering the native division cost did not recover annotated divisions in
this diagnostic panel. Neither run is promoted or submitted.
Downloaded summaries and sample tables: `local_runs/E0063/kaggle` and
`local_runs/E0064/kaggle`. Inspect solver/raw-graph logs before interpreting whether
candidate probabilities or subsequent postprocessing explain the null outcome.

E0065 and E0066 were successfully pushed as **version 1**, explicitly requesting
T4, after both earlier jobs completed. Both authenticated status checks report
RUNNING. No local training or scoring was performed.

## Oracle deployment continuation

E0065 completed: top100 max-offset ranking applies 75 edits and improves the
official diagnostic from 0.943555060 to **0.949924869**, division counts 2/1/10 to
3/2/9 (TP/FP/FN), adjusted edge score 0.928170444 to 0.928496297. Gain is confined
to embryo 44b6; 6bba unchanged. More aggressive cutoffs degrade results. E0066
completed with only +0.000004565 for corrected repair-gate averaging; no new
annotated divisions, so not promoted.

User authorized submission of plausible improvements over public 0.946 and
continued oracle work. E0067 v1 is a production candidate on Kaggle, not yet a
competition submission. It attaches E0053 baseline outputs and E0065 saved crop
models, reproduces geometry pair selection, and must replay the top100 official
score before export. Test inference uses the same embryo-held-out models. The
score threshold is taken from the diagnostic top100, with a candidate-density cap
100/N; this calibrated deployment rule is exploratory, not independent validation.
No test labels are read. Full node/edge schema and topology validation runs inside
Kaggle after export. Inspect deployment_summary.json, diagnostic_replay.json and
submission_audit.json before submitting.

E0068 v1 separately tests within-embryo selection and topology filtering before
ranking, using existing OOF scores only. Fixed budgets per embryo 25/50/100;
baseline and original top100 scores must reproduce exactly. This diagnoses score
calibration and wasted candidate budget without retraining or local experiments.

E0068 completed: topology-first global100 ties 0.949924869 but makes 91 edits
versus 75. Within-embryo50 scores 0.949279599; within-embryo100 is worse again.
No selection variant improves on original globaltop100, so retain the original.

E0067 v1 failed at exact pair-selection replay before crop inference/export; no
submission created. Suspected cause is float64 geometry CSV parsing perturbing
histogram splits during refit, not yet confirmed. Version 2 changes training CSV
read to `float_precision="round_trip"` and adds per-embryo mismatch reports. The
strict replay assertion is retained. Inspect these reports if v2 still fails;
do not remove the check or claim the cause verified until it reproduces.

E0067 v2 completed successfully. Round-trip float parsing restored exact held-out
pair selection (60172 parents for 44b6 and 35961 for 6bba; zero mismatches).
Official diagnostic replay exactly matched 0.9499248688395877. Test inference selected
70 of 67228 parent proposals and applied 55 topology-valid repairs. Full Kaggle
submission audit passed: 122841 nodes, 118564 edges, SHA256
`f7752a96f6fae3c8aa258226b0148fee306bbfb3196c2c5f9b8435aa1b604a07`.
Version 2 submitted as **56146417** at 2026-09-10 14:42:28 UTC, currently PENDING.
Four submissions remain for the day. The 0.946 baseline remains preserved and the
0.97 public target is not achieved until leaderboard evidence establishes it.

## Next experiments after E0067 submission

E0069 tests guards on the original top100/top250 repairs: persistence of both
daughter chains for three further edges, then nondecreasing separation or at least
2 micrometres separation growth. Budgets and guards fixed before scoring; no
replacement candidates are added after rejection. The hypothesis is reduced false
forks, with lost true divisions explicitly measured by official scoring.

E0070 tests selecting the highest-geometry-score feasible daughter pair per parent
before global max-offset ranking. Refit uses round-trip float parsing and must
exactly reproduce saved OOF pair selections first. Geometry labels from the held-out
embryo are not used to train its model. Saved CNN scores are unchanged. Budgets
25/100/250 fixed before scoring. This distinguishes bad pair choice from weak
parent-event ranking. Both notebooks are diagnostic, not new submissions; both
replay baseline and original top100 official scores and run solely on Kaggle T4.
