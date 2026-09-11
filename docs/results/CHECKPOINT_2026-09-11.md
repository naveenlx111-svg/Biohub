# September 11: execution resumed

## Latest verified state (supersedes chronological notes below)

- Production anchor: submission **56159060**, `biohub-lf-dctta-v020` version1,
  **0.947 public**, COMPLETE with no error. Target remains **0.97 public**.
- E0071 submission **56159779** completed successfully but scored **0.944**.
  Live hidden-data inference works; division repairs did not improve public score.
- E0073 completed: official diagnostic **0.940160947**, despite notebook proxy
  **0.947853255**. Not submitted or promoted.
- E0074 completed on local GPU: DeepCenter public/corrected TTA threshold0.20
  scored **0.943558178 / 0.943558621**. Negligible change; not promoted.
- E0072 completed locally; event funnel is described below.
- E0075 completed on Kaggle: best offset-max top250 diagnostic **0.945948327**,
  5 division TP / 11 FP / 7 FN. Worse than previous repair diagnostic
  **0.949924869**; no submission. More recovered divisions came with too many FP.
- **E0076 version1 is RUNNING on Kaggle T4**: independently evaluates HOCT
  temporal edge consensus on the exact cached 0.947 anchor validation graphs.
  Four predeclared configurations: unchanged anchor, ordinary-edge veto,
  all-edge veto, division-edge veto. Nodes and selected tight55 postprocessing
  remain frozen. Uses the hash-verified patched official scorer.

Public HOCT source `sjlee101/biohub-lf-hoctveto-div` applies its veto through a
temporary `write_test_submission` hook; its validation calls bypass that hook.
Thus its displayed validation score is not evidence for HOCT. E0076 explicitly
scores the modified graphs. This is exploratory evaluation on a reused eight-movie
holdout, not independent confirmation or a public leaderboard score.

Builder: `tools/build_hoct_consensus_audit.py`. Notebook and provenance:
`research_members/naveen/experiments/E0076_hoct_consensus_audit/`.
Expected outputs: `hoct_official_summary.json`, `hoct_official_samples.csv`,
`hoct_diagnostics.json`, anchor graph CSVs and cached HOCT edge pairs per movie.

## Chronological execution notes

User explicitly restored local CPU/GPU compute permission today, superseding the
September 10 Kaggle-only constraint (including its older wording in the active goal).
Final target remains 0.97 public leaderboard; verified production anchor 0.946.

E0067 submission 56146417 failed during hidden-data rerun. API status COMPLETE with
blank score is misleading without `error_description`, which reports an unhandled
rerun error. No public score was obtained. Cached public-test predictions are a
confirmed design defect, but the exact hidden traceback is unavailable.

E0071 v1 launched: actual E0053 inference on mounted competition test images,
followed by frozen E0067 repairs. No cached test CSV input or train-GT scoring in
production. Unknown embryo identities retain newly inferred baseline predictions.
Actual test datasets are discovered and checked. Hidden-rerun compatibility still
requires verification; do not claim it fixed merely because a public run succeeds.

Public code list refreshed. `anhadmahajan06/biohub-track-your-cells` claims
0.947–0.950+ in its markdown; this is not independently verified leaderboard evidence.
E0073 is prepared to reproduce its original executable cells and append the pinned
official audit. Its first push was rejected because inherited source metadata
contained the author's numeric kernel ID; that metadata was removed. A second
push was rejected by the two-session GPU limit. It is **not running**. E0071
requests Kaggle T4 and is confirmed running; do not stop other jobs to free space.

E0074 executes locally on the RTX5060: cached Harmonic graph rescoring with public
and corrected DeepCenter TTA at threshold0.20, compared against exact single-view
baseline. This tests the change in `sjlee101/biohub-lf-dctta-v020` independently of
its other Lineage settings. No local environment installation is performed.

E0072 remains a proposed missed-event analysis, not a launched experiment.
Builders: `tools/build_live_repair_submission.py` and
`tools/build_september11_experiments.py`. Local job notebook and outputs are under
`local_runs/E0074`; Kaggle slugs match E0071/E0073 experiment metadata.

E0071 v1 completed: live inference produced the exact public output SHA256
`f7752a96f6fae3c8aa258226b0148fee306bbfb3196c2c5f9b8435aa1b604a07`, 55 repairs,
and valid graph audit; both pair-replay mismatch counts are zero. Submitted as
**56159779** at September11 06:06:48 UTC, PENDING. Hidden rerun is still unverified.
The user's separate `biohub-lf-dctta-v020` submission **56159060** also appears
pending; do not duplicate it. E0073 v1 successfully launched after E0071 completed
and is confirmed RUNNING. E0074 local execution remains live, not yet complete.

## Executed missed-event analysis and jitter experiment

E0072 completed locally on CPU, reproducing both official baseline0.943555060 and
top100 repair0.949924869 exactly before interpreting event identities. Among twelve
GT divisions: two baseline TPs; one recovered by top100; four lost at crop ranking;
one at pair selection; one at proposal generation; three without a matchable
same-frame role triple. The four missed crop-ranked events have best global ranks
617 / 18028 / 57872 / 75645, all on embryo6bba. This supports investigating the crop
domain shift rather than another cutoff sweep. Event identities are diagnostic only.
Outputs: `local_runs/E0072/event_funnel.csv` and `summary.json`.

E0075 prepares the single-change train-only position-jitter experiment: per-example
shifts up to +/-2 Z and +/-8 XY voxels, shared across temporal channels, with edge
padding. The previous code's comment mentioned jitter but implemented flips only.
Examples, embryo folds, architecture, optimizer,25epochs, pair selections, seven
inference offsets and repair cutoffs remain fixed. Jitter uses a separate RNG.
Kaggle kernel `naveenlx111249971939/biohub-e0075-jitter-crop-ranker` attaches E0065
cached graph/parent outputs and scores with the same pinned official evaluator.
