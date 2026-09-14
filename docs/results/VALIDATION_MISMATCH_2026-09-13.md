# Validation-to-public mismatch investigation

## Outcome

Confirmed a validation-design flaw: **all 16 diagnostic movies occur in the
secondary checkpoint's training manifest**. E0082's additional eight were new to
our diagnostic panel, not unseen by that pretrained model. Neither panel is an
independent generalization test of the full pipeline. This corrects earlier
notes calling checkpoint overlap unresolved.

E0081 scored **0.946 public**, below the verified **0.947** anchor. Its original
eight-movie diagnostic of 0.971037 is not evidence that the public 0.97 target
has been reached. Training overlap is established; its contribution to the
exact public regression is not quantified.

## Checkpoint provenance

Evidence resides under
`data/kaggle_input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/`.

- `split_manifest.json`: 199 training movies; original panel overlap 8/8;
  additional panel overlap 8/8. Its 40 listed test movies are also in training.
- `training_config.json`: method `unet_transformer_alltrain_seed314159_v1`,
  199 training datasets and 40 validation datasets.
- `SNAPSHOT_MANIFEST.json`: epoch 400; best epoch 381. Recorded edge checkpoint
  SHA256 `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`
  matches the **Secondary SHA256** printed by the actual E0081 execution log.
- The snapshot's split-manifest hash matches the inspected local file:
  `cbe8ace34ffc157172280538441454b60250f0188faa063d1a9eadfb1ac55c0b`.

This links the overlap to the model actually used, rather than merely to a
similarly named dataset. Strong-edge preservation favors learned links, so
in-sample evaluation could favor it disproportionately. That is a plausible
mechanism, not a proven attribution of the public score difference.

The DeepCenter manifest lists 71 training movies (44b6) and 128 validation
movies (6bba); four movies in each diagnostic panel overlap its listed training
set. The primary checkpoint metadata exposes epoch 402 and best score
0.9834918738001537, but no training membership list. That stored training score
does not establish held-out performance or leaderboard performance. Primary
training membership remains unresolved at the hash-linked manifest level.
September14 follow-up found the original author's explicit description of
UNET400 as all-train, and audited alternative public checkpoints. See
[held-out provenance investigation](HELDOUT_PROVENANCE_2026-09-14.md) for the
evidence distinction and proposed fresh-training plan.

## Official metric check

The pinned scorer matches the available official repository at commit
`075fc5f5a52d11077f9dc2b074644618f26939e2`:

- `metrics.py`: SHA256
  `cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444`.
- Division metric: SHA256
  `0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9`.

Source: [official competition evaluator](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2).
The diagnostic uses the same 7 micrometre matching threshold and summary
implementation. This rules out a stale scorer relative to the public official
source, not undisclosed differences in Kaggle's private server environment.

## E0085: production CSV parity audit — COMPLETE

Existing diagnostics score floating-point coordinates. Production CSV writing
rounds each spatial coordinate to an integer and clamps it below at zero.
The completed audit shows small score changes but **no panel-ranking reversal**.

[Kaggle E0085](https://www.kaggle.com/code/naveenlx111249971939/biohub-e0085-export-parity-audit)
version 1 completed on Kaggle CPU only. No local model experiment, production change,
or submission was made for this investigation.

Controls:

1. Reuse frozen E0079/E0082 processed nodes and edges for both policies on all
   16 movies; no inference or topology modification.
2. Replay floating-point scoring and require exact agreement with saved edge,
   division, and node counts before interpreting results.
3. Export the production ten-column CSV with identical coordinate conversion;
   read it with Polars and the official `build_graph_from_rows` implementation.
4. Score the reconstructed graph and report original/additional/pooled metrics,
   plus coordinate displacement statistics. Expect 64 evaluations and
   `E0085_COMPLETE` before treating the summary as final.

Downloaded outputs: `export_parity_samples.csv`, `export_coordinate_changes.csv`,
and `export_parity_summary.json`, retained under `local_runs/E0085/kaggle`.
Verified 64 unique sample/config/representation evaluations, 32 coordinate
reports, and 12 panel summaries. All exact float replay assertions passed;
the log prints `E0085_COMPLETE` at 805.114 seconds (13.42 minutes).

| Panel | Policy | Float diagnostic | Production CSV diagnostic | Change |
| --- | --- | ---: | ---: | ---: |
| Original eight | Anchor | 0.943402106 | 0.944067207 | +0.000665101 |
| Original eight | Strong055 | 0.971036663 | 0.970906934 | -0.000129729 |
| Additional eight | Anchor | 0.885766175 | 0.883898681 | -0.001867494 |
| Additional eight | Strong055 | 0.925727091 | 0.922500465 | -0.003226626 |
| Pooled sixteen | Anchor | 0.920779306 | 0.920472316 | -0.000306990 |
| Pooled sixteen | Strong055 | 0.952195985 | 0.950891971 | -0.001304014 |

Division TP/FP/FN totals are unchanged for each policy on both panels and pooled:
anchor 3/4/18 versus strong055 5/3/16 across sixteen movies. Maximum observed
coordinate displacement is 1.099796777 micrometres. Rounding affects node matches
and ordinary-edge scoring, but strong055 still ranks above the anchor on each
panel. Thus integer export does **not reproduce the public ranking reversal on
these diagnostic movies**. This does not measure the rounding effect on hidden
public ground truth, and does not establish training overlap as the sole cause.
The pooled CSV score 0.950892 remains in-sample, not a new public 0.95 result.

Builder:
`tools/build_export_parity_audit.py`; committed notebook contains the copied
official graph-conversion function. Rebuilding requires the official source
checkout under ignored `local_runs/metric_review_20260913` at the pinned commit.

## Decisions

- Keep the 0.947 public anchor; do not promote E0081, E0083, or E0084.
- Treat previous panels as in-sample diagnostics, useful for mechanics but not
  sufficient promotion evidence. More movies from the same all-train manifest
  will not repair independence.
- Export parity is measured. Do not change production serialization based on
  these results; use exported-and-reloaded graphs for future scoring parity.
- A clean validation design needs verified held-out membership for **every**
  trained component (primary, secondary, DeepCenter, and any repair model).
  If suitable weights do not exist, this requires retraining with fixed held-out
  movies or embryo groups. Embryo-level splitting is limited by only two embryos.
- Keep the GT-aware oracle separate: it diagnoses potential recoverable errors;
  it is not an achievable-score guarantee for an unseen-data algorithm.

This investigation identifies a concrete reliability problem, not a completed
fix or a promise that correcting it will yield 0.97.
