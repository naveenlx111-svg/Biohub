# September 12: completed-run review and next experiments

Verified public best remains **0.947**, submission56159060 (anchor notebook v1).
The latest submission56159779 scored **0.944**. Neither 0.95 nor the final 0.97
goal has been achieved. September11 turn made progress: E0076 was built, launched,
and recorded. Its completion was verified directly from Kaggle on September12.

## Completed research reviewed

Scores below are the patched official eight-movie diagnostic, **not leaderboard**.
The reused panel has only twelve annotated divisions; a single recovered event
can move the combined metric considerably without demonstrating generalization.

| Experiment | Official diagnostic | Conclusion |
| --- | ---: | --- |
| E0053 Harmonic | 0.943555 | Public0.946; prior production anchor |
| E0057 half feature TTA | 0.953756 | Public0.944; diagnostic gain failed to transfer |
| E0061 / E0062 quarter/no primary TTA | 0.952453 / 0.953797 | Gains concentrated on6bba; no promotion based on this reused panel |
| E0063 / E0064 native division cost | 0.943555 / 0.943554 | No useful new divisions |
| E0065 / E0068–70 crop-repair selection | Best0.949925 | Selection/temporal guards did not improve the best result |
| E0071 live repair deployment | Parent repair diagnostic0.949925 | Hidden rerun succeeded but public0.944; reject |
| E0073 public trackcells reproduction | 0.940161 | Public source proxy0.947853 overstates official metric |
| E0074 DeepCenter threshold0.20 TTA | 0.943559 | Negligible change; local run completed Sept11 |
| E0075 crop jitter | Best0.945948 | 5 divisionTP/11FP/7FN; worse than previous repair policy |
| E0076 exact0.947 anchor | 0.943402 | New reference for HOCT comparisons |
| E0076 ordinary-edge veto | **0.944913** | +0.001511; divisionTP/FP/FN unchanged at2/1/10 |

E0076 all-edge veto and ordinary-only veto have identical official metrics.
Division-only veto gives no metric change. On44b6, ordinary veto changes score
0.935115320 → 0.935042014 (-0.000073306); on6bba,
0.945031506 → 0.947106899 (+0.002075393). Node count/coordinates stay frozen.
Completed runtime was approximately3000seconds for eight movies.

Evidence was read from `local_runs/E0061/kaggle_v2`, `E0062/kaggle_v2`,
`E0068/kaggle`, `E0069/kaggle`, `E0070/kaggle`, `E0073/kaggle`,
`E0075/kaggle`, and `E0076/kaggle`. E0061/62/68/69/70/73/75/76 statuses
were independently rechecked as COMPLETE. Earlier results are retained in the
September9–11 checkpoints and committed evidence snapshots.

## Experiments launched now

**E0077 v1: live HOCT submission candidate, Kaggle T4, confirmed RUNNING.**
Exact0.947 anchor inference with tight55 frozen; HOCT veto removes only ordinary
edges and protects every existing division. It reads actual mounted test images,
not cached public-test outputs. Saves the live anchor CSV separately and checks
schema, actual dataset coverage, image bounds, graph degrees, unchanged nodes,
and that final edges are a subset of the baseline. No train-GT decisions.
This small-gain public probe accepts the tiny44b6 diagnostic regression explicitly;
it is not a claim that the older strict both-embryos promotion gate passed.
Do not replace the0.947 anchor unless the public score actually improves.

**E0078 v1: HOCT fork fusion audit, Kaggle CPU, confirmed RUNNING.**
Reuse E0076 HOCT edges and exact processed anchor nodes. Compare five policies:
anchor, ordinary veto, full HOCT graph, veto plus orphan-only HOCT forks, and
veto plus HOCT forks allowed to replace an ordinary incoming link. New forks
must have one daughter already agreed by both graphs; existing divisions cannot
be stolen from. Labels are used only by the evaluator. Exact per-movie metric
replay of anchor and ordinary veto is mandatory before interpreting variants.

## Next decisions

**E0079 v1 also launched on Kaggle T4:** isolate strong learned-edge preservation
from the new public350-epoch lineage source. Source comparison found three
changes: checkpoint override, threshold0.55, and a replacement
`filter_output_graph` that protects learned links before motion relinking.
E0079 imports only that filter, retains anchor weights/raw graphs, and compares
unchanged anchor versus0.55/0.80/0.95 preservation thresholds. This tests a
different mechanism from the ineffective earlier ILP division-cost changes:
the old motion relinking replaces the ILP-selected graph downstream.
The newer HOCT-b source instead changes secondary feature-TTA weight0.75→1.0;
it is not a different tracking model. No350-epoch weights are used in E0079.

1. Finish E0077, inspect graph audit and runtime, then submit once if valid.
   A successful notebook run is not a successful hidden rerun or a higher score.
2. Use E0078 to determine whether HOCT contributes true new divisions. If it
   does, validate the frozen policy on additional movies before production;
   do not start another crop cutoff sweep on the same twelve events.
3. Expand evaluation beyond the repeatedly reused eight movies. Call this an
   expanded diagnostic, not independent validation, until pretrained checkpoint
   training/validation split provenance is verified.
4. Refresh public code for genuinely different learned models/checkpoints;
   review source and data eligibility before spending GPU time. September12
   list includes a350-epoch lineage source and updated HOCT variants, under review.

No new local training or inference was launched today. Local activity is limited
to notebook construction, result inspection, and small topology unit tests.
