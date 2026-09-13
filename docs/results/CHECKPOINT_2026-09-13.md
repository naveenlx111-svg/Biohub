# September 13: overnight results and Adaptive Temporal Accord review

Latest public result: E0077 submission56190840 is **COMPLETE, 0.946**, with empty
error_description, independently verified after the user's update. Reject the
ordinary-link HOCT veto: its small diagnostic gain did not transfer publicly.
Production best remains0.947 (56159060). E0081 submission56199686 remains
PENDING with empty error_description. E0081 preserves strong learned links and
does not include E0077's HOCT veto; its public result is still unresolved.
The pending-status notes below describe the earlier check.

Latest compute instruction remains **Kaggle only**. Both E0081 and E0082 were
verified COMPLETE, and results/logs downloaded. No local experiment was run.

E0081 live strong055 inference passed its Kaggle graph audit. Output contains
122838 nodes and118504 edges over all four actual test datasets. Downloaded file
SHA256 matches audit:
`25fdb42d14f7023386ca96ba284b8d27be66144d2ef80ab20450bec0293dc1a7`.
The completion log includes `E0081_VALIDATED` at1652.6seconds. Submitted v1 under
the user's existing authorization to submit promising candidates; public score
must be verified before changing the0.947 production best.
Submission **56199686** is confirmed PENDING with empty `error_description`.

E0082 finished at3157seconds. The eight additional movies exclude the original
panel and were chosen by fixed hash order without reading GT contents.

| Panel / embryo | Original graph processing | Strong055 |
| --- | ---: | ---: |
| Original eight (E0079) | 0.943402106 | 0.971036663 |
| Additional eight (E0082) | 0.885766175 | 0.925727091 |
| Additional44b6 | 0.917601753 | 0.925209617 |
| Additional6bba | 0.874916426 | 0.925857231 |

E0082 adjusted edge Jaccard improves0.877432842→0.916636182, with division
counts1TP/3FP/8FN→1TP/2FP/8FN. This confirms a directional improvement on the
additional panel, mostly through ordinary links. It does not imply that either
panel's absolute score predicts the hidden public leaderboard. Pretrained-model
training overlap remains unresolved.

E0077 submission56190840 remains PENDING with blank public score and empty
`error_description` in the direct API check. No evidence of an execution failure
or a known cause for the delay; do not duplicate or cancel it based on the delay.

## User-linked notebook

Inspected `shoaibssm/biohub-adaptive-temporal-accord` through authenticated Kaggle
source download. All six configuration/setup/inference/graph-generation cells
are **AST-identical to our E0081**. Differences are comments, whitespace, CLI
argument setup for the final audit, and a different completion marker.
The datasets and model settings also match. Thus the expected public result is
essentially E0081's under the same runtime and inputs; the title provides no
additional predictive advantage.

The downloaded notebook includes a saved `KeyboardInterrupt` in setup cell3,
with later prediction cells unexecuted. This saved execution is incomplete.
The session-status endpoint returned404, so its current editor/session state
cannot be established from that API; do not infer current failure from the404.
No reliable numerical leaderboard forecast is available. Our equivalent E0081
has a completed and audited artifact, making it the appropriate submission to
evaluate. Evidence: source comparison with `ast.dump(...,include_attributes=False)`;
download retained under `local_runs/frontier_20260913/adaptive_accord/`.

Next: check E0081's public outcome and E0077 pending status. Keep verified0.947
baseline and final0.97 target. Any new experiments must run on Kaggle.
