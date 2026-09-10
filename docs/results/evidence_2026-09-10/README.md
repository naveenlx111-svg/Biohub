# Score evidence snapshot — September 10

Compact copies of completed experiment summaries. E0059 was scored locally on
September 9; the other files here are downloaded Kaggle outputs. All numeric
`score` values in these JSON files are **diagnostic**, not public leaderboard scores.
Source artifacts and notebook versions are recorded in `experiments/ledger.csv`.

Verified public results: E0053 = **0.946** (56127787); E0057 = **0.944** (56128925).
E0067 version 2 submitted as **56146417**, pending when this snapshot was made.
The final target is **0.97 public leaderboard**; it is not yet achieved.

E0067 files preserve its deployment policy, diagnostic replay and submission hash.
Large model checkpoints, competition data, logs and submissions are intentionally
excluded from Git. Canonical private Kaggle notebooks attach the necessary input
datasets and parent notebook outputs. Some historical builder scripts additionally
require ignored downloaded artifacts; use the checked-in generated notebooks for
the complete execution definitions.

See [the checkpoint](../CHECKPOINT_2026-09-10.md) for conclusions and active work.
