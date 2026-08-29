# EXT0003: claimed leaderboard 0.931

The downloaded `pls-upvote-share-higher-scoring-ideas.ipynb` is reported externally to score 0.931. The score is not present in the notebook's cells or outputs, so it remains an unverified claim until reproduced on the team account.

Unlike SDW85, this is a multi-change lineage configuration. Relative to SDW85 it includes:

- detector threshold 0.965 instead of 0.96875;
- secondary detection fusion 0.475 instead of 0.85;
- disappearance weight 2.0 and division weight 1.2;
- gap-2 recovery and adaptive short-track rescue enabled;
- DeepCenter epoch 500 with safe-division veto disabled;
- harmonic reverse weight 0.15 instead of 0.30;
- broader safe-division generation with mutual-nearest and divergence checks.

Its embedded four-volume validator reports adjusted edge Jaccard 0.9158, division counts 1 TP / 1 FP / 4 FN, and proxy combined score 0.9325. These are approximately tied with the SDW75 official baseline and do not independently validate the leaderboard claim.

The original download SHA-256 is `4eb98eb4ded6e606ed0c86b3f4281344615fb7b6145fbd5283b019a57b0b5150`. The repository copy removes unrelated notebook-builder metadata and has SHA-256 `2f9f7624768a82b64d688b90ba6a645df10a271dc12fa14adc9f435c5b2e21e0`; executable cells and outputs are unchanged.

## Team reproduction

Submission `55861008` completed on 2026-08-29 with public score 0.931 after the evaluator reset. The claim is now verified and this configuration replaces SDW85 as the team leaderboard anchor.
