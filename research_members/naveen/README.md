# Naveen's workspace

## Contents

- `notebooks/`: verified leaderboard anchors and imported comparison notebooks.
- `experiments/`: reproducible Kaggle kernel sources and metadata for E0002 onward.
- `notes/`: optional private-to-branch working notes that are suitable for team sharing.

The authoritative experiment status and scores remain in `../../experiments/ledger.csv`. Reviewed findings remain in `../../docs/results/`.

## Current anchor

The verified post-reset team anchor is EXT0005 at 0.934 (submission `55931557`). EXT0006 is the immediate public 0.936 candidate, and the private `0.935` Kaggle notebook has identical executable code. E0045 improved detected-candidate crop AP by about `4.7x` but still failed the low-false-positive gate. E0046 remains pending after Kaggle, local, and Colab infrastructure interruptions; its corrected Colab bootstrap has passed archive, dependency, and checkpoint preflight. The next structural track is fixed-detection HOCT feasibility, followed by multi-view proposal-domain training if E0046 supplies complementary offset evidence.
