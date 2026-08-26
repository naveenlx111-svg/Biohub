# Biohub Cell Tracking Research

Team research repository for the Kaggle **Biohub - Cell Tracking During Development** competition.

## Objective

Build a reproducible 3D+t cell detection and lineage-tracking pipeline that exceeds a post-reset public leaderboard score of **0.95**, while remaining robust under the hidden-test runtime and offline-inference constraints.

## Working principles

- Validate by embryo-disjoint folds; crops from the same embryo must never cross folds.
- Report edge and division components separately, plus node-count calibration and runtime.
- Preserve consecutive-time edges (`t -> t+1`) in the submitted graph.
- Reproduce every leaderboard submission from a versioned configuration.
- Do not rely on invalid coordinates, parser quirks, or metric exploits.

## Repository map

- `docs/competition_notes.md`: metric, dataset, and forum findings.
- `docs/baseline_0926_audit.md`: reproducibility and architecture audit of the post-reset baseline.
- `docs/public_frontier_audit.md`: comparison against legitimate post-reset public frontier notebooks.
- `docs/roadmap.md`: prioritized route from 0.926 to 0.95+.
- `experiments/README.md`: experiment protocol and naming.
- `experiments/ledger.csv`: shared result ledger.

The post-reset 0.926 notebooks are present. Their three Kaggle model/support artifacts and an embryo-disjoint local validation subset are still required before model-level experiments can begin.
