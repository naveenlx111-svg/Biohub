# Public frontier audit

Last updated: 2026-08-26

## Sources reviewed

- `evgendvorkin/biohub-0-927-lb` — reported post-reset public score 0.927.
- `flexonafft/biohub-harmonic-fusion` — reported post-reset public score 0.926.
- Team baseline — post-reset public score 0.926.

Known notebooks explicitly labeled as metric hacks were excluded.

## 0.927 candidate changes

The 0.927 notebook retains the same primary/secondary model family, detector threshold (`0.96875`), dual-seed retention threshold (`0.90`), and most tracking/post-processing settings as the team baseline. Its most credible new signal is bidirectional association:

- run the learned edge predictor in forward and reverse temporal directions;
- align reverse logits to the forward distribution;
- combine forward and reverse probabilities with a weighted harmonic construction;
- use bidirectional weight `0.30`;
- penalize links lacking mutual temporal support.

Other changes are confounded with this comparison:

- DeepCenter switches from the epoch-500 `checkpoint_last.pt` to epoch-2 `best.pt`;
- DeepCenter becomes a veto for both gap and safe-division proposals;
- safe-division geometry is widened to parent `8.0` micrometres, sister `11.0` micrometres, existing child `10.0` micrometres;
- safe divisions require divergence and mutual nearest-neighbour support;
- gap-close radius is `5.8` micrometres.

Because several mechanisms changed together, the public `+0.001` cannot be attributed solely to harmonic association without local ablations.

## Validator warning

The notebook describes its validator as official, but its division implementation is the obsolete connected-component formulation. It traverses complete daughter descendants and asks whether both lineages and a fork share a weakly connected component.

The patched official division metric instead uses a local grandparent/parent/children/grandchildren window, allows a fork one frame early or late, requires a local parent-side anchor, and evaluates two distinct predicted daughter branches. Therefore:

- the notebook's edge proxy is directionally useful;
- its division Jaccard is not comparable to the current leaderboard metric;
- its combined local score must not be used as the promotion criterion;
- all division experiments must be rescored with the current official package.

## Controlled experiment sequence

1. `E0001`: add harmonic bidirectional association only to the frozen 0.926 baseline; test weights `0.15`, `0.20`, and `0.30` on embryo-held-out folds.
2. `E0002`: freeze the best association arm and compare DeepCenter epoch 500 against epoch 2 for gap veto only.
3. `E0003`: enable safe-division DeepCenter veto without widening geometry.
4. `E0004`: test widened safe-division geometry with divergence and mutual-NN requirements.
5. `E0005`: calibrate the fixed 90% frame-retention guard using label-free density and uncertainty features.

Only one mechanism changes per experiment. Every arm must report current official edge/division counts, adjusted score, node-count ratio, runtime, and per-embryo results.
