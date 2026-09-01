# Roadmap to 0.95+

Current verified post-reset team anchor: **0.934**, submission `55931557` from `EXT0005` on 2026-09-01. The strongest reproducible public candidate is EXT0006 at a reported `0.936`; the team's private `0.935` notebook has identical executable code but is not promoted until its submission score is visible.

Current public leaderboard frontier as of 2026-09-01: first and second are `0.962`, third is `0.955`, fourth is `0.954`, and fifth/sixth are tied at `0.951`. A safe top-five target is therefore at least `0.952`, leaving a `0.018` gap from the verified team anchor. The threshold is time-dependent and must be rechecked before promoting a final submission.

The strongest current research direction is division-parent recovery. E0034 measures a `0.972017` GT-aware local oracle ceiling on the former 0.933 configuration, while E0041/E0042 show that annotated-centroid division classifiers do not directly transfer to the dense detector-proposal domain. EXT0006 should first establish the `0.936` production base; structural experiments must then be rebased on its exact graph rather than treating public parameter renames as new methods.

The August 30–31 discussion delta also motivates a separate structural track: dense instance segmentation followed by pseudo-track generation and a 5–8-frame higher-order linker. This is not validated score evidence yet. It should begin as a bounded Focus3D plus HOCT/Trackastra feasibility study after the local dataset is available, while the candidate-domain division experiments continue.

## Phase 0: establish a trustworthy baseline

1. Import the exact 0.926 Kaggle notebook, model artifacts, logs, and submission summary.
2. Pin the official evaluator revision and run a CSV round-trip validation.
3. Build embryo-grouped cross-validation and report per-embryo variance.
4. Profile runtime and peak memory on representative dense samples.

No parameter search should be trusted until these four items are complete.

## Phase 1: error decomposition

For each validation fold, save:

- matched/unmatched nodes by time and density;
- edge TP, FP, and FN, split into ordinary continuation and division neighborhoods;
- physical displacement, intensity, morphology, and confidence distributions;
- predicted/estimated node-count ratio per sample;
- failure slices for births, deaths, crossings, gaps, and crowded regions.

This identifies whether the missing 0.024 is primarily detection recall, localization, association, division recall, or node-count penalty.

## Phase 2: highest-value experiments

### Detection and localization

- Calibrate confidence thresholds per sample using robust intensity/density statistics and `estimated_number_of_nodes`.
- Use physical-space 3D non-maximum suppression and sub-voxel peak refinement before integer rounding.
- Test multi-scale or temporal-context detection, especially in dense and low-SNR frames.
- Ensemble genuinely diverse detectors only when cross-validation confirms complementary node matches.

### Association

- Replace purely pairwise nearest-neighbour linking with a physical-space candidate graph and a globally constrained optimizer.
- Learn or tune costs from displacement, appearance, local density, morphology, motion consistency, and detection confidence.
- Estimate local or lineage-level motion, rather than relying on one global radius.
- Keep graph construction sparse and near-linear enough for the 12-hour hidden-test budget.
- Compare adjacent-frame linking with a bounded 5–8-frame higher-order model using dense pseudo-tracks, but promote it only after embryo-held-out official scoring and runtime profiling.

### Divisions

- Generate one-parent/two-daughter candidates with explicit biological constraints: proximity, daughter separation, intensity/volume conservation, and temporal persistence.
- Optimize continuations and divisions jointly so two strong daughter links are not suppressed by one-to-one assignment.
- Tune division thresholds against micro-averaged division Jaccard, but reject gains that materially harm the dominant edge term.

### Robustness and generalization

- Use embryo-held-out folds and report worst-embryo behavior.
- Train with anisotropy-aware spatial augmentation, intensity variation, blur, noise, and missing-label-aware objectives.
- Consider public external zebrafish or synthetic division data only after confirming competition eligibility and measuring domain-gap effects.

## Decision rule

Promote an experiment only when it improves grouped cross-validation, does not rely on one crop/embryo, fits the runtime budget, and has a complete reproducibility record. Use leaderboard submissions as sparse confirmation, not as the main optimizer.
