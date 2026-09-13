# Roadmap to 0.97

Validation investigation: **all 16 diagnostic movies were in the secondary
model's training manifest**, tied to E0081 by checkpoint and manifest hashes.
They are not independent held-out evaluation. The scorer matches current
official source; E0085 is RUNNING on Kaggle CPU to measure integer CSV export
effects. See [evidence and corrective direction](results/VALIDATION_MISMATCH_2026-09-13.md).
Do not promote another candidate solely on these in-sample diagnostic gains.

Latest verified outcome: **E0081 scored0.946 public**, as did E0077. Neither
improves the0.947 production anchor. Gains on both diagnostic panels failed to
transfer publicly; review validation reliability before using that ranking to
promote another candidate. E0083/E0084 also completed and are weaker diagnostic
variants. The pending-status updates below are historical.

September13: E0081 completed and was submitted after graph audit. E0082's additional
eight movies improve **0.885766→0.925727**, with gains on both embryos. The user's
Adaptive Temporal Accord notebook has the same prediction code as E0081.
Public best remains0.947 while submissions are pending. See
[overnight results](results/CHECKPOINT_2026-09-13.md). Compute remains Kaggle only.

Latest September12: **strong-edge preservation scored0.971037 on the diagnostic
panel**, improving both embryos; public best remains0.947. E0081 live candidate
and E0082 additional-movie validation are RUNNING on Kaggle T4. E0077 HOCT candidate
was submitted after its graph audit passed. All next experiment compute is
Kaggle-only per the user's latest instruction. See the
[latest findings](results/CHECKPOINT_2026-09-12.md).

September12 update: **0.947 public remains best; 0.95 not yet achieved**.
E0076 ordinary-edge HOCT veto improved the official diagnostic by0.001511,
without recovering divisions. E0077 live candidate and E0078 fork-fusion audit
are running; E0079 isolates strong learned-edge preservation before motion
relinking. See [completed-run review and next decisions](results/CHECKPOINT_2026-09-12.md).

September 11 update (supersedes older status and compute restrictions): verified
anchor **0.947 public**, submission56159060 (`biohub-lf-dctta-v020` v1).
E0071 live repair scored0.944; E0073/74/75 did not justify promotion.
E0076 HOCT consensus audit is running on Kaggle T4 against the exact anchor.
Local CPU/GPU permission was explicitly restored today. See
[current checkpoint](results/CHECKPOINT_2026-09-11.md).

September 10 update: **E0053 scored 0.946 public** (56127787), now the verified
anchor. E0057 scored 0.944 (56128925), despite its higher 0.953756 local diagnostic.
Do not promote feature blends solely on this small diagnostic panel. E0061/E0062
version 1 failed because the allocated P100 is unsupported by the installed torch
build; retries explicitly request `NvidiaTeslaT4`. User constraint for September 10:
**Kaggle compute only; no local training, inference, or CPU experiments.**

September 9 end-of-day target update (supersedes historical targets below): the user
set **0.97 public leaderboard** as the final target. E0057 reached 0.953756 on the
eight-movie official diagnostic panel and was submitted as 56128925; public evaluation
remains pending. This is not yet a verified public 0.95 or 0.97 result. E0061/E0062
extend the feature-blend comparison to 25% and 0% averaged primary features, with
detector code and tight55 fixed. Resume from
[the overnight checkpoint](results/FRONTIER_2026-09-09.md#overnight-checkpoint).

Current verified post-reset team anchor: **0.941**, submission `56051745`, notebook `Biohub Cell Tracking v`, version 1, September 6. Verified against the authenticated submissions endpoint on September 9. The previously pending EXT0006 scored 0.936 (submission `55944277`); Kimi v19 subsequently scored 0.938 (`55976149`).

Active September 9 work: E0053/E0054 reproduce the latest primary/secondary edge-feature TTA public notebooks on Kaggle. E0055 tests a confirmed duplicate-transform defect in their nominal eight-view TTA. Local official rescoring gives 0.943555 for Harmonic with tighter motion relinking, below its obsolete 0.951247 division proxy. See [the current audit](results/FRONTIER_2026-09-09.md) for evidence, provenance, and pending executions. The first target remains strictly greater than 0.95.

Historical public leaderboard frontier as of 2026-09-01: first and second were `0.962`, third `0.955`, fourth `0.954`, and fifth/sixth tied at `0.951`. These rankings have not been refreshed in the September 9 audit; the user's current priority is exceeding 0.95, independently of rank.

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
