# Kaggle discussion audit: August 19–29, 2026

## Scope and method

The authenticated Kaggle competition forum was enumerated across all 80 available topics, not only the first discussion page. Every topic was inspected for either a creation date or reply date from 2026-08-19 through 2026-08-29 Asia/Kolkata time.

The resulting audit covers 14 active threads, including all eight threads created during the window and six older threads revived by new replies. Their 64 available comments were read in full; 32 were posted during the window. Kaggle's topic API exposes the title, author, date, votes, and all replies, but omits a separate main-post body. Consequently, zero-reply threads are represented by their titles and metadata only. Empty comments returned by Kaggle are noted but carry no evidence.

Forum statements are participant reports, not organizer-confirmed facts unless explicitly identified as such. Leaderboard and validation numbers are therefore treated as claims rather than ground truth.

## High-value technical findings

### 1. The 0.928 wall is a bottleneck-identification problem

The strongest practical reply came from the [Stuck at 0.928](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101) thread. The recommended workflow is to divide every missed GT edge into:

- an endpoint detection is missing; or
- both endpoints exist but their association is wrong or absent.

The same reply recommends complete-movie, movie-level out-of-fold validation with the official scorer. Random edge-level validation can be badly misleading because it leaks movie-specific conditions and does not evaluate the final graph.

This independently supports our current methodology. E0017 already found that all four division false negatives have parent and both daughter-lineage detections, but no parent-side fork. Our dominant division problem is therefore association/topology, not missing detections.

### 2. High global node recall does not imply strong division Jaccard

In [what layer did ur gains actually come from](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543), one participant reported node recall 0.9945, adjusted edge Jaccard 0.9212, but division Jaccard only 0.1176. Another participant argued that divisions need dedicated handling and reported roughly 0.28 leaderboard division Jaccard despite adjusted edge Jaccard around 0.89.

The most actionable claim is that division performance was much more sensitive to localization within roughly 3 µm, even though official node matching allows 7 µm. The original poster then measured that 39% of their division nodes were outside 3 µm despite remaining inside the official 7 µm gate.

This is a participant observation, not a metric rule. It suggests a new diagnostic for us: measure parent, child, and grandchild localization-error distributions at annotated division windows, especially recall within 3 µm, rather than relying only on overall 7 µm node recall.

### 3. Work in the order detection, linking, then division—but stop revisiting solved layers

One view in the same thread proposed the order detection → linking → division because downstream stages cannot repair absent cells. Other replies qualified this: sparse labels make detector training difficult, and division-specific handling remains necessary even after detection recall becomes high.

The useful synthesis is conditional rather than universal:

1. establish whether endpoints exist;
2. if endpoints are missing, improve detection/localization;
3. if endpoints exist, improve association and fork topology;
4. evaluate division windows separately from ordinary continuation edges.

For our current held-out divisions, step 2 is not the main failure under the 7 µm metric, so the next production work should remain on fork creation and daughter assignment while also auditing the proposed 3 µm localization effect.

### 4. The public checkpoint is widely viewed as near its postprocessing ceiling

The [division jaccard](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737577) discussion describes two parallel strategies: extracting small gains by tuning the public model and training a new model from scratch. The author states that the public approach has reached a final checkpoint where further work produces little gain.

An older thread revived during the window, [What is the best model for this domain so far?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604), similarly recommends retraining rather than relying only on the public checkpoint. The UNet plus node-transformer backbone is still regarded as strong, but suggested gain sources include preprocessing, architecture changes, detector calibration, division ranking, and a separate track-correction model.

This matches our empirical scalar sweeps: detection fusion has produced leaderboard movement, but nearby scalar changes are inconsistent and cannot plausibly supply the full target gap.

### 5. Score decomposition should precede model replacement

The most detailed modeling reply recommends measuring four layers independently:

- node recall within 7 µm and predicted count versus the estimated total;
- linking accuracy conditional on both endpoints being detected;
- an oracle-linker ceiling using the current detections;
- division parent/daughter availability, candidate recall-at-K, ranking quality, and final TP/FP/FN.

It also gives the approximation `edge recall ≈ node recall² × conditional linking accuracy`, illustrating why small detector miss rates can have amplified edge consequences. The exact equality is not guaranteed, but the decomposition is sound.

Our validator already covers most of this. Missing pieces worth adding are an oracle division-repair ceiling, candidate recall-at-K, and conditional association accuracy reported separately for continuation and division edges.

### 6. A general graph-correction layer is claimed to provide large gains, but evidence is incomplete

In [Possible big leaderboard shakeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352), one participant claims a plug-in compatible with multiple public notebooks improves scores by 0.03–0.05 and reaches 0.940 from a public model without tuning. They do not disclose whether it refines detections, reassigns tracks, or replaces the tracker, and no reproducible artifact or validation table is provided in the thread.

This should be treated as a directional clue, not verified evidence. The plausible lesson is that a reusable graph-correction stage may have more headroom than additional scalar tuning. The numeric claims should not drive decisions until reproduced.

### 7. Training from scratch is considered feasible

Multiple participants report training their own models, with one stating it is affordable. Another reports training from scratch yielding division Jaccard around 0.3 while their edge term remained roughly 0.01 below public notebooks. This suggests complementary strengths: public pipelines appear edge-heavy, while specialized training can improve divisions but may lose ordinary association quality.

A promising research direction is therefore not a full replacement initially, but a specialist division model or ranker combined with the strongest edge pipeline.

### 8. Dim cells remain a visible false-negative category

The [Very dim nodes?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896) thread explicitly raises dim-node false negatives. Its available replies contain thanks but no technical solution, because Kaggle did not expose a separate body for the post. This is weak evidence, but it supports stratifying missed detections by intensity and local contrast rather than treating every miss alike.

## Metric, evaluator, and validation context

### Patched evaluator

The older official Discord thread documents the fake-hub division exploit and the organizer response that it was being investigated. The exploit connected real tracks to synthetic nodes outside the acquisition volume to satisfy the old division-component logic. The current patched scorer requires valid local branch topology, so pre-reset connected-component tricks are obsolete.

This reinforces our use of the hash-verified scorer at reference commit `075fc5f`. Any notebook or discussion claim derived from the earlier evaluator must be re-scored before comparison.

### Public notebook rankings

The new [Public Notebook Rankings Need a Metric Refresh](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937) thread had no replies. Its title is consistent with the evaluator reset, but it provides no technical evidence beyond that concern.

### Sparse labels

Several participants note that sparse annotations complicate detector training and local validation. This strengthens the case for embryo-level complete-movie folds, official sparse-label accounting, and avoiding direct interpretation of unmatched predictions as ordinary false positives.

## Competition operations

### Submission runtime

Two active threads report submitted notebooks remaining in evaluation for 7–11 hours. An older reply cites the competition description that the hidden test set is approximately the size of the training set. Long evaluation after a fast visible run is therefore normal and does not by itself indicate a failed submission.

Practical consequence: production notebooks need generous runtime margins under the 12-hour limit, and experiments should be ready before spending a daily submission slot.

### Hand-labeled external data

The [Hand labeling - is it external data?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737103) thread contains disagreement:

- one participant says hand labeling is prohibited;
- others point out that the cited rule appears to prohibit labeling competition validation/test data, not unrelated external data;
- another participant says hand-labeling external data has usually been allowed but cannot confirm this competition's interpretation.

No host answer appears in the available replies. We should not rely on participant interpretation for a rules-sensitive decision. Before creating labels, obtain a written host clarification or restrict work to already public, freely available labeled data clearly allowed by the rules.

### Visualization

The revived napari thread recommends interactive 3D/time visualization and tree views for detecting identity swaps, crossing-track failures, and biologically implausible trajectories that aggregate metrics can hide. This is useful for targeted error analysis even though it does not directly improve the score.

## Thread-by-thread inventory

| Thread | Activity in window | Main contribution |
| --- | ---: | --- |
| [Very dim nodes?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896) | New; 2 replies | Flags dim-cell false negatives; no exposed technical solution |
| [Scoring after notebook ran](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737659) | New; 4 replies | Submitted runs commonly take 7–11 hours |
| [division jaccard](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737577) | New; 2 replies | Public tuning for small gains versus training from scratch |
| [what layer did ur gains actually come from](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543) | New; 7 replies | Detection→linking→division; high recall alone insufficient; possible 3 µm division-localization effect |
| [does anyone have a different design for divisions](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438) | New; 0 replies | Signals broad interest in alternative division design; no evidence supplied |
| [Hand labeling - is it external data?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737103) | New; 5 replies | Conflicting participant interpretations; no host ruling |
| [Stuck at 0.928](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101) | New; 4 replies | Decompose missing endpoints versus bad associations; use complete-movie official OOF |
| [Public Notebook Rankings Need a Metric Refresh](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937) | New; 0 replies | Raises reset-era ranking concern without evidence |
| [Share a custom napari visualizer](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724130) | 2 recent replies | Interactive graph/trajectory inspection |
| [How to get started + Official Discord](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/714101) | 1 recent reply | Older replies document the reported metric exploit and organizer awareness |
| [how long does the submission take?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734237) | 1 recent reply | Hidden set approximately training-set sized |
| [What is happening after submitting?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735307) | 1 recent reply | Long evaluation queue/runtime is common |
| [Possible big leaderboard shakeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352) | 2 recent replies | Unverified graph plug-in claims; scratch training considered feasible |
| [What is the best model for this domain so far?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604) | 1 recent reply | Separate modeling from track optimization; public checkpoint near ceiling |

## Recommended changes to our roadmap

1. Add a division-window localization audit at 3 µm and 7 µm for parent, children, and grandchildren.
2. Add oracle ceilings that distinguish missing detections, absent fork edges, wrong daughter assignment, and scorer-topology failure.
3. Report conditional linking accuracy only on GT edges whose endpoints are both detected.
4. Build a learned division candidate ranker or specialist model using local image crops, motion, sister geometry, and branch continuation rather than relying on geometric postprocessing alone.
5. Begin a controlled retraining track while preserving SDW85 as the strongest leaderboard-backed inference baseline.
6. Treat large undisclosed plug-in gains and unverified notebook scores as hypotheses until reproduced on the team account.
7. Keep the patched official scorer and embryo-level complete-movie folds as mandatory promotion gates.
8. Obtain organizer clarification before any hand-labeling effort.

## Bottom line

The forum does not reveal a copyable secret configuration. It does provide strong independent confirmation that the public pipeline's easy tuning gains are close to exhausted. The most defensible path beyond the 0.928–0.931 range is a reliable full-movie validation system plus either a learned division-specific ranker/correction stage or controlled retraining. For our current errors, the immediate next diagnostic should combine the E0017 topology finding with division localization at the tighter 3 µm scale suggested by the discussion.
