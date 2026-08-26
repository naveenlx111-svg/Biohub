# Competition notes

Last reviewed: 2026-08-26

## Metric

The score is:

`adjusted_edge_jaccard + 0.1 * division_jaccard`

Node matching is performed independently at each timepoint using optimal bipartite assignment with a maximum physical distance of 7.0 micrometres. Distances must use the physical voxel scale `(z, y, x) = (1.625, 0.40625, 0.40625)` micrometres.

The edge term dominates the score. A predicted edge is correct only if both predicted endpoints match the endpoints of a ground-truth edge. Over-predicting total nodes incurs an additional penalty. Division quality is micro-averaged across samples.

The patched division metric is local rather than graph-wide. It evaluates a window around each split (grandparent, parent, children, and grandchildren), permits a predicted fork one timepoint early or late, requires a local parent-side anchor, and requires two distinct daughter branches. Old connected-component interpretations of this metric are obsolete.

## Dataset implications

- Volumes are usually `(T, Z, Y, X) = (100, 64, 256, 256)` and stored as Zarr v3.
- Anisotropy is exactly 4:1 in voxel units between Z and X/Y. All geometric costs, augmentations, kernels, and matching gates should be defined in physical units.
- Labels are sparse. Treating every unlabeled bright object as a negative during training or validation is unsafe.
- Train/test are embryo-disjoint. Cross-validation must group by the embryo prefix before the first underscore.
- `estimated_number_of_nodes` is useful for per-sample detection-count calibration.

## Submission invariants

- Every test dataset must occur in `submission.csv`.
- Node IDs must resolve correctly within their dataset.
- Valid tracking edges should connect consecutive frames. Frame-stride edges cannot match ground-truth edges.
- Coordinates must be integer voxel coordinates inside the volume.
- The final notebook must run offline in at most 12 hours.

## Metric reset / discussion finding

A public notebook added synthetic hubs and division chains at invalid negative time and spatial coordinates. The competition host acknowledged the report and the evaluator was subsequently patched/rescored. The current official tests explicitly classify the exploit topology's synthetic forks as false positives. This repository excludes any approach based on invalid nodes, graph-parser behavior, or artificial connected components.

Historical leaderboard scores from before the patch are not comparable to current scores. Record whether every score is pre-reset or post-reset in the experiment ledger.

## Open items

- Acquire the three model/support artifacts required by the 0.926 notebook.
- Resolve the two remaining Kaggle dataset-version inputs from notebook metadata.
- Archive per-dataset local metric components and node/edge/division counts for that run.
- Reproduce the official evaluator locally at its current revision.
