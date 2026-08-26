# Post-reset 0.926 baseline audit

Audited: 2026-08-26

## Inputs

- `0-926-biohub-divsub.ipynb`
- `biohub-harness-0926-probe.ipynb`

Both notebooks scored 0.926 on the post-reset public leaderboard.

## Reproducibility finding

The concatenated source of every notebook cell has the same SHA-256 in both files:

`8861d14f76fbb722da042f7dc6a6ffcde57c22d798ab8d1a20cecd9f937ad844`

They are therefore repeated executions of one pipeline, not independent model variants. Both report the same final submission SHA-256:

`5f4ec83d56b9fd0620473eed1e92a5a59932bf1c64e231111073ec008de2f7bc`

The prediction runtime was 8.47 and 8.97 minutes on the four visible test videos. The output difference also includes an equivalent alternate Kaggle mount path for the DeepCenter artifact.

## Pipeline

1. Primary temporal 3D U-Net plus node-transformer artifact.
2. Secondary independently seeded temporal model.
3. D4-style spatial test-time augmentation.
4. Low-margin edge-logit consensus and detection blending.
5. Per-frame retention guard: fall back to primary detections when blended detections retain less than 90% of primary candidates.
6. ILP tracking.
7. Conservative motion relinking, single-parent repair, line-fit smoothing, short-track and isolated-node filtering.
8. Center-confirmed synthetic one-frame gap repair using a third DeepCenter model.
9. Safe-division repair with strict graph-degree invariants.

Important configuration includes detector threshold `0.96875`, secondary detection weight `0.475`, secondary edge weight `0.15`, low-margin maximum `0.35`, gap-close radius `5.8` micrometres, and minimum retained track length `6`.

## Output summary

| Dataset | Nodes | Edges | Division parents |
|---|---:|---:|---:|
| `44b6_0113de3b` | 25,361 | 24,656 | 89 |
| `44b6_0b24845f` | 19,622 | 18,366 | 77 |
| `6bba_05b6850b` | 6,159 | 5,957 | 14 |
| `6bba_05db0fb1` | 69,606 | 67,557 | 204 |
| **Total** | **120,748** | **116,536** | **384** |

All reported graphs have maximum indegree 1 and maximum outdegree 2.

## Key diagnostic

The secondary blend triggered primary-detection fallback on 60 of 100 frames, all in `44b6_0b24845f`. Its median candidate retention was about 0.888 and fell to about 0.578. The other three samples triggered no fallback.

This suggests the fixed 90% retention rule is compensating for a sample/domain-specific calibration failure. Replacing the binary fallback with density- and uncertainty-aware blending is a high-value experiment, but it must be validated on embryo-held-out training data.

## Required Kaggle artifacts

- `biohub-tracking-support-pack-50ep-v1` (manifest identifies a 400-epoch snapshot, inference repository, primary weights, and offline wheels)
- `biohub-temporal-unet3d-seed314159-v1` (secondary weights)
- `biohub-deepcenter-unet3d-center-prior-v1` (500-epoch center checkpoint)

The notebook also declares two additional dataset-version inputs whose precise roles should be resolved from their Kaggle metadata.

## Current limitation

The notebook runs inference on the visible test copies and contains no trustworthy embryo-disjoint validation result. A public score alone cannot identify whether the remaining gap is caused by detection, association, division scoring, or node-count adjustment.
