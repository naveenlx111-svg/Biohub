# HOCT feasibility for Biohub

Audit date: 2026-08-31

Local bundle status: prepared and checksum-verified under the ignored
`data/hoct_support/` directory (66 MB). The first Kaggle upload attempt ended
before dataset creation because the upload endpoint sustained very low
throughput; retry after the main competition archive finishes downloading.

## Decision

Run a bounded embryo-held-out HOCT feasibility study, but do not depend on
FOCUS-3D weights yet. The first study should convert the existing detector's
centroids into small non-overlapping instance masks, pass the masks and raw
intensity volumes to HOCT, and score the resulting graph with the patched
official evaluator. This isolates the value of higher-order association from
a simultaneous detector replacement.

## Evidence

- HOCT source commit `2ccc5040823bc944ab67790abd1f56eea7cd4f05` is MIT
  licensed and exposes end-to-end 3D+t tracking from images and instance masks.
- Its default `general_v1` JIT checkpoint is about 25.5 MB and is pinned by
  SHA256 `5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33`.
- The default model uses a five-frame window, candidate links up to three
  frames apart, tiled inference for dense graphs, and an ILP that tries Gurobi
  before falling back to open-source SCIP through `ilpy`.
- The associated paper describes an edge-centric transformer designed for
  divisions and reports state-of-the-art Cell Tracking Challenge results. It
  does not report Biohub competition validation, so transfer remains unknown.
- The public `create_graph_from_points` API is currently a documented stub.
  Direct centroid input is therefore not usable at the audited commit; masks
  must be supplied or the missing graph construction must be implemented and
  validated locally.
- FOCUS-3D source is BSD-3-Clause, but its Hugging Face checkpoints are gated
  and the model card currently declares no license. They are excluded from the
  first experiment until access and competition eligibility are unambiguous.

## E0047 feasibility protocol

1. Freeze the exact E0033/EXT0004 validation detections and node IDs.
2. Rasterize each timepoint into non-overlapping anisotropic seed instances.
   Resolve collisions by nearest physical distance so no node is silently
   dropped; verify a one-to-one label-to-node round trip.
3. Run both HOCT `general_v1` and `ctc_v0` with physical scale
   `(1, 1.625, 0.40625, 0.40625)`, window five, and maximum temporal gap three.
4. Convert selected HOCT edges back to the frozen node namespace. Do not change
   node count or coordinates in this first study.
5. Score complete held-out movies with the hash-verified patched evaluator and
   report adjusted edge Jaccard, division Jaccard, TP/FP/FN, runtime, and peak
   memory against the exact 0.933 parent graph.
6. Audit ordinary and division edges separately. A model that improves
   divisions while losing more ordinary edges is rejected even if a partial
   metric looks attractive.

## Promotion gate

HOCT proceeds to integration only if at least one fixed configuration improves
the complete twelve-movie official score, improves both embryos or leaves the
weaker embryo unchanged, and fits a conservative hidden-test runtime estimate
below twelve hours. If full graph replacement loses edge score but produces
useful division candidates, retain only its candidate scores for a second
fusion study; do not submit the raw replacement.

## Offline deployment requirements

Kaggle inference has internet disabled. The existing tracking support pack
already includes most of the required stack, including GEFF, tracksdata,
`ilpy`, PySCIPOpt, Polars, Zarr, and compatible CPython 3.12 wheels. A small
incremental support dataset should include pinned HOCT and spatial-graph
wheels, the selected JIT checkpoints with hashes, and the solver configuration.
Install HOCT without dependency resolution so its optional fast-path Gurobi
requirement does not force a commercial solver; verify the documented SCIP
fallback in a private smoke test before the full experiment.

## Primary references

- HOCT repository: https://github.com/royerlab/hoct
- HOCT paper: https://arxiv.org/abs/2607.11754
- FOCUS-3D repository: https://github.com/yu-lab-vt/FOCUS-3D
- FOCUS-3D model page: https://huggingface.co/Qinghua-thu/FOCUS-3D
