# Experiment protocol

Use IDs such as `E0001_detector_threshold_calibration`.

Every experiment should provide:

- hypothesis and one primary change;
- parent experiment and git revision;
- grouped fold definition and random seed;
- model/data artifact versions;
- adjusted edge Jaccard and division Jaccard separately;
- node, edge, and predicted-division counts;
- runtime and peak memory;
- post-reset public score, when submitted;
- conclusion and next action.

Change one major factor at a time until the local-to-public relationship is understood. Store large checkpoints and competition data outside Git; record immutable artifact identifiers instead.
