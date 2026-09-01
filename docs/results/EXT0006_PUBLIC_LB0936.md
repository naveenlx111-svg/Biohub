# EXT0006: reproducible public 0.936 candidate

The public notebook `nusrati/0-936` is the strongest reproducible public
candidate found on 2026-09-01. Its completed output contains a valid
`submission.csv` with `119,517` nodes and `115,354` edges. The output SHA-256
is `f9d42e27f6b2cbeba1ea8f433087fba45be7742b41b38d271c4109339e9279c4`.

The local file `/home/naveen/Downloads/0-935.ipynb` has exactly the same
normalized executable-code SHA-256 as the public notebook:
`9482b88a5fb76bd10f2da36036b2ccdcc768f1362c8458fbef7b3a7936665faf`.
The raw notebook hashes differ only because notebook metadata and stored
outputs differ.

Two apparent alternatives do not establish a stronger method:

- `leolin05/biohub-0-935-reproduction-audit-and-validation` produces a
  byte-identical submission to the public 0.936 output;
- `rishabhr0y/biohub-948-sew20` has byte-identical executable code to the
  public 0.936 notebook, and its latest public execution stops during offline
  dependency installation. Its `0.948` title is not score evidence.

Relative to the E0033 local parent configuration, the most important visible
production choice is DeepCenter epoch 2 with the safe-division veto enabled;
E0033 used epoch 500 with that veto disabled. EXT0006 must remain a
leaderboard candidate until a team submission receives a score. The private
team notebook titled `0.935` ran on 2026-09-01 and contains the same executable
pipeline, so its completed `submission.csv` is the preferred immediate
submission candidate.
