# After the measured public score 0.2176009680

Operational handoff, 2026-11-15. The user supplied a Kaggle screenshot showing
this new personal best, plus the local CSV identity. The competition baseline
is 0.2216324580; the remaining gap is **0.00403149**. Target 0.28+ remains separate.

## Preserve the result

- CSV: `yolo_c1_pair_epoch_004.csv`, SHA256
  `1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562`.
- c1 source:
  `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`.
- YOLO source:
  `42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8`.
- Preserve their full-val/test caches, accepted manual-review comparison and CSV
  metadata. Freeze the pair as the public-best pipeline. This public score belongs
  to the ensemble, not either detector alone. The filename is not evidence that
  its source was the numerically damaged epoch-4 weights; those were excluded.
- Public success does not retroactively turn the negative family interval into
  a positive one or prove private-leaderboard performance.

## One bounded next A/B: YOLO horizontal-flip inference

The YOLO course is stopped after a genuine numeric failure: the recorded checkpoint failed numeric validation. Do not resume that
state or clear its STOP because a different frozen pipeline scored well.
Training recovery is separate from the next inexpensive inference experiment.

Local agent, carry out this single predeclared test:

1. Use the exact good frozen YOLO source above. Add a horizontal-flip inference
   adapter in an isolated module, leaving c1 model/inference code and working
   caches unchanged. Compare original global YOLO against original+flip; keep
   input 1280, fixed class-aware NMS 0.6 and cap 500. No new training or LR sweep.
2. Unflip boxes in tensor/canvas coordinates before restoring original geometry.
   Test odd image dimensions/padding, duplicate boxes, classes and the global cap.
   Bind the new adapter and flip flag explicitly in its cache identity. Native
   caches must retain their old identity; flipped predictions need a new cache.
3. Run only the new YOLO full-val inference under the shared GPU lease. Reuse
   the exact c1 global+tiled cache. Reuse prior c1 metrics only when checkpoint,
   labeled-record digest, pipeline and evaluator identity all match.
4. Compare the resulting route with the **current public-best route** on all
   2,846 identical records. The local parent AP is about 0.202613 (read the full
   precision from its report). Comparing only with old c1 or a 192-frame probe
   is insufficient. Record COCO100/500, AP_small, per-class and per-family deltas.
5. Require at least +0.001 full-val pooled AP and retain the existing class/size/
   completeness guards. Keep family uncertainty explicit under manual review.
   If the fixed test fails, preserve it as a negative result and keep the scored
   parent; do not start a large TTA/NMS/resolution search in response.
6. Only after this full comparison, compute the new YOLO test outputs and combine
   with the existing matching c1 test cache. Do not recompute c1 or overwrite the
   scored CSV. Validate all 3,811 IDs/order, coordinates, classes, cap, hashes and
   exact val/test pipeline identity. Notify the user of a genuinely new candidate;
   nothing is submitted automatically.

The flip adapter and full A/B are now implemented and measured; see the latest
dated result in [AGENT_COORDINATION.md](AGENT_COORDINATION.md). The route passed
the local selection guard and has a new CSV for manual Kaggle review. Its Kaggle
score remains unknown; local AP does not predict or guarantee the leaderboard.
The original public-best CSV and frozen pair remain unchanged. Students retain
responsibility for their own academic report.
