# Confirmed public score — 2026-11-15

The user supplied a Kaggle screenshot showing a new personal best **0.2176009680**, improving the previous **0.2021478454**. The gain is **0.0154531226**. The competition baseline is **0.2216324580**, so the remaining gap is **0.00403149**; baseline and the separate 0.28+ goal are not reached.

The user associates this result with `yolo_c1_pair_epoch_004.csv`, SHA256 `1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562`. File identity is from the supplied local report; the web agent has not opened the private CSV or weights.

This score belongs to the frozen class ensemble: c1 for truck/bus and YOLO for the other six classes. Both exact input checkpoint hashes and the observed score are recorded in `experiments/leaderboard_results.json`. Preserve the original CSV, source checkpoints, prediction caches and provenance metadata locally. The family uncertainty from validation remains in the experiment record.

The next bounded experiment is specified in [next_after_02176.md](../next_after_02176.md): original versus horizontal-flip YOLO inference on the same valid frozen weights, retaining the c1 caches. The adapter and its outcome have not yet been implemented or measured by this documentation update. No automatic Kaggle submission is authorized by this note.

The existing YOLO and class-ensemble implementation tests passed on CPU (12 tests). This is separate from the real public score supplied by the user. No new GPU result is inferred.
