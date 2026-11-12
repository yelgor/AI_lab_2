# C1 checkpoint-direction search

This is a technical experiment recipe, not a Kaggle-score guarantee or a
student-written conclusion.

## Hypothesis and method

Let `theta0` be the checkpoint that produced the precise Kaggle score
`0.2021478454` (earlier user report rounded it to `0.206`), and
`theta1` the later continuation checkpoint. Evaluate the finite update direction
with

```text
theta(alpha) = theta0 + alpha * (theta1 - theta0)
```

for `alpha` in `[0, 1]`. This is a line search on a saved weight update; it does
not train a new model or claim that the Kaggle metric is differentiable. The
first fixed 512-frame probe yielded local COCO100 mAP `0.20747`, `0.21445`,
`0.22063`, `0.22243`, `0.22249` at alpha `0`, `.25`, `.5`, `.75`, `1`. It
increased through `.75` and then plateaued. The endpoint nevertheless scored
`0.1919` on Kaggle versus `0.2021478454` for the source. The local curve therefore
justifies testing a partial step, but does not predict leaderboard gain.

The next predeclared candidate is alpha `.5`, halfway back from the regressed
endpoint. The runner evaluates it on the complete 2,846-frame official native
and tiled validation cohort. It exports a test CSV only if both metrics improve
by at least `0.001` against the source checkpoint under the fixed tiled settings
(960 tile, 0.2 overlap, NMS 0.6, edge margin 2). An exported CSV still needs a
fresh Kaggle evaluation to establish whether public score improved.

Limitations: the line search uses one observed update vector; the 512-frame probe
and full official validation share the same seven video families; public score
feedback is sparse; the Kaggle evaluation implementation has not been verified
against local COCO100. Do not reuse the regressed endpoint as a training source.

## Run and inspect

Set `CVLAB_LOCAL_WORK` to the configured local work directory. The source and
candidate checkpoints must both be readable; work storage must be writable and
have space for one checkpoint and a CSV. The script acquires
`$CVLAB_LOCAL_WORK/.local_gpu_queue.lock` and blocks system sleep for the run.

```bash
.venv/bin/python -u scripts/export_c1_interpolated_candidate.py \
  --work "$CVLAB_LOCAL_WORK" \
  --source "$CVLAB_LOCAL_WORK/source_c1d7fcc83dde/best.pt" \
  --candidate "$CVLAB_LOCAL_WORK/runs/rtdetr_c1_continuation_train_selection/best.pt" \
  --alpha 0.5
```

Output is grouped under
`$CVLAB_LOCAL_WORK/runs/rtdetr_c1_path_alpha_0.500/`; the report is written
after native validation and again after tiled validation. A passing run writes
the new file under `$CVLAB_LOCAL_WORK/submissions/`. A failure of either full
validation gate leaves a report and checkpoint but creates no submission.
