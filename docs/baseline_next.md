# Next measured attempt to exceed the competition baseline

Engineering instruction, 2026-11-15. The target is public **>0.2216324580**;
the preserved best is c1 **0.2021478454**. No new score is asserted here.

## Immediate local work

1. Read the actual latest YOLO full-validation result and checkpoint SHA. Keep
   the same training recipe, optimizer and schedule. If a newer single YOLO
   candidate already passes the comparison/review requirements, finish its CSV
   review and notify the user immediately; do not wait for all 20 epochs.
2. The recorded epoch-2 weaknesses are truck and bus. Test one **fixed** class
   route: YOLO supplies person/bicycle/car/van/tricycle/motor; immutable public-best
   c1 supplies truck/bus. Do not race eight class weights or interpolate model
   parameters. No detector is retrained for this experiment.
3. Obtain an immutable snapshot at the nearest completed, validated YOLO epoch.
   If the running course owns the lease, request its normal STOP boundary and
   wait for the saved epoch/validation/export. Do not kill/freeze an unsaved job.
   Resume that same YOLO course after caching; this is a temporary inference slot,
   not rejection or a fresh COCO initialization. Preserve/archive intentional STOP
   before explicitly resuming, and check the local service's restart policy.
4. Cache full official-val predictions once per frozen pipeline. c1 uses the
   original 960 global+tiled pipeline; YOLO uses 1280 global. Compare both singles
   and the fixed route on **all 2,846 records**, with COCO100/500, classes and
   families. CPU comparison can run while the same YOLO course resumes on GPU.
5. Only an eligible route receives test inference/export. Use the exact frozen
   snapshots from the val comparison, even if live YOLO best has since advanced.
   At the next saved GPU boundary generate each test cache once, then resume
   YOLO. Combining caches and checking the CSV are CPU-only.

The native epoch-2 class deficits (truck 0.07690 AP, bus 0.01805) account for
`(0.07690+0.01805)/8 ≈ 0.01187` macro-AP. This motivates the route; it is not a
measured ensemble gain, an upper-bound guarantee or a public-score forecast.
The actual c1 tiled pipeline and global 500-box cap must be evaluated.

## Commands

Work on the current experiment branch containing this change. Resolve
`CVLAB_LOCAL_WORK`, `C1_CONFIG`, `C1_CHECKPOINT` from actual local state; the
checkpoint must have full canonical c1 SHA
`c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`.
Use the original c1 model configuration, not a rejected continuation's weights.
`YOLO_CONFIG` is the existing `runs/yolo11s_1280/config.json`.
Choose a new `PAIR` directory for a new YOLO snapshot, preserving older evidence.

```bash
PAIR="$CVLAB_LOCAL_WORK/ensemble/pair_epoch_N"
YOLO_CONFIG="$CVLAB_LOCAL_WORK/runs/yolo11s_1280/config.json"

.venv/bin/python -u scripts/class_ensemble.py \
  --work "$CVLAB_LOCAL_WORK" --config "$C1_CONFIG" cache \
  --backend c1 --checkpoint "$C1_CHECKPOINT" --split val \
  --output "$PAIR/c1_val"

.venv/bin/python -u scripts/class_ensemble.py \
  --work "$CVLAB_LOCAL_WORK" --config "$YOLO_CONFIG" cache \
  --backend yolo --checkpoint "$CVLAB_LOCAL_WORK/runs/yolo11s_1280/weights/common_best.pt" \
  --split val --output "$PAIR/yolo_val"

nice -n 15 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u scripts/class_ensemble.py \
  --work "$CVLAB_LOCAL_WORK" --config "$YOLO_CONFIG" compare \
  --c1 "$PAIR/c1_val" --yolo "$PAIR/yolo_val" --output "$PAIR/comparison.json"
```

Snapshot SHA is in each `cache.json` at `identity.checkpoint_sha256`; frozen
weights are `WORK/ensemble/snapshots/SHA.pt`. The cache commands acquire the
shared GPU lease, reject unrelated GPU compute, and check RAM/disk headroom.
Run them under the existing local service/sleep-inhibition scheme. Completed
matching caches return without another model inference; changed identities
require a new directory. Compressed caches are local runtime data, never Git.

If `accepted=true`, or the documented manual-review route below is eligible,
run the two cache commands with `--split test`, destinations `c1_test/yolo_test`,
and **the frozen val checkpoint paths**, not live best. Then:

```bash
.venv/bin/python -u scripts/class_ensemble.py \
  --work "$CVLAB_LOCAL_WORK" --config "$YOLO_CONFIG" export \
  --c1 "$PAIR/c1_test" --yolo "$PAIR/yolo_test" \
  --report "$PAIR/comparison.json" \
  --output "$CVLAB_LOCAL_WORK/submissions/yolo_c1_pair_epoch_N.csv"
```

The output preserves template order, class IDs 0–7, original coordinates,
500 detections per frame, full cache/weights/pipeline identities and CSV SHA.
It refuses missing frames, changed test weights/inference, changed validation,
changed route/evaluator, rejected quality reports and wholly empty predictions.

## Explicit manual review instead of pretending family certainty

Strict automatic acceptance still applies `paired-family-class-v1` against
**both** single pipelines. There is also an explicitly recorded manual-review
route for a fixed pipeline: **pooled gain >=0.001 against both singles**, complete
class/family evidence and >=4 families, no per-class drop >0.005 and no AP_small
drop >0.003 against either. Family bootstrap/positive-fraction/worst-family
warnings remain fully visible. Seven repeatedly inspected families do not provide
a calibrated public-score guarantee; those warnings cannot be relabeled as a pass.

When only these family warnings fail, `review_eligible=true` allows the export
command with `--manual-review`. This prepares a CSV for the user's informed
review; strict promotion remains false and nothing is submitted automatically.
It is not an AP-only bypass. If pooled/class/size/completeness guards fail,
even this flag refuses the export. Do not weaken thresholds repeatedly to admit
an unsuccessful route. Report the failure and keep the stronger single model.

## Report next

Publish a short coordination note with snapshot SHAs, exact full-val comparison,
per-class/family deltas, measured inference cost, acceptance/review flags and,
when eligible, validated CSV filename/hash. The user can then test that candidate
on Kaggle while training continues. Record a public result only after Kaggle
returns it. Final train+val fitting comes after selecting a fixed successful
recipe; it has no independent AP on its now-seen validation labels.
