# Toward a score near 0.30: stable learning, then one larger-model comparison

Operational plan, 2026-11-15; not an academic report. The user has now supplied
public score **0.2237335225** for `yolo_c1_pair_epoch_004_fliptta.csv`.
The competition public baseline **0.2216324580 is exceeded by 0.0021010645**.
The new planning target is near 0.30; remaining gap **0.0762664775**. No result
on private evaluation or the 0.30 target is inferred.

## Preserve the measured champion

Keep the CSV, SHA256
`2dee65d0f257ba61569d48ba94ad95e74e740802b7e430cf10556a1fe2ebdce0`, both source
checkpoints, native/flip caches and full-validation reports. The c1 source is
`c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`; learned YOLO
source is `42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8`.
The current route's full official-val COCO100 AP is about **0.216622**, from its
report, distinct from public score. The numerically failed checkpoint is excluded.

## Concrete training fixes

Pinned Ultralytics 8.3.250 omits `amp` from its resume override allowlist. Merely
passing `amp=False` could leave AMP active. `RecoveryTrainer` explicitly applies
the requested mode and the runner verifies actual FP32 mode before updates.
The wrapper also accepts a second resume after its precision-change audit;
omitting a flag cannot silently re-enable AMP after it has been disabled.

Upstream checkpoints store EMA and FP16-converted optimizer moments. The new
`weights/resume_fp32.pt` atomically retains the **raw training model**, separate
EMA, FP32 optimizer state and scaler. Recovery restores raw weights corresponding
to those moments. Invalid numeric state cannot overwrite the previous valid
FP32 recovery file. Inference/epoch files retain their original format; old
c1/native/flip caches keep their identity. This addresses concrete resume risks;
it is not proof that optimizer quantization caused the prior failure.

## First course: learned warm start in FP32

Use a **new run**, preserving the stopped old course and all champion artifacts.
Start from the exact learned YOLO source above, not COCO initialization. This is
an explicitly changed stabilization recipe: FP32, physical batch 1 / nominal
effective batch 16, AdamW LR **0.0001**, frozen BatchNorm statistics, input 1280.
All 24,201 train frames are used; official val stays held out. A fresh optimizer
is intentional for this new lower-LR recipe; do not claim its old moments were
restored. Subsequent resumes of this course do preserve the new moments.

Resolve actual work/config paths locally and verify the source SHA. Launch under
the existing shared lease and sleep-inhibition service:

```bash
.venv/bin/python -u scripts/start_yolo_training.py \
  --work "$CVLAB_LOCAL_WORK" --config "$SOURCE_CONFIG" \
  --name yolo11s_fp32_continuation \
  --init-checkpoint "$GOOD_YOLO_CHECKPOINT" \
  --epochs 20 --hours 4 --batch 1 --imgsz 1280 --workers 2 \
  --no-amp --lr 0.0001 --freeze-batch-norm
```

The source has a full initial validation receipt, once per run. Each completed
epoch has full native validation and a preserved checkpoint. Four hours is a
soft session budget at epoch boundaries, not a measured completion estimate;
20 epochs is a fixed schedule ceiling, not a requirement to wait before review.
Measure actual throughput/headroom in the first running epoch. Repeat the same
command with `--resume` to continue its schedule; `resume_fp32.pt` is preferred.
Do not clear the old failure STOP or resume its invalid state.

## Evaluate the public pipeline every two completed epochs

Native YOLO AP is an interim diagnostic. At epochs 2, 4, etc. compare the new
checkpoint through **the same native+flip inference** with the current scored
champion on all 2,846 official-val records. Reuse c1 predictions. Evaluate both
the fixed truck/bus route and the new all-YOLO output, so old class routing does
not remain mandatory if the new model learns those classes better.

Local agent: the old `yolo_flip_experiment.py` is intentionally bound to the old
one-off source SHA and unflipped parent CSV. Generalize or add a candidate entry
point with explicit new checkpoint and current champion report/CSV identities;
do not replace the historical constants or bypass validation checks. Add tests
rejecting changed parent data, mismatched native/flip/test weights and old-parent
comparisons. This generic current-champion comparison is a required local step;
the one-off exporter is not already a general training controller.

Use >=0.001 full-val gain plus class/size/evidence guards; retain family evidence
and the existing explicit manual-review distinction. Training convergence and
candidate promotion remain separate. Prepare a new validated CSV immediately
when a candidate is eligible. No automatic Kaggle submission and no inferred
public gains. Do not wait for the whole course to report its first two epochs.

## One conditional larger detector

If the stable small model plateaus on several full checks, measure one official
COCO-pretrained **YOLO11m** course using the same data/protocol. The launcher now
supports `--model yolo11m.pt`. Use a new name, batch 1, `--no-amp`, and a bounded
session; measure its first-epoch cost and AP before further allocation. Medium
COCO initialization is a different architecture experiment, not a resume from
the small model. Do not run the two training models concurrently or invent a
GPU speedup. The larger model has not been trained or benchmarked here.

Final training on all labeled train+val comes only after choosing a fixed
recipe. It needs a separate final-fit mode with no independent AP on seen val;
do not feed merged labels into the current selection evaluator. Full CV/OOF,
the student-authored report and private-leaderboard outcomes remain separate
requirements. A 0.30 result cannot be promised from these steps.
