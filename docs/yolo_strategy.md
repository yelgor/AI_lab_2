# YOLO11s challenger: a different learning strategy

Engineering plan, 2026-11-14. Not a student report or an achieved score.

The observed public-best RT-DETR c1 remains **0.2021478454**. The competition
baseline is **0.2216324580**, and 0.28+ is an unachieved goal. Longer RT-DETR
continuations have not demonstrated the required generalization. The next main
experiment changes the detector and uses one continuous course rather than more
one-epoch VFL/LR trials or weight-direction interpolation.

## Fixed first recipe

| Setting | Value |
| --- | --- |
| Detector/source | YOLO11s, official Ultralytics COCO `yolo11s.pt` |
| Images | 1280 square letterbox; no tiling in the first experiment |
| Data | Every prepared train frame, frame_step=1; official val held out |
| Optimizer | AdamW, LR 0.001, cosine to 0.05 of LR, one warmup epoch |
| Physical/effective batch | 2 / nominal 16; stock warmup accumulation |
| Augmentations | Stock detector pipeline, mosaic=0.5, scale=0.3; close mosaic for last 3 epochs |
| Course | Maximum 20 epochs; soft session budget 10h at full epoch boundaries |
| Convergence stop | After at least 5 epochs, 3 checks without >0.001 AP progress |
| Checkpoints | Stock EMA + optimizer/scaler, independent epoch files, common-AP best |
| Validation | Every complete epoch, exact serialized weights, all official val records |
| Manual feedback | Optional CSV of validated training best every 2 epochs |

The stock Ultralytics training loop owns forward/backward, accumulation,
warmup, EMA and schedule. The adapter adds dataset conversion, ignored-region
masking and our common evaluator. Stock YOLO validation is replaced because it
would treat competition ignored regions as ordinary background. Validation
images remain unmasked; the existing COCO crowd rules handle ignored regions.
Every class remains 0–7 in YOLO labels/CSV and 1–8 internally in the common evaluator.

This is a testable architectural hypothesis: a smaller CNN at higher input
resolution may use the laptop's compute more effectively for small objects.
GPU frames/s, memory and AP gains are **not measured yet**. Keep physical batch
fixed once training begins; do not silently fall back to another image size or
batch after OOM. Inspect the first epoch's throughput and headroom first.

## Prepare while current GPU work finishes

The local agent first reads its actual watchdog config and defines
`CVLAB_LOCAL_WORK` and `SOURCE_CONFIG`. The latter is an existing JSON/YAML
configuration with valid `data.root` and `data.prepared`; only those two paths
are used. Output must be writable Linux storage; source images may be read-only.

Install the optional backend in the existing environment. Core Torch/TorchVision
pins do not change; preserve the already-working CUDA build:

```bash
.venv/bin/python -m pip install -e '.[yolo]'
.venv/bin/python -u scripts/start_yolo_training.py \
  --work "$CVLAB_LOCAL_WORK" --config "$SOURCE_CONFIG" --prepare-only
```

For a uv-managed environment without pip use `uv pip install --python .venv/bin/python -e '.[yolo]'`.
Do not reinstall CUDA/driver components to install this optional Python backend.
The converter creates image symlinks plus small label files under `WORK/yolo_data/`;
it does not copy the 10GiB image archive. It checks manifest identity and
train/val flight-family isolation. No train+val mixing or unseen-label claims.

## Launch, continue, inspect

The agent must inspect current work locally and let a healthy epoch/validation
finish before changing owners. To prioritize this challenger, pause further
RT-DETR epochs at a saved boundary, keep all weights, and avoid automatic
watchdog fallback. Do not signal/kill the current GPU process merely to try YOLO.

```bash
mkdir -p "$CVLAB_LOCAL_WORK/logs"
nohup systemd-inhibit --what=sleep:idle --mode=block \
  --why="YOLO challenger training" \
  .venv/bin/python -u scripts/start_yolo_training.py \
  --work "$CVLAB_LOCAL_WORK" --config "$SOURCE_CONFIG" \
  --name yolo11s_1280 --epochs 20 --hours 10 \
  --batch 2 --imgsz 1280 --workers 2 --export-every 2 --export \
  >> "$CVLAB_LOCAL_WORK/logs/yolo11s_1280.log" 2>&1 </dev/null &
```

This launcher waits for the shared GPU lease, rejects duplicate YOLO launchers
and unrelated GPU compute, and checks RAM/storage headroom. It is not a second
simultaneous GPU training process. The watchdog recognizes its process as active.
The soft budget can overrun by the rest of the current epoch plus validation/export;
it is not a claimed 10h completion guarantee.

Use `tail -n 20 -f "$CVLAB_LOCAL_WORK/logs/yolo11s_1280.log"` for progress.
The stock batch progress and `results.csv` measure training; `common_history.jsonl`
records authoritative COCO100/500 AP, per-class/small-object and per-family AP,
checkpoint SHA and full validation fingerprint. `status.json` records the last
complete validation. Initial head adaptation is expected: COCO has 80 classes,
the competition 8. Do not stop on the first low AP or compare numeric YOLO loss
against RT-DETR loss.

Repeat the exact command with `--resume` after a normal pause. The recorded
recipe/data must match; optimizer/scaler and the stock schedule resume from the
last saved epoch. This is standard Ultralytics resume, not a bit-exact claim
about sampler/RNG replay. Resume is not appropriate after all 20 scheduled epochs.
The wrapper prevents stock finalization from stripping optimizer state.

To request a clean pause, create `WORK/runs/yolo11s_1280/STOP`; it is honored
after the current saved epoch and validation. Preserve/archive that intentional
STOP before an explicitly requested continuation. Unsaved partial epochs can repeat.
Twenty independent epoch files plus best/last need disk space; no automatic
deletion of earlier experiment artifacts is performed.

## CSV review and next decision

`--export-every 2` exports the current common-AP best every two completed epochs,
not the latest worse epoch. Repeated identical hashes reuse an existing CSV.
`export_review.json` binds its weights, full validation and structural CSV check.
Files are **QUALITY_REVIEW_REQUIRED**, not automatically promoted or submitted.
Compare the full held-out pipeline against c1, inspect class/family regressions
and the validation/test fingerprints, then notify the user of a genuinely new
candidate. Do not tune against repeated public feedback as if it were clean CV.
An exporter failure preserves healthy training and leaves `export_error.json`.

If a complete course is worse than c1, record it and stop allocating to it;
do not launch automatic recipe roulette. If it improves held-out evidence,
evaluate a single controlled extension (longer same course, resolution, or an
ensemble using saved predictions). Final train+val requires a separately fixed
recipe after selection; no honest validation AP exists on labels already trained on.

The adapter has CPU integration coverage for real stock training/backward,
checkpoint/optimizer persistence, resume, exact-weight common validation, masking,
class mapping and CSV. No laptop GPU throughput or Kaggle gain is asserted.

Sources: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[training documentation](https://docs.ultralytics.com/modes/train/), and
[pinned implementation 8.3.250](https://github.com/ultralytics/ultralytics/tree/v8.3.250).
The optional backend retains its upstream licensing.
