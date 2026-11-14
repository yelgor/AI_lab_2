# Continuous training instead of repeated one-epoch rejection

Engineering change, 2026-11-14. This is not the student report.

The recorded batch-2 full epoch improved native official AP from 0.1822 to
0.1917, but class/family guards prevented a second epoch. The VFL=1.5 attempt
also ended after one improving epoch. These observations do not establish
generalization, but they do show that a promotion heuristic was also acting as
an overly early convergence controller. Repeatedly returning to c1 with a new
recipe is not equivalent to training one recipe to convergence.

## Changed behavior

- New full-data plans default to `continuous`: one fixed model/LR/data recipe,
  full training passes, complete official validation after each pass.
- Training `best.pt` follows native AP. Class/family diagnostics are retained,
  but do not terminate the course after a single check. A plateau requires
  three checks without more than 0.001 AP progress. Existing final promotion
  gates remain in force; a training best is not an approved public best.
- Each completed epoch has a preserved checkpoint/metric sidecar in `epochs/`.
  `last.pt` resumes model, optimizer, scheduler, scaler and RNG. No automatic
  return to c1 between full epochs, LR changes or new trial allocation.
- New plans have five selection epochs by default. Existing plans keep their
  recorded epoch count, recipe and deadline. A policy transition cannot silently
  turn the current three-epoch plan into a new five-epoch schedule.
- `--training-policy continuous` explicitly adopts a legacy plan. Old automatic
  halt evidence is archived rather than deleted; a user-created `STOP` remains
  effective. Resume compatibility permits only these policy fields to change.
- Final native+tiled selection and train+val fitting remain separate. A class
  regression may still prevent final promotion even when training continues.

## Existing local LR-half control

The most recent local note names `c1_lrhalf_fulltrain`, initialized via
`source_c1_lrhalf`, batch=1/accumulation=4, three planned epochs. Preserve that
exact source configuration and schedule. At a saved/validated boundary, the
local agent can resume the same plan with:

```bash
.venv/bin/python -u scripts/start_full_data_training.py \
  --work "$CVLAB_LOCAL_WORK" \
  --source "$CVLAB_LOCAL_WORK/source_c1_lrhalf" \
  --plan-name c1_lrhalf_fulltrain --selection-epochs 3 \
  --batch-size 1 --training-policy continuous
```

Do not launch this while its existing owner is running. The lease waits rather
than interrupting it. Check the actual service/plan first; the Git note is not
a live measurement. An expired recorded deadline is not extended by this command.
The main new architecture experiment is [YOLO11s](yolo_strategy.md); avoid
queuing another sequence of RT-DETR recipes ahead of it.

This change does not implement mid-epoch sampler recovery. An interruption
before the epoch checkpoint still loses the unsaved portion. Prefer a `STOP`
file and wait for the saved boundary; never freeze an unsaved process and then
shut down the laptop.
