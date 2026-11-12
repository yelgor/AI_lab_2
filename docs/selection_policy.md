# Candidate selection and evaluation reuse

This is an engineering protocol, not an academic report or a score guarantee.
It replaces pooled-AP-only promotion for new full-data, efficient-search and c1
interpolation invocations. Existing running Python processes retain their loaded
code; apply updates at a safe saved boundary, preserving healthy work and the GPU
lease. The original public-best c1 checkpoint remains immutable.

## Acceptance

Both native and fixed tiled full-validation comparisons must pass
`paired-family-class-v1` (`paired-family-class-v3` in efficient-search confirmation
records):

- Pooled COCO100 AP gain at least 0.001 on identical labeled records.
- At least four supervised-unseen flight families and at least two thirds with
  positive paired family AP gain (five of seven for the current official val).
- Positive lower endpoint of the approximate 90% whole-family bootstrap interval
  for equal-family mean AP gain. Removing any one family must leave a positive
  mean gain.
- No evaluated class loses more than 0.005 AP, no family more than 0.01 AP, and
  AP_small loses at most 0.003 when that size has evaluable ground truth.
- Missing/incompatible family or class evidence rejects the candidate. Changing
  records or inference invalidates a cached confirmation. Recorded public
  regressions cannot become automatic training sources or be re-exported as
  improvements under the same checkpoint identity.

These fixed conservative thresholds are allocation heuristics. Seven families,
repeated selection and sparse public feedback do not support a calibrated
significance or Kaggle improvement claim. Family AP and pooled AP are different
estimands. Exact parity with the competition Evaluation implementation still
needs verification. Do not tune these thresholds simply to admit a rejected run.

Full-data training selects `best.pt` from candidates passing the native guards
against initialization, ranked by family lower-bound gain then pooled gain.
A higher pooled AP alone cannot overwrite a robust checkpoint. Each row logs
`selection_decision`; a completed full-data epoch with an excessive class/family/
small-object regression writes `selection_halt.json` and prevents another full
epoch on automatic restart. Preserve that evidence; a changed recipe uses a new
plan/run, warm-started from the approved source. Mild or uncertain probe gains
may receive the remaining planned budget. Rejected full validation skips tiled
inference/export and final fitting. Final train+val remains a separate fixed
recipe with no independent AP on those now-seen labels.

## Saved update directions

`theta(alpha) = theta_source + alpha * (theta_candidate - theta_source)` explores
one finite weight-update direction. A monotone local AP curve is not an estimate
of the gradient of Kaggle score. The interpolation exporter now measures its
actual source and interpolated weights through the shared evaluator, verifies
their identities, and applies the same robust gates. It no longer imports a
baseline number/fingerprint from a hard-coded plan. Output directories include
source/target/alpha identity, and reruns preserve the same saved artifact.

## Avoiding repeated GPU work

Training and confirmation share `evaluation_cache/` receipts. They bind exact
checkpoint SHA, full labeled-record digest, model/inference options, precision,
runtime, software versions and evaluator code. Native/tiled results are distinct.
Loader worker/pinning changes do not invalidate deterministic batch-one
inference; runtime thread-count changes do. Original source weights are bound
before the first gradient step as well as newly committed checkpoint files.
A cache hit skips CUDA setup/model inference. Old pooled-only caches cannot pass
the new gates. The cache logs hits and newly measured evaluation wall time.

This change does not implement mid-epoch sampler recovery or a hard deadline for
full epochs. Those remain operational limitations; do not claim their removal.
Prevent suspend/lid-induced interruption using the actual local job configuration.

## Inspect locally

`python3 scripts/check_lab.py --work "$CVLAB_LOCAL_WORK" --out "$CVLAB_LOCAL_WORK/checks/status.json"`
reads local metadata, active process argv and plan states without tensor loading
or restarting training. DataLoader workers may share training argv; several
matching PIDs do not prove several models. The lightweight timer accepts explicit
`--log` and `--state` paths; automatic newest-file selection is a fallback, not
proof of job ownership. No CSV is submitted automatically.

The original auto-launch watchdog recognizes named full-data plans as well as
the default plan. If any such plan has recorded status, it monitors live work
but pauses legacy efficient-search fallback when idle. Its old a7-based recipe
must not start silently after c1 full-data selection finishes or rejects a run.
The local agent must use the dedicated approved-plan service to resume c1 work.
