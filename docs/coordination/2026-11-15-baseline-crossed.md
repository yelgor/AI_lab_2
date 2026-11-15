# Baseline crossed and next learning stage — 2026-11-15

The user supplied a Kaggle screenshot for
`yolo_c1_pair_epoch_004_fliptta.csv`: **0.2237335225**. This exceeds public
baseline **0.2216324580** by **0.0021010645** and improves the prior personal
best by **0.0061325545**. The ledger now records it as the observed public best.
Its CSV/source identities come from the local-agent record; private files were
not accessed by the web agent. The new user goal is a score near 0.30, not an
already measured result.

Preserve the exact champion and its caches. The next priority is the
[stable learning course](../toward_030.md), using the learned good YOLO source
in a new FP32/lower-LR run. Code now enforces explicit AMP disable during resume,
keeps a raw-model FP32 optimizer/EMA recovery checkpoint, fixes repeated resume
after precision-audit metadata, and supports an explicit learned initialization
and a separately selected official YOLO11m experiment. The old failed course's
STOP stays in place. No healthy process is interrupted by publishing this code.

Local agent: verify actual state and execute the first course from the command
in `toward_030.md`; do not restart from COCO or use failed weights. Measure the
first full epoch and report actual cost/AP. During that course, parameterize
the one-off flip evaluator to compare each new frozen candidate against the
current scored flip champion, preserving historical constants/receipts. At each
two-epoch boundary evaluate the real pipeline and produce an eligible CSV.
Reuse c1 caches; published code has not launched this laptop training itself.

Tests cover real CPU training, raw-versus-EMA resume, exact unquantized optimizer
restoration, actual AMP argument forwarding, repeated precision-policy resume,
learned-weight initialization and preservation of the previous recovery file
when a candidate state is nonfinite. These establish implementation behavior,
not a new GPU speed or score near 0.30. Record local measured outcomes in a new
sanitized coordination note; keep machine paths, raw telemetry, weights and
datasets local.

Targeted regression: **55 passed** (YOLO training/recovery, class ensemble,
horizontal flip and watchdog). Ruff, compile and diff checks pass. The medium
model and the new FP32 course have not run on the laptop through this web session.
