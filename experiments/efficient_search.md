# Bounded warm-start search

**Protocol update (2026-11-11):** the user now targets public score **0.28+**
and requests full-data coverage, including train+val for the final fit after
independent model selection. Read [the new full-data protocol](../docs/full_data_protocol.md)
and [coordination notes](../docs/AGENT_COORDINATION.md) before launching another
recipe. The implementation described below remains the original fold-based racer;
this documentation update does not change a running process or implement the new
sampler/final-fit mode. The newest observed public-best is **0.1967910091** for
`rtdetr_51bc69d304b8_efficient_tiles.csv`; `a7cd89fb349d` is a historical fallback.

This is a technical experiment protocol, not the homework report. The original
public reference for this recipe was RT-DETR `a7cd89fb349d`: tiled Kaggle AP 0.1928780288.
The target is 0.2216324580. An accepted local checkpoint is not a verified public
improvement. Keep the original weights and submission throughout the search.

## Mathematical policy

Optimize the differentiable detector loss with AdamW, starting from the learned
weights **and optimizer moments, AMP state and RNG** of the champion. The outer
search is derivative-free: AP is not a training loss and is not assumed convex.
Use a bounded local pattern search in log learning rate:

`x = log10(LR_head); candidates = x + {-r, 0, r}; initially x=-5, r=0.5`.

The backbone/head LR ratio, source training fold, architecture, input resolution
960, augmentation and loss stay fixed. BatchNorm moments are frozen during
training. Initial trials start with the same source RNG to reduce comparison
noise. Following an unsuccessful cycle, halve the log search radius; after a
successful cycle, expand it by 1.2 up to 1.0 and center on the accepted LR.
This is a local pattern search, not a Gaussian-process surrogate or a convergence
guarantee. It searches one sensitive parameter before paying for wider tuning.

Use cumulative training budgets `6000 -> 12000 -> 24000` frames, with batch size
one. Evaluate all three LRs at the first budget, retain at most two for the
second and one for the third. Resume a surviving trial's optimizer/RNG and add
only the missing budget. This uses at most **42,000 training frames** per complete
cycle versus 72,000 for fully training all three configurations: 41.7% fewer
training frames. Validation/export time is additional, so this is not a claim of
a 41.7% reduction in total runtime. Exact previously measured LR/source states
are reused across cycles; they are not trained again.

The fast probe contains up to 512 temporally spread held-out frames. Exclude
every `uavNNNNNNN` flight family seen in training, including other excerpts of
the same flight. The complete original training fold is preserved. Require at
least four unseen families before spending training compute.

For family `v`, measure the paired change
`d_v = COCO100_AP_v(candidate) - COCO100_AP_v(source)` on identical frames.
Bootstrap whole families 2,000 times to obtain approximate 90% lower/upper
bounds `L,U` on the equal-weight mean family gain. This auxiliary macro family
score differs from global COCO AP. Frames are not treated as independent trials.
The interval is an allocation heuristic: small family counts, repeated looks
and selection prevent interpreting it as a calibrated hypothesis test.

Prune observed clearly harmful trials (`U <= -0.001`). Rank the remainder by
optimistic improvement per estimated extra wall time:

`priority = max(U - 0.001, 0) / estimated_seconds_to_next_budget`.

Use measured training/validation time per batch for the cost estimate. Uncertain
trials can continue; a negative point estimate alone does not prove harm.
Do not replace the champion based on loss or a single probe AP increase.

## Confirmation and feedback

At each stage, the strongest candidate with `L > 0` and global probe AP gain
at least 0.001 can receive full confirmation. At most one candidate is checked
per stage. Promotion requires BOTH:

1. Full original-fold native COCO100 AP gain >=0.001 against the current champion.
2. Full official-validation tiled COCO100 AP gain >=0.001 against that champion,
   with the original fixed tile size 960, overlap 0.2, NMS 0.6 and edge margin 2.

Baseline full confirmations are cached by checkpoint SHA, record fingerprint,
model, precision and evaluation options. The verified original tiled result is
reused only when the source SHA, records, resolution and tile settings match.
No new resolution/NMS search is mixed into this training comparison. A CSV is
exported only after both full checks pass. It may therefore be available after
the first stage; absence of a CSV is a deliberate rejection, not hidden progress.

Every rung preserves its actual measured checkpoint independently of `best.pt`
inside the trial. A later regression cannot overwrite an earlier useful state.
An interrupted completed validation is read from history instead of repeating
the training. Pending validation resumes from the saved pre-validation state.
The original public-best run is never overwritten.

`--hours` is the wall-time allowance per invocation **after acquiring the GPU
lease**. `--reserve-hours` reserves estimated time for confirmation/export.
Training checks the remaining search budget at optimizer boundaries, commits
its current checkpoint and finishes validation. Validation and export finish
their current phase rather than being forcibly killed, so this is a soft limit.
No unrelated active training is stopped. The launcher waits for the GPU lease.

After reviewing a worse public submission, use `--reset-to-original` on the next
launch to return to the known public-best weights, shrink the local search
radius and retain all prior evidence. This is an explicit local decision;
the runner does not submit to Kaggle or infer public scores automatically.

## Run and inspect

```bash
WORK="${CVLAB_LOCAL_WORK:?Set CVLAB_LOCAL_WORK to your actual work directory}"
.venv/bin/python -u scripts/start_efficient_search.py \
  --work "$WORK" --hours 4 --reserve-hours 1 --cycles 1 --export-tiles
```

Repeat the same command to resume an unfinished cycle. Once it completes,
another invocation starts a new cycle from the locally accepted champion.
State: `efficient_search/state.json`; runs: `runs/rtdetr_efficient_search/`.
Each `PROBE` line reports actual training frames, global gain, family gain and
the approximate interval. Each full `ACCEPT/ROLLBACK` records its measured gains.

To stop at the next committed boundary:

```bash
mkdir -p "$WORK/runs/rtdetr_efficient_search"
touch "$WORK/runs/rtdetr_efficient_search/STOP"
```

Remove this one marker before intentionally resuming. Public baseline attainment
must be checked using the uploaded candidate's actual Kaggle score.
