# Training efficiency review — 2026-11-12

Technical operations/code review of commit `e6fbd76059eae31e3bd2eadc1b7984afe9482b49`.
This is not an academic report. Repository evidence was reviewed remotely; no
live laptop access, GPU training, service changes or Kaggle submission occurred.

## Main result: the selection signal reversed the public ranking

| Checkpoint | Full official native COCO100 AP | Full official tiled COCO100 AP | Kaggle public |
| --- | ---: | ---: | ---: |
| c1d7fcc83dde, preserved source | 0.1821884 | 0.1879994 | 0.2021478454, precise user screenshot |
| 224c8cc6f987, two full-train passes | 0.1962149 | 0.1988935 | approximately 0.1919, user report recorded locally |

The continuation consumed 48,402 frame presentations: two complete unique passes
over 24,201 training frames. Coverage is now measured, so the old explanation
that only a fixed 6,000-frame prefix is trained no longer describes this run.

Almost all native AP gain is concentrated in truck and bus:
`(0.0543 + 0.0560) / 8 = 0.0137875`, about 98% of the reported 0.0140264
gain. Person and car declined. All 264 bus validation boxes occur in one of
seven validation flight families. Four families improved and three worsened.
These measurements explain why the pooled selection signal is fragile; they
do not prove the cause of the public regression or establish private-test quality.

## Findings and implementation priorities

1. **P1 — a known public regression remains eligible for automatic final-fit
   initialization.** `cvlab/full_data.py:182–215` selects by pooled native and
   tiled AP and subsequently uses the persisted selected checkpoint. It does
   not consume the negative public result in `experiments/leaderboard_results.json`.
   The two gates share the same seven families; they are different inference
   checks, not two independent data confirmations. The local STOP marker
   currently provides the documented manual barrier. Preserve that barrier.
   Respect the user's latest c1-only training-source instruction by pinning the
   full approved source SHA in the active plan and checking it before launch.
   Keep 224 as evidence; do not silently restart training from it.

2. **P1 — interpolation gates do not verify their baseline provenance.**
   `scripts/export_c1_interpolated_candidate.py:82–104` accepts source/candidate
   arguments but unconditionally reads `full_data/c1_continuation/selection_report.json`.
   It copies the report fingerprint rather than computing/checking the current
   cohort and does not verify source SHA or inference identity. Changed
   arguments or data can therefore compare against unrelated AP and falsely
   pass. Check source and target SHA, actual record digest, precision, image
   size, cap and tile options before inference. Bind the output directory to
   these identities and preserve existing artifacts when they differ.

3. **P2 — repeated full evaluation consumes GPU time without new evidence.**
   The engine already measures source initialization and each completed epoch
   on full official native validation. `full_data.py:179–186` then asks
   `_score` for source and candidate native scores again, followed by both
   full tiled scores. The confirmation cache cannot ingest the engine's
   existing measurements. Add a shared, verified evaluation receipt keyed by
   checkpoint SHA, cohort, preprocessing, precision, cap and inference settings.
   Reuse only genuinely identical evaluations; a bare `best_metrics.json`
   without verified weight identity is insufficient. Record phase wall time,
   cache hits and training steps so the GPU-time split can be measured rather
   than guessed. The documented full-epoch duration suggests a meaningful
   opportunity, but the exact wasted percentage is not known.

4. **P2 — long epochs have no mid-epoch recovery and the outer budget is soft.**
   `full_data.py:102–133` explicitly recovers only committed epoch boundaries.
   A suspend/crash before checkpointing loses the unsaved portion of a long
   pass. The reported lid-close, CUDA reset, mount failures and reboot made
   this operationally relevant. Save at optimizer-step boundaries periodically
   with optimizer/scaler/RNG, permutation and the cursor of actually consumed
   frames; prefetch must not advance the committed cursor. Reuse the same epoch
   order after restart. The outer deadline is checked only between epochs,
   while the engine receives `max_hours=48`; propagate the remaining budget
   and reserve confirmation/export time. Do not claim hard budget enforcement.

5. **P2 — supervision is split across incompatible plans.**
   The new read-only `scripts/training_watchdog.py:44–76` still reads
   `efficient_search.log` and `efficient_search/state.json` for every plan.
   It misses direct `-m cvlab train --config` and export-only processes.
   The original `cvlab/training_watchdog.py` worker still launches
   `start_efficient_search.py`. Adding full-data process recognition did not
   make it dispatch the approved c1 plan. Bind supervision to one active-plan
   receipt: source SHA, owner PID/start time, lease, state and log path.
   Inspect locally which units are enabled before changing them. Tie lid/sleep
   protection to that actual job, not a stale list of service names. Keep the
   shared GPU lease and preserve healthy running work.

## Next experiment, after these checks

- First verify the exact competition Evaluation implementation against local
  COCO100/500 on a small known prediction fixture. The metadata name
  `metric_template` alone does not establish a different metric; the Evaluation
  API was inaccessible in the local report. Formula parity remains unknown.
- Do not expand the alpha grid based only on its monotonically improving
  seven-family proxy: its alpha=1 endpoint already reversed the public ranking.
  One already-running, predeclared alpha=0.5 test may finish; it is a hypothesis,
  not a demonstrated improvement.
- Broaden selection evidence with flight-level and per-class deltas and paired
  uncertainty, including supervised-unseen training families when compatible
  with the source. Record all previous selection exposure. New subsets from
  already consulted data are not untouched tests. An honest full GroupKFold
  cannot initialize a held-out fold from c1 if c1 already trained on it.
- Keep all-data fitting as the final fixed recipe after selection. Do not
  interpret AP on training-exposed val as independent evidence or abandon the
  user's eventual all-labeled-data goal. Do not introduce several changes in
  one trial. Compare one controlled warm-start change per budget and retain
  the source on rejection.
- GPU utilization alone is not the target. Track held-out improvement per GPU
  hour, lost/repeated samples, validation/export time and cache reuse. More
  GPU concurrency on an 8 GiB laptop does not resolve a bad selection signal.

## Verification and limits

In an isolated checkout of the reviewed SHA, 69 targeted CPU tests passed:
full-data, generalization training, sampler, coverage, validation isolation,
efficient search and training watchdog. GPU numerical behavior, current units,
filesystem state and the true competition metric were not verified here.
No training/runtime files or academic deliverables were changed.
