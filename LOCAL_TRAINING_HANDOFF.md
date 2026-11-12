Original full official validation native AP 0.16764486387978958; tiled AP
0.18545567592355547. Official validation has 2846 frames; test has 3811.
Original fold training has 16121 frames; held-out fold has 8080.
All competition training has 24201 frames. Earlier all-training fine-tunes saw
the old held-out fold, so those weights cannot use it as independent validation.
Source normalization drift, scene distribution and class changes are concerns;
CSV checksums/order/data fingerprints were checked and did not explain regression.

## Current implementation

Read experiments/efficient_search.md, cvlab/efficient_search.py,
cvlab/sequential_policy.py and scripts/start_efficient_search.py.

The last implementation commit before this handoff is 5e67c561f2bed757e8a088458b15dd93b4497135.
It implements:
- Original learned weights plus optimizer moments, AMP and RNG, no fresh model.
- Three local learning rates around 1e-5, log10 radius 0.5.
- Successive-halving cumulative budgets 6000 / 12000 / 24000 frames.
- Batch size one; at most three / two / one trials at the respective stages.
- A fixed up-to-512-frame probe, excluding training flight families.
- Paired approximate 90% family-bootstrap intervals as allocation heuristics.
- Optimistic AP improvement per estimated additional wall time for allocation.
- Full native held-out-fold AND fixed full official tiled checks before promotion.
- Immutable rung checkpoints, resumable state, cached full confirmations.
- Rebase subsequent comparisons against a newly accepted local champion.
- Original public-best run retained; local acceptance does not establish public gain.

First inspect whether the source fold actually leaves at least four independent
flight families unseen; the runner rejects fewer. Do not silently remove this
requirement or claim independent validation on related/trained clips. If it
cannot run, diagnose and choose a defensible validation design before spending
GPU time. This has been tested on CPU, not on the user's actual dataset/GPU.

Run (after mount/access/CUDA checks and confirming no duplicate job):
```bash
WORK="${CVLAB_LOCAL_WORK:?Set this to the configured local work directory}"
TORCH_HOME="$WORK/cache/torch" HF_HOME="$WORK/cache/huggingface" \
.venv/bin/python -u scripts/start_efficient_search.py \
  --work "$WORK" --hours 4 --reserve-hours 1 --cycles 1 --export-tiles
```
The launcher owns the shared GPU lease and waits for existing jobs.
Use nohup plus systemd-inhibit for a detached run that blocks automatic sleep.
Four hours is a per-invocation soft wall allowance after the lease is acquired;
the reserve is for confirmation/export. Active validation/export may overrun.
State: `WORK/efficient_search/state.json`.
Suggested log: `WORK/logs/efficient_search.log`.
Stop marker: `WORK/runs/rtdetr_efficient_search/STOP` (checkpoint-boundary stop).
`--reset-to-original` restores known public-best weights and preserves evidence.
Earlier adaptive/iterative controllers have their own STOP markers in their
respective run directories. Inspect them rather than launching duplicates.

## Local supervision requested

Inspect processes, parent/worker relationships, GPU load/memory, available RAM,
swap, disk space, new log lines and state JSON directly. Multiple DataLoader
workers with the same argv do not imply several independent training runs.
GPU idle during CPU COCO evaluation is not automatically a hung job.

Previously:
- Kernel RAM OOM killed training during COCO evaluation twice; metrics batching
  and streamed JSON reduced memory use. Avoid retaining duplicate full manifests
  or spawning several large evaluators.
- A CUDA launch failure left CUDA unavailable until reboot. Do not repeatedly
  restart a broken CUDA context.
- A short CPU job can overlap GPU work if RAM permits; one 8 GB GPU does not
  automatically benefit from simultaneous full detector training processes.

For continuous monitoring, use a small local background watchdog/service for
periodic status and alerts; an active Codex CLI session can inspect and respond.
Do not represent an interactive agent as an unlimited always-on daemon. Preserve
the training process when the UI/agent session disconnects. A watchdog should
record compact rotating/bounded status, exit codes and checkpoints, avoid calling
an LLM each minute, and avoid restart loops or killing healthy long phases.
Expose concise progress, accepted/rejected candidates, and paths to ready CSVs.
Retain the GPU lease and checkpoint-before-restart behavior. Never overwrite the
public-best source or restart from random weights.

The user authorizes routine local inspection, reversible code fixes, meaningful
tests and supervised experiment continuation. Do not terminate an active run
just to deploy a different policy. Never print or commit Kaggle/OpenAI tokens.
Do not auto-submit to Kaggle or publish scores that have not been observed.
The academic report must be written by the student; code/technical documentation
help is allowed. Reply in Ukrainian and prioritize concise, actionable updates.

## Verified local follow-up — 2026-11-11

Volume /dev/nvme0n1p5 is now mounted read/write via fuseblk at the expected path; the earlier ntfs3 dirty-volume error remains in this boot journal. No filesystem repair or remount was performed in this follow-up. Active training has successfully written checkpoints.

Existing train-efficient PID 11151 under systemd-inhibit was preserved, with its shared GPU lease and detached log. PyTorch 2.6.0+cu118 successfully executed a CUDA tensor calculation. Trials 0 and 1 completed 6000 frames each, with immutable rung_1 checkpoints; global probe gains +0.01589185 and +0.01311184 across 16 unseen families. Trial 2 is progressing. Full confirmations have not yet established a new champion. Original champion remains a7cd89fb349d.

Installed persistent user timer cvlab-watchdog.timer (2-minute interval); script scripts/training_watchdog.py. Status, rotating log and progress.csv are in local_supervision/ (gitignored). Desktop notifications report changes and alerts. Timer is enabled and service exit status verified 0. No automatic restart/kill is enabled: inspect failures before resuming with the documented launcher, which reuses completed observations and checkpoints. User service runs independently of the Codex turn, within the user systemd session. Pre-existing training exit code is unavailable to watchdog.

Ready historical public-best fallback CSV: `$CVLAB_LOCAL_WORK/submissions/rtdetr_a7cd89fb349d_960_tiles_b57dbbdfb3d4.csv` — verified 3811 rows and 3811 unique image IDs. No Kaggle submission performed.

## Anti-overfitting protocol update — 2026-11-11

New code (not retroactively applied to PID 11151) rejects flight-family leakage including inherited weights, groups all flight excerpts into one fold, and filters full native confirmations to unseen families. Confirmation protocol unseen-flight-families-v2 invalidates old decisions for new confirmation gates while preserving all observations/checkpoints. Legacy folds.json is unchanged. Separate prepared/folds_family_v1.json has 3 partitions (8079/8029/8093 frames, 16 families each). New data.folds_file selects versioned folds; configs/rtdetr_family_generalization.yaml enables family/time sampling and early stopping after 3 validation checks without >0.001 AP improvement. This new CV configuration has not been trained; existing fine-tuned weights must not be reused across changed holdouts. Full clean native holdout from legacy fold has 7329 frames; 751 related-flight frames are excluded.

Sampling-only ablation is queued as user service cvlab-sampling-ablation.service behind shared GPU lease; script imports updated safety checks after acquiring the lease. Source weights/optimizer/RNG preserved; one 6000-frame/1800-second soft block only; no automatic promotion/export. See docs/validation_quality.md. Local CSVs: family_folds.csv and quality_metrics.csv (probe only), both under local_supervision/. Original assignment still unavailable.

## Kaggle verification — 2026-11-11

Fetched official public competition metadata via competitions.CompetitionService/GetCompetition and published pages via competitions.PageService/ListPages. Cached in local_supervision/kaggle_competition_metadata.json and kaggle_ListPages.json. The original assignment is now available in the Description page. Deadline 2026-11-15T21:55:00.767Z (23:55 Chisinau); maxDailySubmissions=10; leaderboardPercentage=30; numScoredSubmissions=2. Evaluation algorithm is a custom metric (metric_template); exact Evaluation page returned 403, not verified against local COCO implementation. Private submissions API returned 401; no local Kaggle credentials found.

Read-only score polling prepared as scripts/check_kaggle_scores.py and cvlab-kaggle-scores.timer every 10 minutes. Without credentials it records blocked_missing_local_kaggle_authentication without network requests or interactive prompts. Uses official Kaggle CLI; never submits or selects final submissions. Status at local_supervision/kaggle_scores_status.json and scored history at kaggle_scores.csv once authenticated. No new Kaggle submission performed.

## Larger training budget requested — 2026-11-11

User raised desired Kaggle target to 0.28+ and requested more training exposure. Updated the previously waiting sampling-only service to a 72000-frame continuation: scripts/start_sampling_experiment.py --frames 72000 --hours 8 --reserve-hours 2. Only the verified lease-waiting job was replaced; live PID 11151/export was preserved. Service is now defined persistently at ~/.config/systemd/user/cvlab-sampling-ablation.service, with KillMode=process to preserve a detached training child when the wrapper is stopped. It waits for the shared GPU lease.

New controller cvlab/generalization_training.py: first 6000-frame sampling ablation, then full randomized 16121-frame passes (last partial), up to 72000 presentations. Same source learned weights, optimizer/RNG and independent 512-frame probe; no held-out frames added to train. Three stale validation checks (>0.001 AP delta required), persistent wall deadline (6 hours training + 2 hours confirmation/export, soft phases), STOP marker, immutable per-block rungs and interrupted-receipt recovery. Best rung receives strict unseen-family native + official tiled confirmation before any promotion/export. It can update the shared local champion with an external_promotions receipt under the held GPU lease. No score of 0.28 has been observed or promised.

Reviewable next plan: WORK/efficient_search/next_training_plan.json. Runtime state after admission: WORK/runs/rtdetr_sampling_flight_time/continuation.json. Tests cover exact budget sequence, repeated invocation, early stopping, keeping original weights, promotion/rejection, and exit after committed metrics without retraining.

The live legacy race exported rtdetr_51bc69d304b8_efficient_tiles.csv and accepted checkpoint 51bc69d304b8 at 2026-11-11 13:08 local time, then continued stage 2 without interruption. Official tiled AP 0.1919728303 vs original 0.1854556759. Native legacy-fold AP 0.2252505649 vs 0.2081339265 remains auxiliary because that fold includes related flights. Full CSV validation passed: 3811 IDs in correct order, 1853253 detections, SHA256 82ef4aaed38ec93fb23d90c277a8a076d60e5a61755c7a14fa20fb547728db61. No Kaggle score observed for it yet.

Watchdog now recognizes start_sampling_experiment.py and its loader descendants, and includes external_promotions CSV paths in notifications. Restarted the supervisor only; live training PID 11151 and queued PID 82978 were preserved. New controller tests include a real tiny CPU-model uninterrupted/resumed comparison with identical weights and loader RNG, plus legacy-block adoption and interrupted receipt recovery.
