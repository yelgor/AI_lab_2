# Local automatic training supervision

This is a standard-library Linux supervisor for the existing efficient-search
controller. It uses no LLM calls or API keys. All configuration, status, hardware
observations, logs and notifications stay on the local machine. Nothing is
uploaded or submitted to Kaggle.

## Install once

From the repository, using the mounted volume and work paths appropriate to your
computer:

```bash
python3 scripts/install_training_watchdog.py \
  --work /path/to/volume/work \
  --mount /path/to/volume \
  --device /dev/your-data-partition \
  --total-hours 8 --session-hours 2 --max-cycles 3 --enable-linger
```

Omit `--device` unless ordinary automatic NTFS mounting is wanted. Omit `--mount`
for a work directory on a permanently mounted Linux filesystem. The installer
writes a **user** systemd unit and starts it; model dependencies are not changed.
Use `--no-start` to generate configuration/unit files for review only.
It can be installed while the data volume is unavailable: runtime state is kept
outside that volume, and training waits until the real volume is writable.

The service checks every 60 seconds. Admission requires writable mounted work
storage, the original checkpoint, the project Python, at least 6 GiB available
system RAM, 8 GiB free work storage and an idle GPU/lease. Thresholds are
configurable. Desktop notifications use `notify-send` when installed and a
notification session is available. Status and bounded event logs work without
desktop notifications; `libnotify-bin` provides the optional command on Ubuntu.

`--enable-linger` asks logind to keep the user manager running after logout and
at boot. If this request is denied, the installer reports it; the service can
still run while the user manager is active. Laptop shutdown/hibernation stops
compute; the worker inhibits ordinary sleep while its training process runs.

## What happens automatically

- Wait for an existing training/validation process and the shared GPU lease.
  Matching DataLoader workers are collapsed into their actual parent job.
- Attempt ordinary unprivileged UDisks NTFS mounting, at most every 15 minutes,
  if a device was supplied and the expected mount is absent. No forced recovery,
  hibernation removal or privileged unattended filesystem repair is performed.
- Run a small CUDA preflight, then launch the existing warm-start search with
  sleep inhibition. No model, loss, split, optimizer or source weights are reset
  by the watchdog. The existing controller makes experiment-selection decisions.
- Continue incomplete search sessions from persistent state, through at most
  three additional completed search cycles and the configured total budget.
- Record process/GPU/RAM/disk/log progress, expose newly verified CSV paths and
  notify about errors, memory pressure, high temperatures and suspected stalls.
- Retry transient exits with backoff (default two retries for the whole plan).
  Configuration errors, actual storage I/O errors, exhausted retries and a zero
  exit without controller-state progress latch a block. CUDA faults wait for
  recovery after reboot or explicit rearming. Restored writable storage can
  unblock an ordinary unavailable/read-only-volume failure.

The total wall deadline starts at the first owned launch and persists across
daemon restarts and reboots. Waiting after that launch counts toward the budget;
the budget is not reset every time the daemon starts. Two-hour sessions reserve
30 minutes for validation/export. The deadline is soft: active checkpoint,
validation and export phases finish instead of being killed. Further launches
stop at the total budget or completed-cycle cap.

The durable job receipt and PID/start/boot identity let a new supervisor adopt
its detached live worker. The user unit uses `KillMode=process`: stopping or
restarting the **supervisor** preserves its training worker. It never kills a
live training process based only on GPU idleness or a quiet log. CPU COCO metrics
can legitimately run without GPU activity; CPU/I/O counters also count as
progress. Suspected stalls trigger a notification rather than destructive
restarts.

No package installation, driver reset, desktop-app termination, checkpoint
overwrite, automatic Kaggle submission or unbounded retry loop is included.
Unexpected failures may need diagnosis/rearming; this supervisor is not an
unlimited coding-agent daemon. Codex CLI can read its local status and event
files when deeper investigation is needed.

## Inspect, pause and rearm

From the repository:

```bash
python3 scripts/install_training_watchdog.py --status
python3 scripts/install_training_watchdog.py --pause
python3 scripts/install_training_watchdog.py --resume
```

Pause sets a local marker plus the controller's checkpoint-boundary STOP marker
when the volume is accessible. Resume removes those markers; it does not erase
the elapsed budget or a latched fault. After fixing the cause, rearm an **idle**
plan to grant a fresh configured budget and retry allowance:

```bash
python3 scripts/install_training_watchdog.py --rearm
```

Rearming refuses a live job and brings the supervisor back even if refused.
Training checkpoints and experimental results are preserved. To disable future
supervision, use `systemctl --user disable --now cvlab-watchdog.service`; a live
worker still completes its bounded session. Use `--pause` first when no further
controller blocks should start. A deliberate OS power-off still interrupts work.

Default local files (XDG directories are honored):

- `~/.config/cvlab-watchdog/config.json`: frozen plan parameters.
- `~/.local/state/cvlab-watchdog/status.json`: latest machine-readable status.
- `~/.local/state/cvlab-watchdog/events.log`: rotating events (3 x 1 MiB).
- `~/.local/state/cvlab-watchdog/jobs/`: durable, bounded-plan job receipts.
- `WORK/logs/watchdog_training.log`: training output (3 x 5 MiB).

Using a different `--name` creates a separate service/configuration. Reinstalling
the same plan is idempotent and reloads the supervisor without killing training;
changing an already configured plan is rejected. Do not create several plans
for the same work directory: the shared training lease still serializes them.

## Validation

`tests/test_training_watchdog.py` checks resource admission, persistent time/cycle
limits, bounded retry and fault classification, absent/read-only mounts, related
process/worker detection, durable result consumption, no-progress latching,
configuration isolation and user-unit quoting. A real subprocess test terminates
the parent supervisor normally and relaunches it while the detached test worker
continues, verifying that only one job ran. CUDA is simulated in that test.
No GPU or systemd user service is activated by the test suite. Actual mount,
desktop notification, CUDA and logind behavior must be verified on the target
machine after installation.
