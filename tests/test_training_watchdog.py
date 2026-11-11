import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from cvlab import training_watchdog as wd


@pytest.fixture
def config(tmp_path):
    project, work, state = (
        tmp_path / name for name in ("project", "volume/work", "state")
    )
    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").symlink_to(sys.executable)
    (work / "runs/rtdetr_fold0_local").mkdir(parents=True)
    (work / "runs/rtdetr_fold0_local/best.pt").write_bytes(b"public-best")
    state.mkdir()
    return dict(
        project=str(project),
        work=str(work),
        state_dir=str(state),
        mount=None,
        device=None,
        gpu=0,
        total_hours=8,
        session_hours=2,
        reserve_hours=0.5,
        max_cycles=3,
        max_retries=2,
        interval=60,
        min_ram_mib=4096,
        min_disk_gib=8,
        stall_minutes=30,
        inhibit=False,
    )


@pytest.fixture
def sample():
    return dict(
        mounted=True,
        writable=True,
        work_free_gib=50,
        state_free_mib=1000,
        available_mib=8000,
        roots=[],
        stop=False,
        cpu_ticks=0,
        io_bytes=0,
        boot_id="boot-a",
        controller={"complete_cycles": 0, "observations": 0},
        gpu={"ready": True, "compute_pids": [], "utilization": 0},
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"mounted": False}, "waiting_mount"),
        ({"writable": False}, "waiting_mount"),
        ({"stop": True}, "paused"),
        ({"work_free_gib": 0.1}, "waiting_disk"),
        ({"state_free_mib": 200}, "waiting_disk"),
        ({"available_mib": 1000}, "waiting_ram"),
        ({"gpu": {"ready": False, "compute_pids": []}}, "waiting_gpu"),
        ({"gpu": {"ready": True, "compute_pids": [123]}}, "waiting_gpu"),
        ({"roots": [{"pid": 4}]}, "monitor"),
        ({"job_alive": True}, "monitor"),
    ],
)
def test_admission_holds_without_killing_healthy_work(config, sample, change, expected):
    sample.update(change)
    assert wd.decide(config, {}, sample, 100)[0] == expected


def test_budget_is_persistent_and_live_validation_can_finish(config, sample):
    state = {}
    first = wd.make_job(config, state, sample, 100)
    deadline = state["deadline"]
    second = wd.make_job(config, state, sample, deadline - 1800)
    assert first["hours"] == 2 and second["hours"] == 0.5
    assert second["reserve_hours"] < second["hours"]
    assert state["deadline"] == deadline
    assert wd.decide(config, state, sample, deadline)[0] == "finished"
    sample["job_alive"] = True
    assert wd.decide(config, state, sample, deadline)[0] == "monitor"


def test_cycle_limit_counts_new_cycles_only(config, sample):
    state = {"baseline_cycles": 4}
    sample["controller"]["complete_cycles"] = 6
    assert wd.decide(config, state, sample, 100)[0] == "start"
    sample["controller"]["complete_cycles"] = 7
    assert wd.decide(config, state, sample, 100)[0] == "finished"


def test_bounded_retries_and_cuda_recovery_only_after_reboot(config, sample):
    state = {}
    job = {"returncode": 137, "reason": "killed_or_oom"}
    for index in range(3):
        wd.consume_job(state, job, sample, config, 1000 + index)
    assert state["failures"] == 3 and state["blocked"]["kind"] == "retry_limit"
    assert wd.decide(config, state, sample, 100000)[0] == "blocked"
    state = {"blocked": {"kind": "cuda_fault", "boot_id": "boot-a"}}
    assert wd.decide(config, state, sample, 100)[0] == "blocked"
    sample["boot_id"] = "boot-b"
    assert wd.decide(config, state, sample, 100)[0] == "start"


def test_filesystem_recovery_is_automatic_but_io_errors_stay_latched(config, sample):
    state = {"blocked": {"kind": "filesystem_fault", "boot_id": "boot-a"}}
    sample["writable"] = False
    assert wd.decide(config, state, sample, 100)[0] == "waiting_mount"
    sample["writable"] = True
    assert wd.decide(config, state, sample, 100)[0] == "start"
    state["blocked"]["kind"] = "storage_io_fault"
    assert wd.decide(config, state, sample, 100)[0] == "blocked"


@pytest.mark.parametrize(
    "text,code,kind",
    [
        ("RuntimeError: CUDA error: unspecified launch failure", 1, "cuda_fault"),
        ("AssertionError: CUDA unavailable", 1, "cuda_fault"),
        (
            "ValueError: Need at least four unseen video families",
            1,
            "configuration_fault",
        ),
        ("OSError: Read-only file system", 1, "filesystem_fault"),
        ("OSError: Input/output error", 1, "storage_io_fault"),
        ("RuntimeError: CUDA out of memory", 1, "gpu_memory"),
        ("killed", 137, "killed_or_oom"),
        ("temporary failure", 1, "transient_exit"),
    ],
)
def test_failure_classification(text, code, kind):
    assert wd.failure_kind(text, code) == kind


def test_success_without_state_progress_does_not_loop(config, sample):
    state = {}
    wd.consume_job(
        state, {"returncode": 0, "before": sample["controller"]}, sample, config, 100
    )
    assert state["blocked"]["kind"] == "no_progress"


def fake_process(proc, pid, ppid, argv, cwd):
    folder = proc / str(pid)
    folder.mkdir()
    fields = ["0"] * 22
    fields[0], fields[1], fields[11], fields[12], fields[19], fields[21] = (
        "R",
        str(ppid),
        "15",
        "5",
        str(pid * 10),
        "1024",
    )
    (folder / "stat").write_text(f"{pid} (loader worker) " + " ".join(fields))
    (folder / "io").write_text("rchar: 100\nwchar: 200\nread_bytes: 0\n")
    (folder / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    (folder / "cwd").symlink_to(cwd)


def test_process_matching_collapses_workers_and_ignores_wrappers_and_strings(tmp_path):
    proc, project = tmp_path / "proc", tmp_path / "project"
    proc.mkdir()
    project.mkdir()
    work = tmp_path / "work"
    argv = ["python", "-u", "-m", "cvlab", "train-efficient", "--work", str(work)]
    fake_process(proc, 101, 1, argv, project)
    fake_process(proc, 102, 101, argv, project)
    fake_process(proc, 103, 101, argv, project)
    fake_process(proc, 104, 1, ["systemd-inhibit", *argv], project)
    fake_process(
        proc, 105, 1, ["python", "-c", "print('python -m cvlab train')"], project
    )
    fake_process(
        proc,
        106,
        1,
        ["python", "-m", "cvlab.training_watchdog", "--worker", "job.json"],
        project,
    )
    roots, ticks, io = wd.cvlab_processes(project, work, proc)
    assert [p["pid"] for p in roots] == [101]
    assert ticks == 60 and io == 900


def test_missing_mount_does_not_create_fake_work_tree(config, sample, monkeypatch):
    work = Path(config["work"])
    shutil = __import__("shutil")
    shutil.rmtree(work)
    config["device"] = "/dev/test-volume"
    sample.update(mounted=False, writable=False)
    monkeypatch.setattr(wd, "snapshot", lambda *a: sample)
    calls = []
    monkeypatch.setattr(
        wd,
        "command",
        lambda argv, **kw: (calls.append(argv) or (1, "", "volume is dirty")),
    )
    watchdog = wd.Watchdog(config)
    monkeypatch.setattr(watchdog, "event", lambda *a: None)
    watchdog.tick()
    watchdog.tick()
    assert not work.exists()
    assert len(calls) == 1 and "--filesystem-type" in calls[0]
    assert not any("force" in arg or "sudo" in arg for arg in calls[0])
    assert (
        wd.read_json(Path(config["state_dir"]) / "status.json")["action"]
        == "waiting_mount"
    )
    watchdog.handler.close()


def test_daemon_restart_adopts_live_worker_and_consumes_receipt_once(
    config, sample, monkeypatch
):
    clock = [10000]
    monkeypatch.setattr(wd.time, "time", lambda: clock[0])

    def snapshot(cfg, state):
        result = copy.deepcopy(sample)
        if state.get("active_job"):
            result["job"] = wd.read_json(state["active_job"])
            result["job_alive"] = result["job"]["status"] == "running"
        return result

    monkeypatch.setattr(wd, "snapshot", snapshot)
    calls = []

    class Process:
        def poll(self):
            return None

    def spawn(argv, **kwargs):
        calls.append(argv)
        job = wd.read_json(argv[-1])
        job["status"] = "running"
        wd.write_json(argv[-1], job)
        return Process()

    monkeypatch.setattr(wd.subprocess, "Popen", spawn)
    first = wd.Watchdog(config)
    monkeypatch.setattr(first, "event", lambda *a: None)
    first.tick()
    deadline = first.state["deadline"]
    receipt = Path(first.state["active_job"])
    first.handler.close()
    second = wd.Watchdog(config)
    monkeypatch.setattr(second, "event", lambda *a: None)
    second.tick()
    assert len(calls) == 1
    job = wd.read_json(receipt)
    job.update(status="finished", returncode=137, reason="killed_or_oom")
    wd.write_json(receipt, job)
    second.tick()
    second.tick()
    assert second.state["failures"] == 1 and len(calls) == 1
    assert second.state["deadline"] == deadline
    second.handler.close()


def test_real_worker_writes_receipt_strips_overrides_and_never_repeats_job(
    config, sample, monkeypatch
):
    project = Path(config["project"])
    (project / "scripts").mkdir()
    (project / "scripts/start_efficient_search.py").write_text(
        "import os,sys\nprint('epoch=0 training; up to 6000 batches', flush=True)\n"
        "assert 'PYTHONPATH' not in os.environ and 'CVLAB_DATA_ROOT' not in os.environ\n"
        "assert '--export-tiles' in sys.argv\nsys.exit(7)\n"
    )
    monkeypatch.setenv("PYTHONPATH", "unrelated-path")
    monkeypatch.setenv("CVLAB_DATA_ROOT", "unrelated-data")
    old_log = Path(config["work"]) / "logs/watchdog_training.log"
    old_log.parent.mkdir()
    old_log.write_text("An older job: CUDA error: unspecified launch failure\n")
    # Only the CUDA preflight is simulated; the training wrapper is a real child process.
    monkeypatch.setattr(
        wd.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "", ""),
    )
    job = wd.make_job(config, {}, sample, 100)
    path = Path(config["state_dir"]) / "job.json"
    wd.write_json(path, job)
    wd.run_worker(path)
    result = wd.read_json(path)
    assert result["status"] == "finished" and result["returncode"] == 7
    assert result["reason"] == "transient_exit"
    log = Path(config["work"]) / "logs/watchdog_training.log"
    assert "epoch=0 training" in log.read_text()
    before = log.read_bytes()
    wd.run_worker(path)
    assert log.read_bytes() == before


def test_actual_parent_exit_and_relaunch_preserve_detached_worker(config, tmp_path):
    """Real process lifecycle, simulated CUDA only; no service/GPU is activated."""
    project, work = Path(config["project"]), Path(config["work"])
    (project / "cvlab").symlink_to(
        Path(__file__).parents[1] / "cvlab", target_is_directory=True
    )
    python = project / ".venv/bin/python"
    python.unlink()
    python.write_text(
        '#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec '
        + str(sys.executable)
        + ' "$@"\n'
    )
    python.chmod(0o755)
    (project / "scripts").mkdir()
    (project / "scripts/start_efficient_search.py").write_text(
        "import json,time\nfrom pathlib import Path\ntime.sleep(0.8)\n"
        f"p=Path({str(work / 'efficient_search/state.json')!r})\np.parent.mkdir(exist_ok=True)\n"
        "p.write_text(json.dumps({'cycles':[{'complete':True,'observations':[{}]}]}))\n"
        "print('epoch=0 training complete; checkpoint saved', flush=True)\n"
    )
    config.update(min_disk_gib=0.01, min_ram_mib=1)
    path = tmp_path / "config.json"
    wd.write_json(path, config)
    parent_code = (
        "from cvlab import training_watchdog as w; "
        "w.gpu_info=lambda i: {'ready':True,'compute_pids':[],'utilization':0}; w.main()"
    )

    def run_parent():
        result = subprocess.run(
            [sys.executable, "-c", parent_code, "--config", str(path), "--once"],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr

    run_parent()
    jobs = list((Path(config["state_dir"]) / "jobs").glob("*.json"))
    assert len(jobs) == 1
    receipt = jobs[0]
    deadline = time.monotonic() + 10
    while wd.read_json(receipt)["status"] == "queued" and time.monotonic() < deadline:
        time.sleep(0.01)
    job = wd.read_json(receipt)
    assert job["status"] == "running" and wd.alive(job["pid"], job["start"])
    assert os.getpgid(job["pid"]) != os.getpgrp()
    run_parent()
    assert len(list(receipt.parent.glob("*.json"))) == 1
    while wd.read_json(receipt)["status"] != "finished" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert wd.read_json(receipt)["returncode"] == 0
    run_parent()
    assert len(list(receipt.parent.glob("*.json"))) == 1
    assert wd.read_json(Path(config["state_dir"]) / "state.json")["active_job"] is None
    assert (work / "runs/rtdetr_fold0_local/best.pt").read_bytes() == b"public-best"


def test_sensor_error_is_visible_in_status_and_does_not_launch(config, monkeypatch):
    def fail(*a):
        raise ValueError("corrupted controller metadata")

    monkeypatch.setattr(wd, "snapshot", fail)
    watchdog = wd.Watchdog(config)
    monkeypatch.setattr(watchdog, "event", lambda *a: None)
    watchdog.tick()
    assert (
        wd.read_json(Path(config["state_dir"]) / "status.json")["action"]
        == "sensor_error"
    )
    assert not watchdog.state["active_job"]
    watchdog.handler.close()


def test_worker_setup_failure_also_produces_a_durable_receipt(config, sample):
    (Path(config["work"]) / "logs").write_text("a file blocks the log directory")
    path = Path(config["state_dir"]) / "job.json"
    wd.write_json(path, wd.make_job(config, {}, sample, 100))
    wd.worker_entry(path)
    job = wd.read_json(path)
    assert job["status"] == "finished" and job["reason"] == "configuration_fault"
    assert "File exists" in job["error"]


def test_pause_and_resume_preserve_budget_and_checkpoints(config, monkeypatch):
    directory = Path(config["state_dir"])
    path = directory / "config.json"
    wd.write_json(path, config)
    wd.write_json(directory / "state.json", {"failures": 1, "deadline": 12345})
    original = (directory / "state.json").read_bytes()
    for mode in ("--pause", "--resume"):
        monkeypatch.setattr(sys, "argv", ["watchdog", "--config", str(path), mode])
        wd.main()
        expected = mode == "--pause"
        assert (directory / "PAUSE").exists() == expected
        assert (
            Path(config["work"]) / "runs/rtdetr_efficient_search/STOP"
        ).exists() == expected
        assert (directory / "state.json").read_bytes() == original
    assert (
        Path(config["work"]) / "runs/rtdetr_fold0_local/best.pt"
    ).read_bytes() == b"public-best"


def test_configuration_rejects_unbounded_budgets_and_state_on_data_volume(config):
    wd.validate_config(config)
    broken = {**config, "state_dir": str(Path(config["work"]) / "state")}
    with pytest.raises(ValueError, match="outside"):
        wd.validate_config(broken)
    for key, value in [
        ("total_hours", float("inf")),
        ("interval", 0),
        ("max_retries", -1),
    ]:
        with pytest.raises(ValueError):
            wd.validate_config({**config, key: value})


def installer_module():
    path = Path(__file__).parents[1] / "scripts/install_training_watchdog.py"
    spec = importlib.util.spec_from_file_location("watchdog_installer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_writes_user_unit_and_preserves_live_plan(tmp_path, monkeypatch):
    module = installer_module()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        sys, "argv", ["install", "--work", str(tmp_path / "volume/work"), "--no-start"]
    )
    module.main()
    config_path = tmp_path / "config/cvlab-watchdog/config.json"
    unit = (tmp_path / "config/systemd/user/cvlab-watchdog.service").read_text()
    assert "KillMode=process" in unit and "Restart=on-failure" in unit
    assert "sudo" not in unit and "API_KEY" not in unit
    assert wd.read_json(config_path)["total_hours"] == 8
    before = config_path.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["install", "--work", str(tmp_path / "different"), "--no-start"]
    )
    with pytest.raises(SystemExit):
        module.main()
    assert config_path.read_bytes() == before


def test_unit_quotes_paths_and_cannot_inject_directives():
    module = installer_module()
    unit = module.render_unit(
        '/tmp/project %x "quoted"', "/usr/bin/python3", "/tmp/config $var.json"
    )
    assert 'WorkingDirectory=/tmp/project %%x "quoted"' in unit and "$$var" in unit
    with pytest.raises(ValueError):
        module.unit_quote("/tmp/project\nExecStart=bad")


def test_failed_rearm_still_restarts_supervisor(tmp_path, monkeypatch):
    module = installer_module()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(sys, "argv", ["install", "--rearm"])
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        if "--rearm" in argv:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert calls[0][:3] == ["systemctl", "--user", "stop"]
    assert calls[-1][:3] == ["systemctl", "--user", "start"]
