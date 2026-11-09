import importlib.util
from pathlib import Path

import yaml


def launcher():
    path = Path(__file__).parents[1] / "scripts/start_overlap_queue.py"
    spec = importlib.util.spec_from_file_location("overlap_start", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_process(root, pid, parent, args):
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(("\0".join(args) + "\0").encode())
    (directory / "status").write_text(f"Name:\tpython\nPPid:\t{parent}\n")


def test_handoff_identifies_only_parent_trainer_for_the_selected_run(tmp_path):
    module = launcher()
    config = tmp_path / "config.yaml"
    run = tmp_path / "run"
    config.write_text(yaml.safe_dump({"output": str(run)}))
    argv = ["/venv/bin/python", "-u", "-m", "cvlab", "train", "--config", str(config)]
    proc = tmp_path / "proc"
    fake_process(proc, 100, 1, argv)
    fake_process(proc, 101, 100, argv)
    fake_process(proc, 102, 1, ["/usr/bin/systemd-inhibit", *argv])
    assert module.training_parents(run, proc) == [100]


def test_handoff_does_not_signal_inhibitor_or_unrelated_queue(tmp_path):
    module = launcher()
    argv = ["/venv/bin/python", "scripts/local_gpu_queue.py", "--work", str(tmp_path)]
    proc = tmp_path / "proc"
    fake_process(proc, 100, 1, argv)
    fake_process(proc, 101, 1, ["/usr/bin/systemd-inhibit", *argv])
    fake_process(proc, 102, 1, [*argv[:-1], str(tmp_path / "other")])
    assert module.legacy_launchers(tmp_path, proc) == [100]


def test_handoff_requires_the_completed_validation_log_message(tmp_path):
    log = tmp_path / "log"
    module = launcher()
    log.write_text("epoch=0 inference complete; computing COCO metrics\n")
    assert module.latest_metric_line(log) is None
    with log.open("a") as stream:
        stream.write(
            "epoch=0 loss=1.0 mAP=0.2 best=0.2\nepoch=1 training; up to 20 batches\n"
        )
    assert module.latest_metric_line(log) == "epoch=0 loss=1.0 mAP=0.2 best=0.2"


def test_handoff_stops_trainer_only_after_completed_validation_then_executes(
    tmp_path, monkeypatch
):
    import signal
    import sys
    from types import SimpleNamespace

    module = launcher()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/retinanet_fold0_local.yaml").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "start",
            "--work",
            str(tmp_path),
            "--take-over",
            "--handoff-run",
            str(tmp_path / "run"),
            "--handoff-log",
            str(tmp_path / "log"),
        ],
    )
    markers = iter([None, "epoch=0 loss=1 mAP=.2 best=.2"])
    launchers = iter([[100], []])
    monkeypatch.setattr(module, "latest_metric_line", lambda path: next(markers))
    monkeypatch.setattr(module, "legacy_launchers", lambda work: next(launchers))
    monkeypatch.setattr(module, "training_parents", lambda run: [200])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    signals, executions = [], []
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(module.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        module.os, "execv", lambda executable, args: executions.append(args)
    )
    module.main()
    assert signals == [(100, signal.SIGTERM), (200, signal.SIGTERM)]
    assert executions[0][2:5] == ["-m", "cvlab", "train-overlap"]
