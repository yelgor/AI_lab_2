import copy
import os
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from cvlab.inference_experiments import balanced_subset, wait_for_idle


def test_subset_is_balanced_spread_and_reproducible():
    records = [
        {"id": i * 10 + j, "sequence_id": clip}
        for i, clip in enumerate(["a", "b", "c"])
        for j in range(10)
    ]
    selected = balanced_subset(records, 9)
    assert selected == balanced_subset(records, 9)
    assert Counter(record["sequence_id"] for record in selected) == {
        "a": 3,
        "b": 3,
        "c": 3,
    }
    assert [record["id"] for record in selected[:3]] == [0, 4, 9]
    assert balanced_subset(records, 100) is records
    with pytest.raises(ValueError):
        balanced_subset(records, 0)


def test_wait_excludes_own_pid_and_waits_for_other_jobs(monkeypatch):
    from cvlab import inference_experiments as experiments

    responses = iter([f"{os.getpid()}\n98765\n", f"{os.getpid()}\n"])
    waits = []
    monkeypatch.setattr(
        experiments.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout=next(responses), stderr=""
        ),
    )
    monkeypatch.setattr(
        experiments.time, "sleep", lambda seconds: waits.append(seconds)
    )
    monkeypatch.setattr(experiments, "_is_cvlab_job", lambda pid: True)
    wait_for_idle(0.1)
    assert waits == [0.1]


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["/venv/bin/python", "-u", "-m", "cvlab", "train-overlap"], True),
        (["/venv/bin/python3.13", "-m", "cvlab", "evaluate-resolutions"], True),
        (
            [
                "systemd-inhibit",
                "/venv/bin/python",
                "-m",
                "cvlab",
                "evaluate-resolutions",
            ],
            False,
        ),
        (["bash", "-c", "python -m cvlab evaluate-resolutions"], False),
        (
            ["python", "-c", "wrapper code", "-m", "cvlab", "evaluate-resolutions"],
            False,
        ),
    ],
)
def test_gpu_wait_matches_actual_jobs_not_wrappers(tmp_path, argv, expected):
    from cvlab.inference_experiments import _is_cvlab_job

    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    assert _is_cvlab_job(123, tmp_path) is expected


def test_gpu_wait_under_wrapper_does_not_wait_for_its_parent():
    import subprocess
    import sys
    from pathlib import Path

    child = "from cvlab.inference_experiments import wait_for_idle; wait_for_idle(.01); print('idle')"
    wrapper = "import subprocess, sys; subprocess.run([sys.executable, '-c', sys.argv[1]], check=True)"
    result = subprocess.run(
        [sys.executable, "-c", wrapper, child, "-m", "cvlab", "evaluate-resolutions"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "idle"


@pytest.mark.parametrize("name", ["retinanet", "fcos", "fasterrcnn"])
def test_cnn_checkpoint_resize_preserves_weights_and_uses_requested_size(
    name, monkeypatch
):
    from cvlab.engine import load_trained
    from cvlab.models import build_model

    torch.set_num_threads(2)
    source_config = {
        "name": name,
        "pretrained": False,
        "image_size": 128,
        "max_detections": 20,
        "score_threshold": 0.001,
    }
    original = build_model(source_config)
    checkpoint = {
        "config": {"model": source_config},
        "model": original.state_dict(),
        "adapter": None,
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    requested = copy.deepcopy(source_config)
    requested["image_size"] = 160
    restored = load_trained(
        "own-checkpoint.pt", {"model": requested}, torch.device("cpu")
    )
    assert restored.core.transform.min_size == (160,)
    assert restored.core.transform.max_size == 160
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    restored.eval()
    with torch.inference_mode():
        prediction = restored.predict([torch.rand(3, 160, 160)])[0]
    assert torch.isfinite(prediction["boxes"]).all()


def test_query_detector_resolution_override_stays_rejected(monkeypatch):
    from cvlab.engine import load_trained

    old = {"name": "rtdetr", "image_size": 128}
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: {"config": {"model": old}}
    )
    with pytest.raises(ValueError, match="differs"):
        load_trained(
            "own-checkpoint.pt",
            {"model": {**old, "image_size": 160}},
            torch.device("cpu"),
        )


def test_global_lock_prevents_multiple_waiting_experiments(tmp_path, monkeypatch):
    import fcntl
    from cvlab import inference_experiments as experiments

    monkeypatch.setattr(experiments.tempfile, "gettempdir", lambda: str(tmp_path))
    lock_path = tmp_path / f"cvlab-resolution-{os.getuid()}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already queued"):
            experiments.compare_resolutions(
                {}, "unused.pt", tmp_path / "other/result.json", wait_gpu=True
            )
