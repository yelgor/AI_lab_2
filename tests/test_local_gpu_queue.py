import copy
import fcntl
import importlib.util
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def queue():
    path = Path(__file__).parents[1] / "scripts/local_gpu_queue.py"
    spec = importlib.util.spec_from_file_location("local_gpu_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def work(tmp_path):
    project = Path(__file__).parents[1]
    reference = yaml.safe_load((project / "configs/retinanet.yaml").read_text())
    reference["loader"].update(batch_size=1, workers=2, prefetch_factor=1)
    reference["train"]["accumulation"] = 4
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/retinanet_fold0_local.yaml").write_text(
        yaml.safe_dump(reference)
    )
    return tmp_path


def test_queue_order_resume_and_reference_preservation(queue, work, monkeypatch):
    original = (work / "configs/retinanet_fold0_local.yaml").read_text()
    reference = yaml.safe_load(original)
    calls, waits = [], []
    monkeypatch.setattr(queue, "wait_for_gpu", lambda pid=None: waits.append(pid))

    def fake_run(project, arguments):
        calls.append(arguments)
        if arguments[0] == "train":
            config = yaml.safe_load(Path(arguments[2]).read_text())
            output = Path(config["output"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "last.pt").touch()
            (output / "val_predictions.json").write_text("{}")
            assert config["loader"] == reference["loader"]
            assert config["data"] == reference["data"]

    monkeypatch.setattr(queue, "run_cli", fake_run)
    queue.main(
        ["--work", str(work), "--wait-pid", "123", "--models", "fcos", "fasterrcnn"]
    )
    assert [
        yaml.safe_load(Path(call[2]).read_text())["model"]["name"] for call in calls
    ] == ["fcos", "fasterrcnn"]
    assert waits == [123, None, None]
    queue.main(["--work", str(work), "--models", "fcos"])
    assert calls[-1][-2:] == ["--resume", work / "runs/fcos_fold0_local/last.pt"]
    assert (work / "configs/retinanet_fold0_local.yaml").read_text() == original


def test_existing_custom_configuration_is_not_overwritten(queue, work):
    project = Path(__file__).parents[1]
    reference = yaml.safe_load(
        (work / "configs/retinanet_fold0_local.yaml").read_text()
    )
    path, _ = queue.model_config(project, work, reference, "fcos")
    custom = copy.deepcopy(yaml.safe_load(path.read_text()))
    custom["model"]["image_size"] = 640
    path.write_text(yaml.safe_dump(custom))
    with pytest.raises(ValueError, match="preserved"):
        queue.model_config(project, work, reference, "fcos")
    assert yaml.safe_load(path.read_text())["model"]["image_size"] == 640


def test_duplicate_queue_does_not_start_jobs(queue, work, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("A duplicate queue tried to start work")

    monkeypatch.setattr(queue, "wait_for_gpu", unexpected)
    with (work / ".local_gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.main(["--work", str(work)])
