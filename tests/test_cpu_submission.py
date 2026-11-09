import csv
from pathlib import Path

import pytest

from cvlab.common import write_json
from cvlab.submit_cpu import create_submission


def fixture_run(tmp_path):
    data, prepared, run = (tmp_path / name for name in ("data", "prepared", "run"))
    data.mkdir()
    records = [
        {
            "id": i,
            "submission_id": f"frame_{i}",
            "file_name": f"test/{i}.jpg",
            "width": 100,
            "height": 50,
        }
        for i in range(3)
    ]
    write_json(prepared / "test.json", {"images": records})
    config = {
        "data": {"root": str(data), "prepared": str(prepared)},
        "model": {
            "name": "fasterrcnn",
            "score_threshold": 0.001,
            "max_detections": 500,
        },
        "runtime": {"device": "cuda", "precision": "fp16"},
        "loader": {"batch_size": 2, "workers": 2},
        "evaluation": {"warmup_batches": 2},
    }
    write_json(run / "config.json", config)
    (run / "best.pt").write_bytes(b"frozen weights")
    (data / "test_images.csv").write_text(
        "image_id,file_name,width,height\n"
        + "".join(f"frame_{i},test/{i}.jpg,100,50\n" for i in range(3))
    )
    (data / "sample_submission.csv").write_text(
        "image_id,PredictionString\nframe_2,\nframe_0,\nframe_1,\n"
    )
    return run


def test_cpu_submission_streams_freezes_weights_and_verifies_ids(tmp_path, monkeypatch):
    from cvlab import engine

    run = fixture_run(tmp_path)
    seen = []
    monkeypatch.setattr(engine, "setup", lambda config: "cpu")

    def load(path, config, device):
        assert Path(path).read_bytes() == b"frozen weights"
        assert config["runtime"]["device"] == "cpu"
        assert config["runtime"]["precision"] == "fp32"
        assert config["loader"]["workers"] == 0
        (run / "best.pt").write_bytes(b"new training checkpoint")
        return object()

    def predict(model, config, records, device):
        assert len(records) <= 2
        seen.extend(record["id"] for record in records)
        return (
            [
                {
                    "image_id": record["id"],
                    "category_id": 8,
                    "score": 0.9,
                    "bbox": [1, 2, 3, 4],
                }
                for record in records
                if record["id"] != 1
            ],
            {},
        )

    monkeypatch.setattr(engine, "load_trained", load)
    monkeypatch.setattr(engine, "predict", predict)
    output = tmp_path / "submission.csv"
    result = create_submission(run, output, chunk_size=2)
    assert result["valid"] and result["images"] == 3 and result["empty_images"] == 1
    assert seen == [2, 0, 1]
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["PredictionString"].split()[0] == "7"
    assert rows[2]["PredictionString"] == ""
    assert not output.with_suffix(".csv.tmp").exists()
    assert output.with_suffix(".csv.metadata.json").is_file()


def test_cpu_submission_preserves_existing_file(tmp_path):
    output = tmp_path / "submission.csv"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        create_submission(tmp_path / "absent", output)
    assert output.read_text() == "existing"


def test_cpu_submission_never_publishes_empty_predictions(tmp_path, monkeypatch):
    from cvlab import engine

    run = fixture_run(tmp_path)
    monkeypatch.setattr(engine, "setup", lambda config: "cpu")
    monkeypatch.setattr(engine, "load_trained", lambda *a: object())
    monkeypatch.setattr(engine, "predict", lambda *a: ([], {}))
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match="All predictions are empty"):
        create_submission(run, output)
    assert not output.exists()


def test_cpu_submission_rejects_duplicate_writer(tmp_path):
    import fcntl

    output = tmp_path / "submission.csv"
    with output.with_suffix(".csv.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            create_submission(tmp_path / "unused", output)
