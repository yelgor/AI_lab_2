import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvlab import kaggle_data


def write_archive(path, prefix=""):
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            "test_images.csv",
            "sample_submission.csv",
            "train/images/a/1.jpg",
            "train/labels/a/1.txt",
            "val/images/b/1.jpg",
            "val/labels/b/1.txt",
            "test/images/c/1.jpg",
        ):
            archive.writestr(prefix + name, "test")


def test_download_extract_and_reuse_without_credentials(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(kaggle_data.shutil, "which", lambda _: "/test/kaggle")

    def run(arguments, check):
        calls.append(arguments)
        assert check
        assert arguments[:4] == [
            "/test/kaggle",
            "competitions",
            "download",
            kaggle_data.COMPETITION,
        ]
        write_archive(tmp_path / f"{kaggle_data.COMPETITION}.zip", "nested/")

    monkeypatch.setattr(kaggle_data.subprocess, "run", run)
    root = kaggle_data.download_data(tmp_path)
    assert root == tmp_path / "nested"
    assert not list(tmp_path.glob("*.zip"))
    assert kaggle_data.download_data(tmp_path) == root
    assert len(calls) == 1
    marker = json.loads((tmp_path / kaggle_data.MARKER).read_text())
    assert marker["competition"] == kaggle_data.COMPETITION


def test_existing_archive_does_not_need_cli(tmp_path, monkeypatch):
    path = tmp_path / f"{kaggle_data.COMPETITION}.zip"
    write_archive(path)
    monkeypatch.setattr(kaggle_data.shutil, "which", lambda _: None)
    assert kaggle_data.download_data(tmp_path, keep_archive=True) == tmp_path
    assert path.is_file()


def test_failed_download_does_not_mark_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_data.shutil, "which", lambda _: "/test/kaggle")

    def fail(arguments, check):
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(kaggle_data.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="401.*403"):
        kaggle_data.download_data(tmp_path)
    assert not (tmp_path / kaggle_data.MARKER).exists()


def test_archive_paths_validated_before_extraction(tmp_path):
    path = tmp_path / f"{kaggle_data.COMPETITION}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("good.txt", "test")
        archive.writestr("../outside.txt", "test")
    with pytest.raises(ValueError, match="escapes"):
        kaggle_data.download_data(tmp_path)
    assert not (tmp_path / "good.txt").exists()
    assert not (tmp_path.parent / "outside.txt").exists()


def test_checks_extraction_space(tmp_path, monkeypatch):
    write_archive(tmp_path / f"{kaggle_data.COMPETITION}.zip")
    monkeypatch.setattr(
        kaggle_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    with pytest.raises(RuntimeError, match="disk space"):
        kaggle_data.download_data(tmp_path)
    assert not (tmp_path / kaggle_data.MARKER).exists()


def test_other_competition_cache_is_rejected(tmp_path):
    (tmp_path / kaggle_data.MARKER).write_text(
        json.dumps({"competition": "another-course"})
    )
    with pytest.raises(ValueError, match="different competition"):
        kaggle_data.download_data(tmp_path)


@pytest.mark.parametrize("cached", [False, True])
def test_notebook_hidden_token_or_cached_data(tmp_path, monkeypatch, capsys, cached):
    import getpass

    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/quickstart.ipynb").read_text()
    )
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["id"] == "cvlab-kaggle-data-download"
    )
    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda p: False if str(p) == "/kaggle" else real_exists(p)
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "KAGGLE_CONFIG_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    # Register this key with monkeypatch before the cell changes it, so it is restored.
    monkeypatch.setenv("KAGGLE_API_TOKEN", "")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.setattr(
        kaggle_data, "ready_data_root", lambda _: tmp_path if cached else None
    )
    prompts, calls = [], []

    def hidden_input(prompt):
        prompts.append(prompt)
        return "offline-test-token"

    monkeypatch.setattr(getpass, "getpass", hidden_input)
    monkeypatch.setattr(
        subprocess, "run", lambda arguments, check: calls.append(arguments)
    )
    exec(compile(source, "notebook-auth-cell", "exec"), {})
    if cached:
        assert not prompts and not calls
    else:
        assert len(prompts) == len(calls) == 1
        assert os.environ["KAGGLE_API_TOKEN"] == "offline-test-token"
        assert "offline-test-token" not in " ".join(calls[0])
    assert "offline-test-token" not in capsys.readouterr().out
