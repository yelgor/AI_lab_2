import json
import io
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

    def run(arguments):
        calls.append(arguments)
        assert arguments[:3] == [
            "competitions",
            "download",
            kaggle_data.COMPETITION,
        ]
        write_archive(tmp_path / f"{kaggle_data.COMPETITION}.zip", "nested/")

    monkeypatch.setattr(kaggle_data, "run_kaggle_cli", run)
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
    monkeypatch.setattr(
        kaggle_data, "run_kaggle_cli", lambda _: pytest.fail("Unexpected CLI call")
    )
    assert kaggle_data.download_data(tmp_path, keep_archive=True) == tmp_path
    assert path.is_file()


def test_failed_download_does_not_mark_complete(tmp_path, monkeypatch):
    def fail(arguments):
        raise kaggle_data.KaggleCommandError(1, "HTTPError: 401 Unauthorized")

    monkeypatch.setattr(kaggle_data, "run_kaggle_cli", fail)
    with pytest.raises(RuntimeError, match="401 Unauthorized") as error:
        kaggle_data.download_data(tmp_path)
    assert "403" not in str(error.value)
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


@pytest.mark.parametrize("runtime", ["colab", "kaggle"])
@pytest.mark.parametrize("use_secret", [False, True])
def test_secret_stores_and_hidden_input(
    tmp_path, monkeypatch, capsys, runtime, use_secret
):
    import getpass

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "KAGGLE_CONFIG_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "")
    prompts, secret_calls = [], []

    def secret(name):
        secret_calls.append(name)
        return "offline-test-token" if use_secret else None

    monkeypatch.setitem(
        sys.modules,
        "google.colab",
        SimpleNamespace(userdata=SimpleNamespace(get=secret)),
    )
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(get_secret=secret)),
    )

    def hidden_input(prompt):
        prompts.append(prompt)
        return "offline-test-token"

    monkeypatch.setattr(getpass, "getpass", hidden_input)
    kaggle_data.configure_kaggle_credentials(runtime)
    assert secret_calls == ["KAGGLE_API_TOKEN"]
    assert len(prompts) == (0 if use_secret else 1)
    assert os.environ["KAGGLE_API_TOKEN"] == "offline-test-token"
    assert "offline-test-token" not in capsys.readouterr().out


def make_layout(root, sample=False):
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
        "test/images",
    ):
        (root / relative).mkdir(parents=True)
    (root / "test_images.csv").write_text(
        "image_id,file_name,clip,frame,width,height\n"
    )
    if sample:
        (root / "sample_submission.csv").write_text("image_id,PredictionString\n")
    return root


def test_root_discovery_without_optional_submission_template(tmp_path):
    root = make_layout(tmp_path / "competition")
    assert kaggle_data.find_data_root(tmp_path) == root


def test_partial_layout_explains_what_is_missing(tmp_path):
    root = make_layout(tmp_path)
    (root / "train/labels").rmdir()
    with pytest.raises(kaggle_data.DatasetNotFoundError, match="train/labels"):
        kaggle_data.find_data_root(root)


@pytest.mark.parametrize("runtime", ["colab", "kaggle"])
def test_missing_input_downloads_in_either_service(tmp_path, monkeypatch, runtime):
    monkeypatch.delenv("CVLAB_DATA_ROOT", raising=False)
    monkeypatch.setattr(kaggle_data, "detect_runtime", lambda: runtime)
    authentication, downloads = [], []
    monkeypatch.setattr(
        kaggle_data, "configure_kaggle_credentials", lambda r: authentication.append(r)
    )
    monkeypatch.setattr(
        kaggle_data,
        "download_data",
        lambda p: downloads.append(p) or tmp_path / "ready",
    )
    root = kaggle_data.ensure_data_root(
        download_dir=tmp_path / "cache", input_dir=tmp_path / "empty-input"
    )
    assert root == tmp_path / "ready"
    assert authentication == [runtime]
    assert downloads == [tmp_path / "cache"]


def test_attached_dataset_never_requests_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("CVLAB_DATA_ROOT", raising=False)
    root = make_layout(tmp_path / "inputs" / "competition")
    monkeypatch.setattr(
        kaggle_data,
        "configure_kaggle_credentials",
        lambda _: pytest.fail("Unexpected credential request"),
    )
    assert kaggle_data.ensure_data_root(input_dir=tmp_path / "inputs") == root


def test_cached_dataset_never_requests_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("CVLAB_DATA_ROOT", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    write_archive(cache / f"{kaggle_data.COMPETITION}.zip")
    ready = kaggle_data.download_data(cache)
    monkeypatch.setattr(
        kaggle_data,
        "configure_kaggle_credentials",
        lambda _: pytest.fail("Unexpected credential request"),
    )
    assert (
        kaggle_data.ensure_data_root(download_dir=cache, input_dir=tmp_path / "empty")
        == ready
    )


def test_multiple_mounted_datasets_need_explicit_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("CVLAB_DATA_ROOT", raising=False)
    a, b = make_layout(tmp_path / "a"), make_layout(tmp_path / "b")
    with pytest.raises(ValueError, match="Multiple datasets"):
        kaggle_data.ensure_data_root(input_dir=tmp_path)
    monkeypatch.setenv("CVLAB_DATA_ROOT", str(b))
    assert kaggle_data.ensure_data_root(input_dir=a) == b


def test_colab_signal_takes_priority_over_kaggle_folder(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test-colab-runtime")
    monkeypatch.setattr(Path, "is_dir", lambda _: True)
    assert kaggle_data.detect_runtime() == "colab"


def test_bare_kaggle_folder_is_not_a_runtime_signal(monkeypatch):
    for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION", "KAGGLE_KERNEL_RUN_TYPE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(
        sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: None)
    )
    monkeypatch.setattr(Path, "is_dir", lambda p: str(p) == "/kaggle")
    assert kaggle_data.detect_runtime() == "local"


def test_notebook_preserves_resolved_data_root(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("CVLAB_DATA_ROOT", "")
    monkeypatch.setattr(importlib, "reload", lambda module: module)
    monkeypatch.setattr(kaggle_data, "ensure_data_root", lambda: tmp_path)
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/quickstart.ipynb").read_text()
    )
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["id"] == "cvlab-kaggle-data-download"
    )
    namespace = {}
    exec(compile(source, "data-cell", "exec"), namespace)
    assert namespace["DATA_ROOT"] == tmp_path
    assert os.environ["CVLAB_DATA_ROOT"] == str(tmp_path)


def fake_cli_process(monkeypatch, output, returncode):
    calls = []

    class Process:
        def __init__(self):
            self.stdout = io.StringIO(output)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return returncode

    def popen(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"]
        return Process()

    monkeypatch.setattr(kaggle_data.subprocess, "Popen", popen)
    return calls


def test_cli_uses_current_python_and_streams_output(monkeypatch, capsys):
    calls = fake_cli_process(monkeypatch, "Kaggle CLI test-version\n", 0)
    result = kaggle_data.run_kaggle_cli(["--version"])
    assert calls == [[sys.executable, "-m", "kaggle.cli", "--version"]]
    assert "Kaggle CLI test-version" in result
    assert "Kaggle CLI test-version" in capsys.readouterr().out


@pytest.mark.parametrize(
    "failure",
    [
        "HTTPError: 401 Unauthorized",
        "HTTPError: 403 Forbidden",
        "ModuleNotFoundError: No module named 'missing-package'",
    ],
)
def test_cli_preserves_real_failure_and_redacts_tokens(monkeypatch, capsys, failure):
    token = "KGAT_offline-secret-token"
    monkeypatch.setenv("KAGGLE_API_TOKEN", token)
    fake_cli_process(
        monkeypatch,
        f"Authorization: Bearer another-secret\ntoken={token}\n{failure}\n",
        1,
    )
    with pytest.raises(kaggle_data.KaggleCommandError) as caught:
        kaggle_data.run_kaggle_cli(["competitions", "files", kaggle_data.COMPETITION])
    logged = capsys.readouterr().out + str(caught.value)
    assert failure in logged
    assert token not in logged and "another-secret" not in logged
    assert "[REDACTED]" in logged
    assert caught.value.returncode == 1


def test_access_check_downloads_only_small_csv_and_cleans_up(monkeypatch):
    authenticated, calls = [], []
    monkeypatch.setattr(
        kaggle_data, "configure_kaggle_credentials", lambda: authenticated.append(True)
    )

    def run(arguments):
        calls.append(arguments)
        destination = Path(arguments[arguments.index("--path") + 1])
        (destination / "test_images.csv").write_text("image_id,file_name\n")

    monkeypatch.setattr(kaggle_data, "run_kaggle_cli", run)
    kaggle_data.check_access()
    assert authenticated == [True]
    assert len(calls) == 1
    assert calls[0][:3] == ["competitions", "download", kaggle_data.COMPETITION]
    assert calls[0][calls[0].index("--file") + 1] == "test_images.csv"
    assert not Path(calls[0][calls[0].index("--path") + 1]).exists()
