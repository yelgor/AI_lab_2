"""Download the course competition with the official Kaggle CLI.

Credentials are inherited from the environment or Kaggle's standard credential
files. This module never accepts, prints or writes credential values.
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections import deque
from pathlib import Path, PurePosixPath

COMPETITION = "multiple-object-video-detection-hw-2-ucu-ai-2026"
COMPETITION_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"
MARKER = ".cvlab-download-complete.json"


class DatasetNotFoundError(ValueError):
    """No supported dataset layout was found in the requested directory."""


def redact_cli_output(text):
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"KGAT_[A-Za-z0-9_+/=.-]+", "[REDACTED]", text)
    return re.sub(r"(?im)(authorization\s*[:=]\s*)[^\r\n]+", r"\1[REDACTED]", text)


class KaggleCommandError(RuntimeError):
    def __init__(self, returncode, details):
        self.returncode = returncode
        self.details = redact_cli_output(details)
        super().__init__(
            f"Kaggle CLI exited with code {returncode}. Actual output:\n{self.details}"
        )


def run_kaggle_cli(arguments):
    """Stream the real CLI output through the notebook's Python interpreter."""
    command = [sys.executable, "-m", "kaggle.cli", *map(str, arguments)]
    tail = deque(maxlen=25)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        for line in process.stdout:
            clean = redact_cli_output(line)
            print(clean, end="", flush=True)
            tail.append(clean.rstrip())
        returncode = process.wait()
    details = "\n".join(tail)
    if returncode:
        raise KaggleCommandError(returncode, details) from None
    return details


def check_access(competition=COMPETITION):
    """Test the actual download endpoint using only the small test CSV."""
    validate_competition(competition)
    configure_kaggle_credentials()
    print(f"Python interpreter: {sys.executable}")
    print(
        "Перевіряю завантаження лише test_images.csv, без повного dataset.", flush=True
    )
    with tempfile.TemporaryDirectory(prefix="cvlab-kaggle-check-") as temporary:
        run_kaggle_cli(
            [
                "competitions",
                "download",
                competition,
                "--file",
                "test_images.csv",
                "--path",
                temporary,
            ]
        )
        if not any(path.is_file() for path in Path(temporary).rglob("*")):
            raise RuntimeError(
                "Kaggle CLI exited successfully but did not save the test file"
            )
    print("Тестовий файл отримано. Можна запускати повне завантаження.")


def detect_runtime():
    """Prefer actual Colab signals over incidental /kaggle directories."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return "colab"
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and type(shell).__module__.startswith("google.colab"):
            return "colab"
    except ImportError:
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if Path("/content").is_dir():
        return "colab"
    return "local"


def configure_kaggle_credentials(runtime=None):
    runtime = runtime or detect_runtime()
    credential_dir = Path(
        os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    )
    if (
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or any(
            (credential_dir / name).is_file()
            for name in ("access_token", "kaggle.json")
        )
    ):
        return
    token = None
    try:
        if runtime == "colab":
            from google.colab import userdata

            token = userdata.get("KAGGLE_API_TOKEN")
        elif runtime == "kaggle":
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    except Exception:
        pass  # An unavailable optional secret store falls back to hidden input.
    if not token:
        from getpass import getpass

        token = getpass("Kaggle API token (приховане введення): ")
    if not token or not token.strip():
        raise ValueError(
            "Потрібен секрет KAGGLE_API_TOKEN або token із https://www.kaggle.com/settings/api"
        )
    os.environ["KAGGLE_API_TOKEN"] = token.strip()


def find_data_root(directory):
    """Find exactly one complete competition layout, including nested archives."""
    directory = Path(directory)
    roots = []
    candidates = list(directory.rglob("test_images.csv"))
    missing = []
    for csv in candidates:
        root = csv.parent
        folders = [
            root / split / folder
            for split in ("train", "val")
            for folder in ("images", "labels")
        ]
        folders.append(root / "test" / "images")
        absent = [str(p.relative_to(root)) for p in folders if not p.is_dir()]
        if not absent:
            roots.append(root)
        else:
            missing.append(f"{root}: missing {', '.join(absent)}")
    if len(roots) > 1:
        raise ValueError(
            "Multiple datasets found; set CVLAB_DATA_ROOT to one explicitly: "
            + ", ".join(map(str, roots))
        )
    if not roots:
        detail = "; ".join(missing[:5]) if missing else "test_images.csv not found"
        raise DatasetNotFoundError(
            f"Dataset not found in {directory}: {detail}. "
            "On Kaggle, use Add Input to attach the competition, or call ensure_data_root() to download it through the API."
        )
    return roots[0]


def ensure_data_root(download_dir=None, input_dir="/kaggle/input"):
    """Use an explicit path or mounted data; otherwise download with a secret."""
    configured = os.environ.get("CVLAB_DATA_ROOT")
    if configured:
        return find_data_root(configured)
    runtime = detect_runtime()
    print(f"Середовище: {runtime}")
    if input_dir is not None:
        try:
            root = find_data_root(input_dir)
            print(f"Використовую під'єднані дані: {root}")
            return root
        except DatasetNotFoundError as error:
            print(str(error))
    if download_dir is None:
        download_dir = {
            "kaggle": Path("/kaggle/working/drone-data"),
            "colab": Path("/content/drone-data"),
        }.get(runtime, Path.cwd() / "data" / COMPETITION)
    download_dir = Path(download_dir)
    ready = ready_data_root(download_dir)
    if ready is not None:
        print(f"Дані вже готові: {ready}")
        return ready
    try:
        return find_data_root(download_dir)
    except DatasetNotFoundError:
        pass
    configure_kaggle_credentials(runtime)
    return download_data(download_dir)


def ready_data_root(directory, competition=COMPETITION):
    directory = Path(directory)
    marker = directory / MARKER
    if not marker.is_file():
        return None
    metadata = json.loads(marker.read_text())
    if metadata.get("competition") != competition:
        raise ValueError(
            "Download directory contains a different competition; choose a separate --out"
        )
    try:
        return find_data_root(directory)
    except ValueError:
        return None


def archive_members(archive, output):
    """Validate all paths before writing any extracted file."""
    root = Path(output).resolve()
    result = []
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        target = root.joinpath(*relative.parts).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.is_relative_to(root)
        ):
            raise ValueError(
                f"Archive path escapes download directory: {member.filename}"
            )
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Archive contains a symbolic link: {member.filename}")
        result.append((member, target))
    return result


def validate_competition(competition):
    if not competition or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in competition
    ):
        raise ValueError("Use a Kaggle competition slug, not a URL")


def download_data(output, competition=COMPETITION, keep_archive=False):
    validate_competition(competition)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = ready_data_root(output, competition)
    if ready is not None:
        print(f"Dataset already prepared: {ready}")
        return ready
    archive_path = output / f"{competition}.zip"
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        print(
            f"Downloading {competition}. The archive contains approximately 11 GB of competition data.",
            flush=True,
        )
        arguments = [
            "competitions",
            "download",
            competition,
            "--path",
            str(output),
        ]
        if archive_path.exists():
            arguments.append("--force")
        # Tokens stay in the inherited environment, never in command arguments.
        run_kaggle_cli(arguments)
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        raise RuntimeError(
            "Kaggle did not produce a complete ZIP archive; rerun the download cell"
        )
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive, output)
        # During a retry, atomic files of the same size came from completed
        # extraction steps; an incomplete file is retained only as .cvlab-part.
        pending = [
            (m, p)
            for m, p in members
            if not m.is_dir() and (not p.is_file() or p.stat().st_size != m.file_size)
        ]
        required = sum(m.file_size for m, _ in pending)
        if shutil.disk_usage(output).free < required:
            raise RuntimeError(
                f"Not enough disk space to unpack: need another {required / 2**30:.1f} GiB. Free space or use a larger runtime disk."
            )
        print(
            f"Extracting {len(pending)} files ({required / 2**30:.1f} GiB).", flush=True
        )
        for index, (member, target) in enumerate(pending, 1):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".cvlab-part")
            with archive.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)
            if index % 10000 == 0:
                print(f"Extracted {index}/{len(pending)} files", flush=True)
        for member, target in members:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
    root = find_data_root(output)
    temporary_marker = output / (MARKER + ".tmp")
    temporary_marker.write_text(
        json.dumps({"competition": competition, "root": str(root.relative_to(output))})
    )
    temporary_marker.replace(output / MARKER)
    if not keep_archive:
        archive_path.unlink()
    print(f"DATA_ROOT={root}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="Download only test_images.csv to diagnose access",
    )
    args = parser.parse_args()
    if args.check_access:
        check_access(args.competition)
    elif args.out:
        download_data(args.out, args.competition, args.keep_archive)
    else:
        parser.error("--out is required unless --check-access is selected")


if __name__ == "__main__":
    main()
