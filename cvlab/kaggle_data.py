"""Download the course competition with the official Kaggle CLI.

Credentials are inherited from the environment or Kaggle's standard credential
files. This module never accepts, prints or writes credential values.
"""

import argparse
import json
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

COMPETITION = "multiple-object-video-detection-hw-2-ucu-ai-2026"
COMPETITION_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"
MARKER = ".cvlab-download-complete.json"


def find_data_root(directory):
    """Find exactly one complete competition layout, including nested archives."""
    directory = Path(directory)
    roots = []
    for csv in directory.rglob("test_images.csv"):
        root = csv.parent
        required = [root / "sample_submission.csv"]
        folders = [
            root / split / folder
            for split in ("train", "val")
            for folder in ("images", "labels")
        ]
        folders.append(root / "test" / "images")
        if all(p.is_file() for p in required) and all(p.is_dir() for p in folders):
            roots.append(root)
    if len(roots) != 1:
        raise ValueError(
            f"Expected one dataset with train/, val/, test/, test_images.csv and sample_submission.csv; found {len(roots)} in {directory}"
        )
    return roots[0]


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


def download_data(output, competition=COMPETITION, keep_archive=False):
    if not competition or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in competition
    ):
        raise ValueError("Use a Kaggle competition slug, not a URL")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = ready_data_root(output, competition)
    if ready is not None:
        print(f"Dataset already prepared: {ready}")
        return ready
    archive_path = output / f"{competition}.zip"
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        executable = shutil.which("kaggle")
        if executable is None:
            raise RuntimeError(
                "Install the downloader first: python -m pip install -e '.[kaggle]'"
            )
        print(
            f"Downloading {competition}. The archive contains approximately 11 GB of competition data.",
            flush=True,
        )
        arguments = [
            executable,
            "competitions",
            "download",
            competition,
            "--path",
            str(output),
        ]
        if archive_path.exists():
            arguments.append("--force")
        try:
            # Tokens stay in the inherited environment, never in command arguments.
            subprocess.run(arguments, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "Kaggle download failed. For HTTP 401, check your API token. For HTTP 403, "
                f"join the competition and accept its rules using the same account: https://www.kaggle.com/competitions/{competition}/rules"
            ) from None
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    download_data(args.out, args.competition, args.keep_archive)


if __name__ == "__main__":
    main()
