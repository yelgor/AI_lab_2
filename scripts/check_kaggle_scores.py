"""Read submission scores with the official CLI; never submit or prompt for tokens."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "local_supervision"
    output.mkdir(exist_ok=True)
    credentials = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    ready = bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or any((credentials / name).is_file() for name in ("access_token", "kaggle.json"))
    )
    status = {"checked_at": time.time(), "competition": "multiple-object-video-detection-hw-2-ucu-ai-2026", "authenticated": ready}
    if not ready:
        status["status"] = "blocked_missing_local_kaggle_authentication"
    else:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "kaggle.cli", "competitions", "submissions",
                 status["competition"], "--csv"],
                capture_output=True, text=True, timeout=45,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
            status["exit_code"] = result.returncode
            # Never persist CLI errors: these may contain credentials or headers.
            if result.returncode:
                status["status"] = "cli_failed_check_authorization_or_connectivity"
            else:
                rows = list(csv.DictReader(io.StringIO(result.stdout)))
                if rows and not {"fileName", "publicScore"} <= set(rows[0]):
                    status["status"] = "unexpected_cli_csv_schema"
                else:
                    path = output / "kaggle_scores.csv"
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(result.stdout)
                    temporary.replace(path)
                    status.update(status="ok", submissions=len(rows), scores_csv=str(path))
        except (OSError, subprocess.TimeoutExpired):
            status["status"] = "cli_unavailable_or_timed_out"
    path = output / "kaggle_scores_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(path)
    print(status["status"])


if __name__ == "__main__":
    main()
