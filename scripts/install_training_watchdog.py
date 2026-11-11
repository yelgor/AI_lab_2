"""Install a per-user systemd training supervisor with a bounded automatic plan."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cvlab.training_watchdog import validate_config, write_json  # noqa: E402


def unit_quote(value, *, executable=False):
    value = str(value)
    if any(c in value for c in ("\n", "\r", "\0")):
        raise ValueError("Unit paths must not contain newlines or NUL")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if executable:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def render_unit(project, python, config):
    # WorkingDirectory consumes a raw path, unlike ExecStart's quoted argv parser.
    unit_quote(project)  # Validate characters before interpolating a unit directive.
    working_directory = str(project).replace("%", "%%")
    return "\n".join(
        [
            "[Unit]",
            "Description=CVLab bounded training watchdog",
            "StartLimitIntervalSec=300",
            "StartLimitBurst=3",
            "",
            "[Service]",
            "Type=simple",
            "WorkingDirectory=" + working_directory,
            "ExecStart="
            + unit_quote(python, executable=True)
            + " -m cvlab.training_watchdog --config "
            + unit_quote(config, executable=True),
            "Environment=PYTHONUNBUFFERED=1",
            "UnsetEnvironment=PYTHONPATH CVLAB_DATA_ROOT",
            "Restart=on-failure",
            "RestartSec=30",
            "KillMode=process",
            "TimeoutStopSec=15",
            "StandardOutput=journal",
            "StandardError=journal",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--mount", type=Path)
    parser.add_argument(
        "--device",
        help="Optional block device for ordinary unprivileged UDisks mounting",
    )
    parser.add_argument("--name", default="cvlab-watchdog")
    parser.add_argument("--total-hours", type=float, default=8)
    parser.add_argument("--session-hours", type=float, default=2)
    parser.add_argument("--reserve-hours", type=float, default=0.5)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-ram-mib", type=int, default=6144)
    parser.add_argument("--min-disk-gib", type=float, default=8)
    parser.add_argument("--stall-minutes", type=float, default=30)
    parser.add_argument(
        "--enable-linger",
        action="store_true",
        help="Ask logind to keep the user manager running after logout",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Write configuration/unit without enabling the service",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rearm",
        action="store_true",
        help="Reset an idle plan's budget/retry latch; preserves all training artifacts",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.name):
        parser.error("Use a lowercase service name with letters, digits and hyphens")
    home = Path.home()
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    state_root = Path(os.environ.get("XDG_STATE_HOME", home / ".local/state"))
    config_path = config_root / args.name / "config.json"
    unit = args.name + ".service"
    if args.status or args.pause or args.resume or args.rearm:
        if args.rearm:
            subprocess.run(["systemctl", "--user", "stop", unit], check=True)
        mode = (
            "--status"
            if args.status
            else "--pause"
            if args.pause
            else "--resume"
            if args.resume
            else "--rearm"
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cvlab.training_watchdog",
                    "--config",
                    str(config_path),
                    mode,
                ],
                cwd=PROJECT,
                check=True,
            )
        finally:
            if args.rearm:
                subprocess.run(["systemctl", "--user", "start", unit], check=True)
        return
    if not args.work:
        parser.error("--work is required for installation")
    config = {
        "project": str(PROJECT),
        "work": str(args.work.resolve()),
        "mount": str(args.mount.resolve()) if args.mount else None,
        "device": args.device,
        "state_dir": str((state_root / args.name).resolve()),
        **{
            k: getattr(args, k)
            for k in (
                "total_hours",
                "session_hours",
                "reserve_hours",
                "max_cycles",
                "max_retries",
                "interval",
                "gpu",
                "min_ram_mib",
                "min_disk_gib",
                "stall_minutes",
            )
        },
    }
    validate_config(config)
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            parser.error(
                "Existing service has a different plan; use a different --name rather than changing live configuration"
            )
    Path(config["state_dir"]).mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    unit_path = config_root / "systemd/user" / unit
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(render_unit(PROJECT, sys.executable, config_path))
    print("Config:", config_path)
    print("Local status:", Path(config["state_dir"]) / "status.json")
    if args.enable_linger:
        result = subprocess.run(["loginctl", "enable-linger", str(os.getuid())])
        if result.returncode:
            print(
                "Linger was not enabled; the service still works while the user manager is running."
            )
    if not args.no_start:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        running = (
            subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", unit]
            ).returncode
            == 0
        )
        subprocess.run(["systemctl", "--user", "enable", "--now", unit], check=True)
        if running:
            subprocess.run(["systemctl", "--user", "restart", unit], check=True)
        print("Started:", unit)
    print(
        "Inspect: python3 scripts/install_training_watchdog.py --name",
        args.name,
        "--status",
    )


if __name__ == "__main__":
    main()
