"""Thin simulation launcher for the supported sim entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys

from vla_world_model_control.shared import PROJECT_ROOT


SIM_TARGETS = {
    "so100-standalone": PROJECT_ROOT / "sim" / "isaac" / "standalone_so100.py",
    "franka-standalone": PROJECT_ROOT / "sim" / "isaac" / "standalone_vla.py",
    "smolvla-rl": PROJECT_ROOT / "scripts" / "train_smolvla_rl.py",
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run a supported simulation entry point.")
    parser.add_argument(
        "target",
        choices=tuple(SIM_TARGETS),
        nargs="?",
        default="smolvla-rl",
        help="Simulation workflow to launch.",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    cmd = [sys.executable, str(SIM_TARGETS[args.target]), *passthrough]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
