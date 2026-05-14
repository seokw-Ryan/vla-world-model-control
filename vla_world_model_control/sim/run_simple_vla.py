"""Small wrapper for simple SO-100 VLA simulation commands.

This keeps the user-facing surface intentionally small:
1. camera preset
2. instruction text
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from vla_world_model_control.shared import PROJECT_ROOT


CAMERA_PRESETS: dict[str, dict[str, object]] = {
    # "topdown" defers to configs/sim/so100_standalone.yaml so the viewport-authored
    # pose (Translate / Orient / Focal Length) is the single source of truth.
    "topdown": {},
    "front": {
        "eye": (-0.38, 0.14, 0.58),
        "target": (-0.18, 0.14, 0.42),
        "focal_length": 1.5,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SO-100 SmolVLA sim with a simple camera preset + text command.",
    )
    parser.add_argument(
        "camera",
        choices=tuple(CAMERA_PRESETS),
        help="Camera preset to use for the VLA input image.",
    )
    parser.add_argument(
        "text",
        help='Instruction text, e.g. "extend the arm fully" or "open the claw".',
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum control steps.")
    parser.add_argument(
        "--policy_path",
        default=None,
        help="Optional SmolVLA checkpoint override. Defaults to lerobot/smolvla_base.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preset = CAMERA_PRESETS[args.camera]

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "sim" / "isaac" / "standalone_so100.py"),
        "--task",
        args.text,
        "--max_steps",
        str(args.max_steps),
    ]
    if "eye" in preset:
        cmd.extend(["--camera_eye", *(str(v) for v in preset["eye"])])
    if "focal_length" in preset:
        cmd.extend(["--camera_focal_length", str(preset["focal_length"])])
    if "orientation_wxyz" in preset:
        cmd.extend(["--camera_orientation_wxyz", *(str(v) for v in preset["orientation_wxyz"])])
    elif "target" in preset:
        cmd.extend(["--camera_target", *(str(v) for v in preset["target"])])

    if args.headless:
        cmd.append("--headless")
    if args.policy_path:
        cmd.extend(["--policy_path", args.policy_path])

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
