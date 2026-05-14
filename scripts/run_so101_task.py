#!/usr/bin/env python3
"""Run the SO-101 VLA loop with machine-local defaults.

Examples:
    python scripts/run_so101_task.py "pick up the red cube"
    python scripts/run_so101_task.py "pick up the red cube" --dry-run --steps 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_so101_vla.py"

DEFAULT_POLICY_PATH = os.environ.get("SO101_POLICY_PATH", "lerobot/smolvla_base")
DEFAULT_ROBOT_PORT = os.environ.get("SO101_ROBOT_PORT", "/dev/ttyACM0")
DEFAULT_ROBOT_ID = os.environ.get("SO101_ROBOT_ID", "my_so101")
DEFAULT_CAMERA_INDEX = int(os.environ.get("SO101_CAMERA_INDEX", "1"))
DEFAULT_DEVICE = os.environ.get("SO101_DEVICE", "cuda")
DEFAULT_WIDTH = int(os.environ.get("SO101_CAMERA_WIDTH", "640"))
DEFAULT_HEIGHT = int(os.environ.get("SO101_CAMERA_HEIGHT", "480"))
DEFAULT_FPS = int(os.environ.get("SO101_CAMERA_FPS", "30"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SO-101 VLA controller with a plain-text task.")
    parser.add_argument("task", help='Instruction for the robot, for example: "pick up the red cube".')
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH, help="HF model ID or local pretrained model.")
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT, help="SO-101 serial port.")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="SO-101 calibration ID.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="OpenCV camera index.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--steps", type=int, default=200, help="Control loop steps.")
    parser.add_argument("--fps", type=float, default=10.0, help="Control frequency in Hz.")
    parser.add_argument("--log-every", type=int, default=10, help="Print action summary every N steps.")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help="Safety clamp on per-step target motion.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run camera + VLA inference only.")
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow calibration prompts when controlling the real arm.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    camera = json.dumps(
        {
            "front": {
                "type": "opencv",
                "index_or_path": args.camera_index,
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "fps": DEFAULT_FPS,
            }
        }
    )
    rename_map = json.dumps({"observation.images.front": "observation.images.camera1"})

    cmd = [
        sys.executable,
        str(RUNNER),
        "--policy_path",
        args.policy_path,
        "--robot_port",
        args.robot_port,
        "--robot_id",
        args.robot_id,
        "--camera",
        camera,
        "--rename_map",
        rename_map,
        "--task",
        args.task,
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--fps",
        str(args.fps),
        "--log_every",
        str(args.log_every),
        "--max_relative_target",
        str(args.max_relative_target),
    ]

    if args.dry_run:
        cmd.append("--dry_run")
    if args.calibrate:
        cmd.append("--calibrate")
    else:
        cmd.append("--no-calibrate")

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
