#!/usr/bin/env python3
"""Compatibility wrapper for the moved SmolVLA real-robot runner."""

from vla_world_model_control.robot.run_smolvla import main


if __name__ == "__main__":
    raise SystemExit(main())
