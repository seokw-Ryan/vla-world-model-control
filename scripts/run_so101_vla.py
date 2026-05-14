#!/usr/bin/env python3
"""Compatibility wrapper for the moved SO-101 real-robot runner."""

from vla_world_model_control.robot.run_so101_vla import main


if __name__ == "__main__":
    raise SystemExit(main())
