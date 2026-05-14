#!/usr/bin/env python3
"""Thin launcher for supported simulation entry points."""

from vla_world_model_control.sim.dispatch import main


if __name__ == "__main__":
    raise SystemExit(main())
