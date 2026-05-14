# vla-world-model-control

Thin integration layer around vendored `lerobot` for two workflows:

1. Isaac Lab / Isaac Sim simulation work, currently centered on SO-100 and SmolVLA.
2. Real SO-100 / SO-101 follower-arm execution using `lerobot` policies and drivers.

## Repo boundary

- `lerobot/`: vendored upstream code. Avoid editing it during normal cleanup work.
- `isaaclab_arena_vla/`: the main package for this repo's Isaac Lab integration.
- `sim/` and `scripts/`: executable entry points and compatibility wrappers.
- `src/models/vla/`: shared OpenVLA wrapper used by the simulation-side code.

## Current canonical path

The simulation path is the primary focus:

- Isaac Lab / Arena environment code lives under `isaaclab_arena_vla/`.
- SO-100 standalone and Gym wrappers live under `sim/isaac/`.
- SmolVLA training/eval scripts live under `scripts/`.

The real-robot side is intentionally thinner and mostly wires local hardware setup into vendored `lerobot`.

## Cleanup policy

- Prefer changing this repo's integration layer over patching `lerobot/`.
- Keep generated artifacts, caches, and local outputs out of version control.
- Reuse `isaaclab_arena_vla.utils` for repo-root, `PYTHONPATH`, and OpenVLA config handling instead of duplicating that logic in scripts.
