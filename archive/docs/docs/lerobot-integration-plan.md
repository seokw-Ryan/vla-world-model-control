# LeRobot on Isaac Sim — Implementation Plan

## What is LeRobot?

HuggingFace's open-source robot learning platform (22k+ stars). It provides policies (ACT, Diffusion Policy, Pi0, SmolVLA, GR00T N1.5), a standardized dataset format (LeRobotDataset v3), and a Gymnasium-based env interface for training and evaluation.

## Key Finding: Official Integration Already Exists

NVIDIA and HuggingFace have an **Isaac Lab-Arena** integration (`isaaclab_arena` env type in LeRobot). However, that requires Isaac Lab + Isaac Lab-Arena packages. This project uses **Isaac Sim standalone** with a custom Franka setup, so we need a lighter custom integration.

---

## Proposed Architecture (4 phases)

### Phase 1: Gymnasium Wrapper for Isaac Sim Env

Create `sim/isaac/gym_wrapper.py` — wraps the existing standalone Franka scene as a `gymnasium.Env`:

- `observation_space`: Dict with `observation.state` (7-DOF joints) + `observation.images.front` (camera RGB)
- `action_space`: Box(7,) — 3 pos delta + 3 rot delta + 1 gripper
- `reset()`: resets world, returns obs
- `step(action)`: applies delta via IK solver, steps sim, returns obs/reward/terminated/info
- Reuses all existing scene setup, camera, IK solver code from `standalone_vla.py`

### Phase 2: Data Collection Pipeline

Create `scripts/collect_lerobot_data.py` — collects demonstrations into LeRobotDataset v3 format:

- Uses `LeRobotDataset.create()` with the feature schema
- Supports scripted demos (e.g., move-to-cube-and-grasp) or teleoperation
- Saves episodes as Parquet + MP4, compatible with LeRobot training
- Can push to HuggingFace Hub

### Phase 3: LeRobot Policy Training

Create `configs/lerobot/` with training configs for:

- **ACT** (recommended first — trains fast, works with ~50 demos, ~80M params fits 16GB GPU)
- **Diffusion Policy** (alternative)
- Train with: `lerobot-train --policy=act --dataset.repo_id=your/dataset`

### Phase 4: LeRobot Policy Evaluation in Isaac Sim

Create `scripts/eval_lerobot_policy.py` — loads trained LeRobot policy and runs closed-loop in Isaac Sim env:

- Loads policy checkpoint via LeRobot's `make_policy()`
- Runs inference loop similar to `standalone_vla.py` but using LeRobot policy output
- Reports success rate metrics

---

## File Changes

| File | Action | Purpose |
|------|--------|---------|
| `sim/isaac/gym_wrapper.py` | **New** | Gymnasium wrapper around Isaac Sim Franka env |
| `scripts/collect_lerobot_data.py` | **New** | Demonstration collection -> LeRobotDataset v3 |
| `scripts/eval_lerobot_policy.py` | **New** | Evaluate trained LeRobot policy in sim |
| `configs/lerobot/act_franka.yaml` | **New** | ACT training config |
| `requirements-lerobot.txt` | **New** | `lerobot>=0.5.0` + deps |

## Dependencies

- `lerobot>=0.5.0` (installed in Isaac Sim's Python env via `python.sh -m pip install`)
- Existing Isaac Sim 5.1.0 setup stays unchanged

## Why This Approach?

- **Reuses existing scene/robot setup** — no need for Isaac Lab or Arena
- **Standard Gymnasium interface** — compatible with any LeRobot policy out of the box
- **ACT first** — smallest model, fastest to train, proven on manipulation tasks, fits 16GB VRAM
- **Upgradable** — once the gym wrapper exists, swap in Diffusion Policy, Pi0, SmolVLA, etc. with zero env changes
