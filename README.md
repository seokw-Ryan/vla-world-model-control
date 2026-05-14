# vla-world-model-control

A research scaffold for evaluating **Vision-Language-Action (VLA) policies** — OpenVLA and SmolVLA — on the **LeRobot SO-100 / SO-101** follower arm, in **Isaac Sim 5.1** and on real hardware, with a single shared observation / action contract between the two so a checkpoint trained in sim runs on the arm with no code changes.

The longer-term motivation (planned future work) is to extend the stack with a learned **world model** for imagination-augmented fine-tuning. The full design rationale, results, and forward plan are in the two reports at the repository root:

- [`report.md`](report.md) — academic write-up (~5 pages)
- [`VIP_report.md`](VIP_report.md) — VIP engineering-design report (~5 pages)

---

## Repository Layout

| Path | Purpose |
|---|---|
| `vla_world_model_control/` | Main Python package |
| `vla_world_model_control/sim/` | Simulation-facing workflows (RL trainer, dispatchers) |
| `vla_world_model_control/robot/` | Real-robot workflows (SO-101 follower runner) |
| `vla_world_model_control/shared/` | Repo paths, YAML loaders, OpenVLA wrapper |
| `sim/isaac/` | Isaac Sim standalone scene + Gymnasium wrapper |
| `scripts/` | Thin entry-point launchers (delegate to the package) |
| `configs/sim/` | Scene YAML (camera, table, cube spawn, physics rate) |
| `configs/vla/` | VLA-specific config (OpenVLA defaults) |
| `assets/so100/` | SO-100 URDF / USD / meshes |
| `outputs/` | Per-run trace logs and checkpoints (gitignored) |
| `lerobot/` | Vendored upstream LeRobot — **do not edit during normal work** |
| `archive/` | Legacy integrations and one-off operational material |
| `tests/` | Unit tests (paths, OpenVLA utils) |

---

## Prerequisites

- **Isaac Sim 5.1 standalone** at `/home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/` (uses its bundled Python 3.11)
- **NVIDIA GPU** with ≥ 16 GB VRAM
- For real hardware: a **LeRobot SO-100 or SO-101** follower arm, a USB camera (OpenCV or RealSense), and the `lerobot` conda environment

All simulation commands run through `python.sh` from the Isaac Sim install; do **not** invoke them from a system Python.

---

## Quick Start — Simulation

> All commands assume the working directory is the repo root.

#### 1. Open the SO-100 scene with no policy (smoke test)

```bash
/home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    scripts/view_so100.py
```

#### 2. Closed-loop VLA control with a camera preset + instruction

```bash
# Top-down view (camera pose lives in configs/sim/so100_standalone.yaml)
/home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    scripts/run_simple_vla.py topdown "extend the arm fully"

# Same scene, different instruction
/home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    scripts/run_simple_vla.py topdown "open the claw"
```

Available camera presets: `topdown`, `front`. Per-step traces (JSONL + PNGs) land under `outputs/standalone_so100/<timestamp>/`.

#### 3. Online RL fine-tuning of SmolVLA on the cube-lift task

```bash
/home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    scripts/train_sim.py \
    --sim_config configs/sim/so100_standalone.yaml \
    --policy_path lerobot/smolvla_base
```

Checkpoints land under `outputs/smolvla_rl/<timestamp>/checkpoints/`, TensorBoard logs in the same directory.

---

## Quick Start — Real Hardware (SO-101 follower)

> Runs in the `lerobot` conda environment, **not** Isaac Sim's Python.

#### Dry-run (reads camera + state, computes actions, does NOT move the arm)

```bash
conda run -n lerobot python scripts/run_smolvla.py \
    --policy_path lerobot/smolvla_base \
    --robot_type so101_follower \
    --robot_port /dev/ttyACM0 \
    --robot_id my_so101 \
    --task "pick up the red cube" \
    --steps 3 \
    --dry_run
```

#### Live execution (drop `--dry_run`)

```bash
conda run -n lerobot python scripts/run_smolvla.py \
    --policy_path lerobot/smolvla_base \
    --robot_type so101_follower \
    --robot_port /dev/ttyACM0 \
    --robot_id my_so101 \
    --task "pick up the red cube"
```

Per-step traces land under `outputs/run_smolvla/<timestamp>/`. Pass `--policy_path` a path to a trained-in-sim checkpoint to evaluate fine-tuned policies on the real arm.

---

## Canonical Entry Points

| Script | Purpose |
|---|---|
| `scripts/run_sim.py` | Simulation dispatcher (so100, franka, smolvla-rl) |
| `scripts/train_sim.py` | SmolVLA reward-weighted RL fine-tuning |
| `scripts/run_simple_vla.py` | Camera preset + instruction → SO-100 closed-loop |
| `scripts/run_robot.py` | Real-robot launcher |
| `scripts/run_smolvla.py` | SmolVLA on real SO-100 / SO-101 follower |
| `scripts/view_so100.py` | Open Isaac Sim viewport on the SO-100 scene |

Older compatibility wrappers (`scripts/run_so101_vla.py`, `scripts/train_smolvla_rl.py`, etc.) remain for backward compatibility but delegate to the canonical scripts above.

---

## Configuration

The single source of truth for the simulated cell is **`configs/sim/so100_standalone.yaml`**. It is consumed by every Isaac Sim entry point, so editing it propagates to both `run_simple_vla.py topdown` and `train_sim.py`.

Key knobs:
- `physics.dt`, `physics.control_decimation` — physics rate (60 Hz default) and control decimation (12 ⇒ 5 Hz control)
- `robot.position`, `robot.orientation` — arm placement on the table
- `camera.position` / `camera.orientation_euler_xyz_deg` / `camera.focal_length` — copied directly from the Isaac Sim viewport's *Property* panel
- `scene.cube.spawn` — table-local rectangle used for episode-reset cube randomization

VLA-side defaults (OpenVLA quantization, prompt template, gripper threshold) live in `configs/vla/openvla_default.yaml`. SmolVLA is loaded by Hugging Face repo id (`--policy_path lerobot/smolvla_base` by default).

---

## Repository Boundary

- **Do not edit `lerobot/`** during normal cleanup work — that is a vendored upstream snapshot.
- Prefer adding behavior in the repo's integration layer (`vla_world_model_control/`) over patching the vendored code.
- Top-level scripts in `scripts/` should stay thin and delegate business logic into the package.
- Keep generated artifacts, caches, and local outputs (`outputs/`, run dirs, checkpoints) out of version control.

---

## Reports

Two write-ups of this project live at the repository root:

- **[`report.md`](report.md)** — Academic-style write-up covering background, system architecture, methodology, preliminary results (including two negative findings), and the planned world-model extension. Targeted at a professor/committee audience.
- **[`VIP_report.md`](VIP_report.md)** — VIP final report structured around the engineering design process: Problem Statement → Prior Art → Potential Solutions → Implemented Solution (with development war stories) → Future Work.

Both reports cite the commands in this README as the reproduction recipe.
