# IsaacLab-Arena + LeRobot EnvHub Integration Log

## 2026-03-30 — Phase 1 & 2: Initial Implementation + Smoke Test

### What was done

**Phase 1: Environment Setup**
- Created `scripts/setup_arena.sh` — automated setup script that:
  - Initializes IsaacLab-Arena submodules
  - Installs IsaacLab into Isaac Sim 5.1.0's Python
  - Installs all deps (warp-lang, flatdict, prettytable, pin-pink, onnxruntime, lightwheel-sdk)
  - Removes conflicting `pink.py` from Isaac Sim's site-packages (shadows IK library)
  - Installs IsaacLab-Arena, VLA Arena extension, LeRobot, and OpenVLA deps
- Rewrote `docker/lerobot.Dockerfile` based on Arena's `Dockerfile.isaaclab_arena` pattern:
  - Base image: `nvcr.io/nvidia/isaac-sim:5.1.0` (bumped from Arena's 5.0.0)
  - Build context: IsaacLab-Arena repo root (VLA project mounted at runtime)
  - Note: requires `docker` group membership or sudo for Docker build/run

**Phase 2: Register SO-100 as Arena Environment**
Created the `isaaclab_arena_vla/` package with full Arena integration:

| File | Description |
|------|-------------|
| `isaaclab_arena_vla/embodiments/so100.py` | `SO100Embodiment(EmbodimentBase)` — 6-DOF joint position control, `@register_asset`, USD spawn |
| `isaaclab_arena_vla/tasks/pick_and_place.py` | `SO100PickAndPlaceTask(TaskBase)` — table + red cube, lift-height success, custom `cube_above_height` termination |
| `isaaclab_arena_vla/environments/so100_pick_and_place.py` | `SO100PickAndPlaceEnvironment(ExampleEnvironmentBase)` — composes embodiment + task |
| `isaaclab_arena_vla/policies/openvla_policy.py` | `OpenVLAPolicy(PolicyBase)` — wraps `OpenVLAWrapper`, maps 7D→6D |
| `isaaclab_arena_vla/config/so100_joint_space.yaml` | Joint limits, defaults, control params |
| `setup.py` | Project-level package installation |
| `scripts/test_arena_env.py` | Smoke test: zero-action policy in Arena env |
| `scripts/setup_arena.sh` | One-command setup (deps, submodules, packages) |

### Smoke test results (2026-03-30)

**PASSED** — SO-100 Arena environment runs successfully:

```
[test] Environment created successfully.
[test] Environment reset OK.
[test] Step 0/20
[test] Completed 20 steps.
[test] Metrics: {'success_rate': 0.0, 'num_episodes': 0}
```

Run command:
```bash
ISAACLAB_PATH=/home/rocket/Projects/IsaacLab-Arena/submodules/IsaacLab \
  /home/rocket/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  scripts/test_arena_env.py --headless --num_envs 1 --num_steps 20
```

### Architecture decisions
- **SO-100 uses direct joint position control** (no IK), matching the existing `so100_gym_wrapper.py`
- **VLA 7D → SO-100 6D mapping** from `standalone_so100.py`: dx→shoulder_pan, dy→shoulder_lift, dz→elbow_flex, drx→wrist_flex, dry→wrist_roll
- **Task scene includes table + cube** via `sim_utils.CuboidCfg` spawners (no separate USD)
- **Lazy imports** everywhere — Isaac Sim requires SimulationApp before `omni.*` / `isaaclab.*`
- **`setup.py` at project root** (not inside `isaaclab_arena_vla/`) so `pip install -e .` works

### Dependency issues resolved
- **`pink.py` conflict**: Isaac Sim 5.1 ships a code-formatter `pink.py` that shadows `pin-pink` (IK lib). Setup script removes it.
- **`isort` version**: `pin-pink` pulls `isort>=8` which breaks Isaac Sim's omni packages. Pin to `isort<6`.
- **Missing deps**: `warp-lang`, `flatdict`, `prettytable`, `onnxruntime`, `lightwheel-sdk` not in base Isaac Sim.
- **numpy<2**: Required by IsaacLab and numba; `pin-pink` tries to upgrade to 2.x.

### Docker setup
Build from IsaacLab-Arena root:
```bash
docker build -t vla-arena:latest \
  -f /home/rocket/Projects/vla-world-model-control/docker/lerobot.Dockerfile \
  /home/rocket/Projects/IsaacLab-Arena

docker run --gpus all --rm -it \
  -v /home/rocket/Projects/vla-world-model-control:/workspace/vla \
  vla-arena:latest

# Inside container:
cd /workspace/vla && /isaac-sim/python.sh -m pip install -e .
/isaac-sim/python.sh scripts/test_arena_env.py --headless --num_envs 1
```

**Note**: User needs Docker group membership (`sudo usermod -aG docker $USER`).

### Files structure
```
vla-world-model-control/
├── setup.py                          # Package installer for isaaclab_arena_vla
├── isaaclab_arena_vla/
│   ├── __init__.py
│   ├── embodiments/
│   │   ├── __init__.py               # Lazy imports
│   │   └── so100.py                  # SO100Embodiment + configs
│   ├── environments/
│   │   ├── __init__.py
│   │   └── so100_pick_and_place.py   # Arena environment composition
│   ├── tasks/
│   │   ├── __init__.py               # Lazy imports
│   │   └── pick_and_place.py         # Task + scene + terminations
│   ├── policies/
│   │   ├── __init__.py               # Lazy imports
│   │   └── openvla_policy.py         # OpenVLA as Arena policy
│   └── config/
│       ├── __init__.py
│       └── so100_joint_space.yaml
├── scripts/
│   ├── setup_arena.sh                # One-command setup
│   └── test_arena_env.py             # Smoke test
└── docker/
    └── lerobot.Dockerfile            # Arena + LeRobot container
```

### Next steps (Phase 3 & 4)
- [x] Create Arena environment + embodiment + task
- [x] Run smoke test: zero-action policy passes
- [ ] Build Docker image and verify in container
- [ ] Collect training data using Arena env
- [ ] Publish SO-100 env to HuggingFace EnvHub
- [ ] Create ACT training config for SO-100
- [ ] Train ACT baseline on collected data
- [ ] Benchmark OpenVLA vs ACT vs SmolVLA
- [ ] Evaluate OpenVLA on existing GR1/G1 Arena tasks
