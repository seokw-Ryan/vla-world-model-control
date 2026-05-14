# VLA World Model Control — VIP Final Report

**Project:** Vision-Language-Action Policies on the SO-100/SO-101 Arm — Building a Sim-to-Real Evaluation Stack
**Code repository:** [`github.com/seokw-Ryan/vla-world-model-control`](https://github.com/seokw-Ryan/vla-world-model-control)
**Reproduction commands:** see the repository [`README.md`](README.md) — sections *Quick Start — Simulation* and *Quick Start — Real Hardware* list the exact invocations for every workflow described below.

---

## 1. Problem Statement

Vision-Language-Action (VLA) models — neural policies that map a camera image and a natural-language instruction to a low-level robot action — have recently emerged as a candidate "general" recipe for robot manipulation. Models such as **OpenVLA** (a 7B-parameter Llama-2-based policy) and **SmolVLA** (a 500M-parameter LeRobot policy built on the SmolVLM2 backbone) ship as open-source checkpoints and promise zero-shot or few-shot generalization across tasks.

In practice, two questions remain open for anyone wishing to *use* such a policy on a new arm:

1. **Does the released checkpoint actually work on my arm and my task?** Public VLAs are trained predominantly on demonstrations collected with WidowX-class arms; whether they transfer to a low-cost open-source arm like the LeRobot SO-100/SO-101 is empirically unclear.
2. **If it doesn't, what is the minimum viable fine-tuning loop that makes it work?** Online RL fine-tuning, offline behavior cloning on a small demonstration set, and LoRA-on-top-of-the-VLM are all candidates, and each requires a different software substrate.

To answer either question rigorously, an engineer needs a **single piece of infrastructure that supports four things**:

(a) loading a public VLA checkpoint inside a simulator,
(b) running it in closed loop on a chosen embodiment and task,
(c) collecting trajectories, computing rewards, and updating the policy in that simulator, and
(d) re-deploying the resulting checkpoint on the physical arm with **no further code changes** so that sim-to-real comparisons are valid.

No such stack existed for the SO-100/SO-101 at the start of this VIP semester. The high-level engineering problem this project addressed is therefore:

> **Design and build a complete, reproducible Sim→Train→Real pipeline for evaluating VLA policies on the LeRobot SO-100/SO-101 follower arm, and use it to determine what is required to make a public VLA checkpoint behave usefully on that embodiment.**

The downstream research motivation — adding a learned **world model** so that the same policy can be fine-tuned against imagined rollouts (the eventual reason the repository is named `vla-world-model-control`) — is *out of scope* for this VIP semester and is treated as future work in §5.

---

## 2. Research / Prior Art

The project sits at the intersection of three active research areas; the engineering design was informed by the state of the art in each.

### 2.1 Vision-Language-Action Models

The two checkpoints actually evaluated in this project represent two design philosophies. **OpenVLA** (Kim et al., 2024) is a 7-billion-parameter monolithic model trained on the Open X-Embodiment dataset, outputting 7-DoF end-effector deltas as discretized tokens. **SmolVLA** (LeRobot team, 2024) inverts the trade-off: it pairs a much smaller 500M-parameter VLM backbone with a *flow-matching action expert head* and accepts up to three camera streams plus a proprioceptive state vector, producing chunked action sequences. SmolVLA is explicitly designed to be fine-tunable on a single GPU.

Other published VLAs — π0 (Black et al., 2024), RT-2 (Brohan et al., 2023), Octo (Octo Team, 2024) — were considered for inclusion but ruled out either because the weights are not publicly accessible (π0, RT-2) or because the action representation is incompatible with the SO-100 joint-delta interface (Octo). OpenVLA and SmolVLA were the two viable candidates.

### 2.2 The LeRobot Ecosystem and the SO-100 / SO-101

LeRobot (Cadène et al., 2024) is the de-facto open-source robotics framework for low-cost arms. It bundles policy implementations, dataset format, hardware drivers, and a teleoperation/calibration toolchain. The SO-100 / SO-101 are 3D-printable 6-DoF follower arms (5 arm joints + 1 gripper) controlled over USB-CDC serial. They were chosen as the target hardware because (a) the LeRobot driver is reliable, (b) the arm is cheap enough to risk policy-driven failures, and (c) prior work in the LeRobot community has shown imitation learning works on it.

### 2.3 Simulator Choice

Three simulators were realistically usable: **MuJoCo** (`mujoco` Python bindings), **PyBullet**, and **NVIDIA Isaac Sim 5.1 / Isaac Lab**. Isaac Sim was chosen for the engineering reasons discussed in §3.2; Isaac Lab's manager-based environment system was also evaluated and the scaffolding remains in the repository for future Franka-class experiments.

### 2.4 World Models (Forward Reference)

The intended next phase of the project is to fit a learned generative model of the camera frame and proprioceptive state (a "world model" — Ha & Schmidhuber, 2018; Hafner et al., 2023, Dreamer-V3) on data collected from the stack built this semester, and to compare model-free vs. model-based fine-tuning. None of that has been built yet; it is documented in §5.

---

## 3. Potential Solutions

The core problem of §1 admits at least four reasonable engineering architectures. Each was considered, and the trade-offs that drove the final choice are summarized below.

### 3.1 Option A — MuJoCo-based stack with `lerobot`-native integration

**What it would look like:** Use LeRobot's existing `lerobot-sim` MuJoCo bindings, define the SO-100 in MuJoCo XML, and run SmolVLA in the loop using LeRobot's own training scripts.

**Pros:** Stays inside LeRobot's normal Python environment; fastest first time to a working loop; community has shipped reference configs.

**Cons:** MuJoCo's USD/import story for the SO-100 is brittle, the renderer is comparatively low-fidelity, and Isaac Lab's parallel-environment / tiled-camera benefits would be unavailable for the future world-model phase. The visual gap to the real-arm camera would be larger.

### 3.2 Option B — Isaac Sim standalone Python with a custom Gym wrapper

**What it would look like:** Run Isaac Sim 5.1 in its bundled Python 3.11 standalone mode, import the SO-100 URDF directly onto the live USD stage, write a thin `gymnasium.Env` wrapper around the world, and load the VLA inside the *same* Python process (because Isaac Sim refuses to be a subprocess of normal Python).

**Pros:** GPU-accelerated PhysX 5; high-quality RTX renderer with realistic camera intrinsics (matters for sim-to-real); easy path to Isaac Lab's parallel environments later; native USD asset pipeline that survives into the world-model phase.

**Cons:** Isaac Sim's bundled Python ships its own (newer) transformers, which conflicts with OpenVLA's pinned dependencies; the SimulationApp must be created *before* any `torch`/`omni` import, which constrains module structure; the shutdown sequence has a known atexit-time crash in `libomni.graph.core` that masks Python tracebacks.

### 3.3 Option C — Isaac Lab manager-based RL environment

**What it would look like:** Define the SO-100 task as an Isaac Lab `ManagerBasedRLEnv` config class, plug into Isaac Lab's standard reward/termination/observation managers, and use Isaac Lab's RL runners.

**Pros:** Idiomatic for Isaac Lab; easy to scale to massively parallel environments; tooling for differentiable rendering.

**Cons:** Steeper learning curve; the manager system is heavy for a single-arm tabletop task; loading a VLA policy *inside* an Isaac Lab env is awkward because Isaac Lab assumes batched parallel envs and VLAs are inherently per-env-batch-1 (each inference is ~hundreds of ms).

### 3.4 Option D — Out-of-process VLA inference server

**What it would look like:** Run the VLA in a separate Python process (or GPU container) and have the simulator talk to it over gRPC / HTTP.

**Pros:** Decouples the VLA's Python environment from Isaac Sim's; transformers/lerobot version conflicts vanish; the same VLA server can be used by the real-robot runner.

**Cons:** Significant added latency for what is already a slow inference call; harder to attach a debugger across the boundary; complicates per-step logging; the VLA's GPU footprint plus the simulator's RTX footprint may not coexist on a single 16 GB card.

### 3.5 Decision

**Option B was chosen**, with the following supporting decisions:

- **A vendored `lerobot/` checkout under `lerobot/src/`** (rather than the system-installed `lerobot==0.4.4`) — the system package transitively imports a broken `lerobot.policies.groot` module that does not load under Isaac Sim's Python 3.11. The vendored version is editable and bypasses the broken import path.
- **A single YAML scene config** (`configs/sim/so100_standalone.yaml`) shared between all entry points so that the camera pose, table geometry, cube spawn rectangle, and physics rate are *one source of truth*.
- **Identical observation- and action-packing code paths between sim and hardware** (`sim/isaac/so100_gym_wrapper.py` and `vla_world_model_control/robot/run_smolvla.py` build the same dict shape with the same numeric conventions), so a checkpoint that runs in sim runs on hardware with zero code changes.

The other options were not blind alleys: the Isaac Lab scaffolding from Option C remains in the repository for a future Franka comparison, and a `--dry_run` flag on the hardware runner gives most of Option D's safety properties without the latency cost.

---

## 4. Implemented Solution

### 4.1 Architecture Overview

The implemented stack is organized in five layers, from bottom to top:

```
┌─────────────────────────────────────────────────────────────┐
│  Entry-point scripts (scripts/run_simple_vla.py,            │
│      scripts/train_sim.py, scripts/run_smolvla.py, ...)     │
├─────────────────────────────────────────────────────────────┤
│  vla_world_model_control/ Python package                    │
│      sim/         — train_smolvla_rl.py, dispatchers        │
│      robot/       — run_smolvla.py (real hardware)          │
│      shared/      — paths, OpenVLA wrapper, YAML loaders    │
├─────────────────────────────────────────────────────────────┤
│  sim/isaac/                                                 │
│      standalone_so100.py    (single-process closed loop)    │
│      so100_gym_wrapper.py   (gym.Env for training)          │
│      lab_env/               (Isaac Lab Franka scaffolding)  │
├─────────────────────────────────────────────────────────────┤
│  Vendored lerobot/  (SmolVLA, drivers, dataset format)      │
├─────────────────────────────────────────────────────────────┤
│  Isaac Sim 5.1 standalone Python (bundled torch + PhysX 5)  │
└─────────────────────────────────────────────────────────────┘
```

The data path is identical in both modes. At each control step:

1. The camera returns a (3, H, W) uint8 RGB image and the arm returns a (6,) state vector (5 joint radians + 1 gripper sim unit).
2. These are packed into the LeRobot-conventional observation dict and the task string is tokenized using the SmolVLM2 processor (max 48 tokens).
3. The policy (`SmolVLAPolicy.select_action`) returns a (6,) action in $[-1, 1]^6$.
4. The first five action dimensions are scaled by `0.1`, added to the current arm joints, and clamped to URDF limits; the sixth is thresholded to the gripper open/closed positions.
5. In sim, the target joints are written through `Robot.set_joint_positions` and physics is integrated for 12 steps at 60 Hz (yielding a 5 Hz control rate that matches the LeRobot hardware driver); on hardware, the joint targets in degrees are sent through the LeRobot follower driver at the same 5 Hz.
6. A per-step JSONL trace and (optionally) a PNG snapshot of the policy's input image are written to `outputs/<runner>/<timestamp>/` for offline analysis.

### 4.2 How the Solution Works — Component Detail

**Scene construction.** The SO-100 is imported from URDF directly onto the live Isaac Sim stage using `URDFParseAndImportFile`. The imported articulation lands with an authored zero root translate, which silently hides the runtime placement in the Stage UI; the solution writes the real position and orientation through `UsdGeom.Xformable.AddTranslateOp / AddOrientOp` immediately after import. A 0.70 × 0.60 × 0.04 m table and a 3 cm dynamic red cube are added to the scene; the cube's spawn position is sampled at every `reset()` from a table-local rectangle (`tabletop_rect` mode) so the policy sees a different scene every episode without depending on the robot's frame.

**Camera authoring.** The simulator's camera pose has to match what the operator sees in the Isaac Sim viewport. The wrapper supports three precedence levels: explicit `orientation_wxyz`, explicit `orientation_euler_xyz_deg` (matching USD's Property-panel "Orient" widget, applied as intrinsic XYZ via `scipy.spatial.transform.Rotation.from_euler("xyz", …, degrees=True)`), or a fallback computed look-at from `(eye, target)`. This is the single most-used knob during day-to-day iteration: the operator drags the camera to the desired view in the viewport, copies the Translate / Orient / Focal Length values into the YAML, and every entry-point script then loads at that exact pose.

**Policy loading.** SmolVLA is loaded with `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`. The wrapper inspects `policy.config.input_features` at load time and discovers which image keys the checkpoint expects — the base checkpoint expects `observation.images.camera1/camera2/camera3`, while some fine-tuned variants use `observation.images.front`. The single Isaac Sim camera is broadcast to every expected slot so the script is robust to either convention.

**OpenVLA fallback.** The older OpenVLA loader (`shared/openvla_wrapper.py`) remains available for comparison. Loading it inside Isaac Sim required several compatibility shims: patching missing tokenizer symbols (`PaddingStrategy`, `PreTokenizedInput`, etc.) from `tokenization_utils_base` into `tokenization_utils`, forcing eager attention, injecting `GenerationMixin` into the dynamically-imported Prismatic model class, and rewriting the `tie_weights` signature to absorb the newer `recompute_mapping` kwarg.

**Online RL fine-tuning.** `train_smolvla_rl.py` implements reward-weighted regression: episodes are rolled out with Gaussian exploration noise on the arm dimensions and a 5 % Bernoulli flip on the gripper bit; transitions are buffered in a 20 000-capacity replay; every 5 episodes the trainer draws 10 batches of 4 transitions, computes per-sample loss, weights each sample by `exp(2.0 · standardized_return)` clipped at 20, and takes a single AdamW step per batch (lr = 1e-4, grad clip 5.0). The vision encoder is frozen; only the SmolVLA *expert* head, state projection, and action-time MLPs are trainable.

**Hardware deployment.** `vla_world_model_control/robot/run_smolvla.py` connects to the SO-101 over `/dev/ttyACM0`, opens an OpenCV camera, runs the calibration flow if needed, and executes the same per-step loop as the simulator — including the same 0.1 delta scale and gripper threshold rule. `--dry_run` reads observations and computes actions but does not send them to the servos, providing a safe smoke test.

### 4.3 Problems Encountered During Development

The path from "Isaac Sim opens" to "policy runs end-to-end" exposed a chain of non-obvious failures, each of which required diagnosis rather than a Google-able fix.

1. **The lerobot package on the system is partially broken.** The system-installed `lerobot==0.4.4` ships a `lerobot.policies.groot` submodule whose `@dataclass` declaration has non-default arguments following default ones, which raises at import time and aborts any `from lerobot.*` import (Python 3.11 dataclass change). Solution: vendor `lerobot/src/` and prepend it to `sys.path`.

2. **OpenVLA's remote-code class is not compatible with the transformers bundled by Isaac Sim 5.1.** The Prismatic processor's remote code uses `transformers.tokenization_utils.PaddingStrategy`, which has moved to `tokenization_utils_base` in newer transformers. Solution: at load time, alias the missing symbols on the older module path before any OpenVLA code executes.

3. **OpenVLA produced bit-exact identical actions for 80 consecutive steps.** Initial debugging suspected a stale camera frame or instruction caching, but the per-step image-hash log added during development showed the camera *was* updating and the policy *was* re-reading both. The policy itself collapsed to a constant output under the SO-100 visual distribution — consistent with WidowX-trained statistics. This negative result motivated the switch to SmolVLA.

4. **SmolVLA loaded but errored at `select_action` with "All image features missing from the batch".** The base checkpoint expects image keys named `camera1` / `camera2` / `camera3`, not `front` as used in fine-tuned variants. Solution: inspect `policy.config.input_features` and feed the single camera to every expected slot.

5. **`KeyError: 'observation.language.tokens'`.** The model code reads `OBS_LANGUAGE_TOKENS = "observation.language.tokens"` (a *dot* between `language` and `tokens`), but the LeRobot real-robot runner script used the underscore form. The dotted form is the authoritative one defined in `lerobot/utils/constants.py`.

6. **Camera framing was off-axis even after we set look-at.** Isaac Sim's `lookat_to_quatf` returns the identity quaternion when looking straight down with `+Y` as the up hint, which is correct in USD's camera frame but not in world coordinates. Solution: support an explicit `orientation_euler_xyz_deg` override that bypasses look-at entirely and uses USD's intrinsic-XYZ Euler convention with `camera_axes="usd"` on `set_world_pose`.

7. **Every successful run was followed by a fatal-looking segfault traceback.** Isaac Sim 5.1 has a known atexit-time crash inside `libomni.graph.core` during `SimulationApp.close()` — but importantly, that traceback was *also* burying genuine Python `NameError`s that happened during the run (in our case, a missing `from scipy.spatial.transform import Rotation` import was visible only after grepping past 150 lines of carb crash-reporter noise). Solution: after the trace files are flushed and `world.stop()` is called, hard-exit with `os._exit(0)` to bypass the broken atexit handlers, and use `grep -v crashreporter` in run scripts to see real errors first.

8. **The arm hit joint limits within ~30 steps.** With an action scale of `0.1` rad per step at 5 Hz, even the SmolVLA-base policy — which was effectively asking for the max delta in every dimension — drove every arm joint to its limits and pinned there for the remainder of the episode. The reward signal during these runs was therefore mostly noise around a zero baseline, which is precisely why RL fine-tuning failed to converge.

### 4.4 What the Implementation Produced

Three concrete deliverables exist at the end of the semester:

- **A working, reproducible Sim→Real pipeline**, exercised end-to-end: an Isaac Sim closed-loop runner, a Gym environment, an RL trainer, and a real-SO-101 deployment script that all consume the same YAML and produce the same observation/action shapes.
- **Two policy threads evaluated**: OpenVLA zero-shot in sim (negative — degenerate constant output) and SmolVLA-base in sim + on SO-101 hardware (negative — goal-undirected motion). Both negative results are documented with per-step traces in `outputs/`.
- **An RL fine-tuning loop that runs but does not converge.** Nine timestamped runs are archived under `outputs/smolvla_rl/`. The diagnosis (no successful rollouts to weight on; arm saturates against limits within ~30 steps) is concrete and points to the fix described in §5.

---

## 5. Future Work

The single most important next step is to **provide bootstrap data before invoking reinforcement learning**. The empirical evidence from this semester is unambiguous: reward-weighted regression cannot bootstrap a useful policy when the prior policy never succeeds, because there are no high-return rollouts to up-weight. Three concrete near-term work items follow from this:

1. **Scripted-IK demonstration collection.** Write a heuristic controller that uses inverse kinematics to move the gripper above the cube, descend, close, and lift, recording (observation, action) pairs at the same 5 Hz rate as the policy. Twenty to fifty such demonstrations should be enough to behavior-clone SmolVLA into a policy that succeeds at the task most of the time, after which the existing RL loop becomes refinement rather than search.

2. **Domain randomization on the simulator camera.** The negative real-robot result confirmed that the simulated visual distribution is too far from the real-arm camera. Future runs will randomize lighting, table texture, and cube color, and the actual camera intrinsics and extrinsics from the real cell will be measured and reproduced in the YAML scene config.

3. **Action-scale and reset heuristics.** The arm-saturation issue can be mitigated cheaply by (a) lowering the delta scale from 0.1 to ~0.03 so the arm reaches limits in ~100 steps instead of 30, and (b) detecting saturation and triggering an early reset so the trainer sees more diverse rollouts.

Beyond these immediate fixes, the larger research direction — which is the eventual reason the project is named **vla-world-model-control** — is to introduce a learned **world model**. Once §5.1–5.3 yield a policy that succeeds reasonably often, the plan is to train a Dreamer-V3-style generative model of the camera frame and proprioceptive state on the trajectories collected, and then compare three regimes:

- **Model-free baseline:** continue reward-weighted RL on simulator rollouts.
- **Model-based (imagination):** roll out the policy *inside* the world model and apply policy-gradient updates on those imagined trajectories, re-grounding against the simulator only periodically. This is the Dreamer / DayDreamer recipe.
- **Model-based (planning):** at inference time, run short-horizon MPC over the world model to refine the action proposed by SmolVLA, leaving the policy weights unchanged.

The infrastructure described in §4 was deliberately built so that this comparison is a matter of swapping the update rule, not the surrounding plumbing. The world-model component is the central piece of work for the next phase of this project.

Finally, once a working pipeline exists for the SO-100/SO-101, two natural further comparisons become straightforward: different VLA backbones on the same task (SmolVLA vs. π0 vs. OpenVLA-LoRA vs. an ACT baseline) using the LeRobot policy registry, and different action representations (joint deltas vs. delta end-effector pose with IK vs. action chunking with diffusion expert heads) by swapping the action-application stage of the Gym wrapper.

---

## 6. Code and Reproducibility

All code, configs, URDF/USD assets, and per-step trace logs from the runs reported in §4 are available at:

**https://github.com/seokw-Ryan/vla-world-model-control**

The repository [`README.md`](README.md) contains the **exact reproduction commands** for every workflow discussed above, organized as:

- *Quick Start — Simulation* — scene viewer, closed-loop VLA control with the `topdown` / `front` camera presets, and online SmolVLA RL fine-tuning;
- *Quick Start — Real Hardware* — SO-101 follower dry-run and live execution.

Both this VIP report (`VIP_report.md`) and the longer academic write-up (`report.md`) are committed at the repository root for future reference. The commands are reproducible with the Isaac Sim 5.1 standalone Python and a 16 GB+ NVIDIA GPU.
