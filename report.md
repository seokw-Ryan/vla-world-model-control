# VLA World Model Control: Building a Simulation-to-Hardware Stack for Vision-Language-Action Policies on the SO-100/SO-101 Arm

*Independent Study Report*

**Code repository:** [`github.com/seokw-Ryan/vla-world-model-control`](https://github.com/seokw-Ryan/vla-world-model-control) — all scripts, configs, and per-step trace logs referenced in this report are reproducible from this commit. **The exact commands used to produce every result reported below are listed in the repository [`README.md`](README.md)** (sections *Quick Start — Simulation* and *Quick Start — Real Hardware*).

---

## 1. Introduction

Vision-Language-Action (VLA) models such as OpenVLA (Kim et al., 2024) and SmolVLA (Hugging Face / LeRobot, 2024) propose a unified architecture in which a pretrained vision-language model is fine-tuned to output low-level robot actions conditioned on an image and a natural-language instruction. These models have shown promising generalization on tabletop manipulation tasks when trained on large heterogeneous robot datasets, but their behavior on novel embodiments — particularly low-cost open-source arms such as the LeRobot SO-100 / SO-101 follower — remains poorly characterized.

The aim of this independent study is to construct an end-to-end research stack that supports two complementary modes of evaluation:

1. **In simulation**, using NVIDIA Isaac Sim 5.1 and Isaac Lab to expose a Gymnasium-compatible SO-100 tabletop environment in which VLA policies can be evaluated, imitated, or reinforcement-learning fine-tuned;
2. **On real hardware**, using LeRobot's drivers to execute the same policies on a physical SO-101 follower arm with an RGB camera as the sole exteroceptive input.

The project is named `vla-world-model-control` because its longer-term motivation is to extend the system with a **learned world model** — a generative model of the camera frame and proprioceptive state under simulated actions — and to compare model-free fine-tuning of VLAs against model-based planning over latent rollouts. **This report covers the infrastructure and preliminary policy evaluation phase only; the world-model component is future work** and is described as such in §6.

The remainder of the report is organized as follows. §2 reviews relevant background. §3 describes the hardware and software architecture of the stack. §4 details the policy stack and the three training/evaluation threads pursued. §5 reports preliminary results, including two important *negative* findings. §6 discusses limitations and outlines the planned trajectory toward a model-based extension.

---

## 2. Background and Related Work

### 2.1 Vision-Language-Action Models

OpenVLA (Kim et al., 2024) fine-tunes a 7B-parameter Llama-2-based VLM (Prismatic VLM backbone) on the Open X-Embodiment dataset (Open X-Embodiment Collaboration, 2024), producing a 7-degree-of-freedom (DoF) end-effector delta as a sequence of discretized action tokens. SmolVLA (LeRobot team, 2024) compresses this paradigm into a substantially smaller architecture built around the SmolVLM2-500M-Video-Instruct backbone (Hugging Face TB, 2024) and a flow-matching action expert head; it accepts up to three camera streams, a proprioceptive state vector, and a tokenized task description, and emits a chunked action sequence.

Both models share two assumptions that make zero-shot transfer to new embodiments fragile: (i) the action space is defined relative to the *training-set robot* (typically a WidowX or Franka-class arm), and (ii) the visual distribution is conditioned on the embodiments, cameras, and lighting of the training data. Cross-embodiment generalization therefore typically requires at least light fine-tuning on the target arm.

### 2.2 The LeRobot Project and SO-100 / SO-101

LeRobot (Cadène et al., 2024) is an open-source Python framework that bundles policy implementations (ACT, Diffusion, SmolVLA, π0, etc.), datasets, drivers for several low-cost arms, and a teleoperation/calibration toolchain. The SO-100 and its successor SO-101 are 6-DoF (5 arm joints + 1 gripper) 3D-printable follower arms designed by the LeRobot community as accessible hardware for imitation-learning research, controlled over a USB serial bus.

### 2.3 Isaac Sim and Isaac Lab

NVIDIA Isaac Sim 5.1 is a USD-based GPU-accelerated simulator built on Omniverse, and Isaac Lab (Mittal et al., 2023) is a manager-based RL environment layer on top of it. The combination supports parallel environment instantiation, tiled cameras for differentiable rendering, and PhysX 5 articulations. This work uses Isaac Sim's standalone Python entry point (Python 3.11) and treats the simulator as a Gymnasium environment via a custom wrapper.

### 2.4 World Models

A world model (Ha and Schmidhuber, 2018) is a learned generative model of observations and rewards conditioned on actions. Recent work — Dreamer-V3 (Hafner et al., 2023), DayDreamer (Wu et al., 2022), GR00T-N1 (NVIDIA, 2024), 1X World Model — has shown that learned dynamics in pixel or latent space can be used both for model-based control and for *imagination-augmented* fine-tuning of model-free policies. The intended next phase of this project is to fit such a model on data collected from the Isaac Sim stack described below and to compare model-based and model-free fine-tuning of SmolVLA on the SO-100 cube-lifting task.

---

## 3. System Architecture

### 3.1 Hardware

The target platform is the **LeRobot SO-100 follower** (and the mechanically compatible SO-101). It is a 6-DoF articulation comprising five arm joints — `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` — and a parallel-jaw `gripper`, with joint limits given by

$$\mathbf{q}_{\text{low}} = [-2.0,\, 0.0,\, -\pi,\, -2.5,\, -\pi]\text{ rad}, \quad \mathbf{q}_{\text{high}} = [+2.0,\, +3.5,\, 0.0,\, +1.2,\, +\pi]\text{ rad}$$

(see `sim/isaac/so100_gym_wrapper.py:64–65`). The gripper sim range is `[-0.1, +1.5]` rad, mapped on hardware to a 0–100% duty cycle on the gripper servo. A single RGB camera (640 × 480, OpenCV or RealSense backend) supplies the exteroceptive observation.

On hardware, the arm is driven through the LeRobot `SO101Follower` driver over USB-CDC (`/dev/ttyACM0`) at a 5 Hz control rate, with a `max_relative_target` safety clamp of 10° per step. Calibration is handled by LeRobot's interactive flow; the calibration file is stored under `~/.cache/huggingface/lerobot/`.

### 3.2 Simulator Stack

The simulator stack is layered as:

| Layer | File(s) | Responsibility |
|---|---|---|
| Scene assets | `assets/so100/so100.urdf`, `so100.usd` | 6-link articulation, meshes, joint properties |
| Standalone scene | `sim/isaac/standalone_so100.py` | Single-process Isaac Sim app: URDF import, table+cube spawn, camera rig, closed-loop VLA control |
| Gym environment | `sim/isaac/so100_gym_wrapper.py` | `gym.Env` interface: `reset`, `step`, observation/action spaces, dense reward |
| Isaac Lab env | `sim/isaac/lab_env/` | Manager-based RL environment for Franka (preliminary; not the focus of this report) |
| Scene config | `configs/sim/so100_standalone.yaml` | Physics dt, decimation, robot/table/cube placement, camera intrinsics & extrinsics, cube spawn rectangle |

Key simulation parameters: physics is integrated at **60 Hz** (`dt = 1/60`), and the control loop runs at **5 Hz** through a control decimation of 12, matching the real-robot rate. Each control step the policy reads a (3, H, W) RGB image and a (6,) state vector, emits a (6,) action in $[-1, 1]^6$, and the wrapper applies `action[:5] * 0.1` as joint-position deltas (clamped to limits) plus a binary gripper command via threshold on `action[5]`. The cube is randomized over a table-local rectangle at each `reset()`, and the dense reward is

$$r_t = w_{\text{lift}} \cdot \max(0,\, z_{\text{cube},t} - z_{\text{rest}}) + b_{\text{succ}}\cdot \mathbb{1}[z_{\text{cube},t} > z_{\text{lift}}],$$

with $w_{\text{lift}} = b_{\text{succ}} = 10$ and $z_{\text{lift}} = 0.48$ m (see `train_smolvla_rl.py:258–265`).

### 3.3 Camera Pose Authoring

A practical pain point of the integration was matching the camera pose used at policy inference to the pose actually seen by the operator in the Isaac Sim viewport. The Gym wrapper originally computed a `look_at` quaternion from `(eye, target)` and ignored an explicit orientation override; this produced subtly wrong framings (rolled image, off-axis) for top-down views. The wrapper now reads `orientation_euler_xyz_deg` from YAML (matching the USD `Orient` widget convention) with fallback precedence `quaternion > Euler XYZ > look_at`, so the pose authored in the viewport's *Property* panel is the single source of truth shared by both the standalone runner (`standalone_so100.py`) and the training script (`train_smolvla_rl.py`).

### 3.4 Real-Robot Stack

The real-robot runner is `vla_world_model_control/robot/run_smolvla.py`. It mirrors the simulation observation–action protocol: the camera is read at the LeRobot frame rate, the proprioceptive state is converted from degrees to radians and concatenated with a sim-units gripper value, the SmolVLA policy is run on GPU, and the resulting (6,) action is mapped to joint position targets via the same `0.1` scale and gripper threshold rule as the simulator. A `--dry_run` flag short-circuits the actual hardware command for smoke testing, and a per-step JSONL trace (state, image summary, policy action, robot action, dry-run flag) is persisted under `outputs/run_smolvla/<timestamp>/` for offline analysis.

---

## 4. Policy Stack and Experimental Threads

Three policy threads were explored. They share the simulator stack of §3 and differ only in the policy module and the training/evaluation loop.

### 4.1 Thread A — OpenVLA Zero-Shot Inference (`shared/openvla_wrapper.py`, `standalone_so100.py`)

OpenVLA-7B was loaded in 4-bit NF4 quantization via BitsAndBytes (~4 GB GPU footprint) and queried at each control step with the camera image and the prompt template `"In: {instruction}\nOut:"`. The 7-DoF action was un-normalized using the `bridge_orig` key (the WidowX / BridgeData V2 statistics distributed with the OpenVLA checkpoint) and mapped to a 5-DoF arm delta + 1-DoF gripper command. Substantial engineering was required to load the model inside Isaac Sim's bundled Python (transformers version mismatch with OpenVLA's `auto_map`-distributed Prismatic processor; `tokenization_utils` symbol patching; forcing eager attention; injecting `GenerationMixin`; rewriting `tie_weights` to accept the newer `recompute_mapping` kwarg — see `openvla_wrapper.py:69–140`).

### 4.2 Thread B — SmolVLA Online RL Fine-Tuning (`vla_world_model_control/sim/train_smolvla_rl.py`)

The principal experimental thread. The SmolVLA-base checkpoint is loaded with a partial-freeze policy in which the vision encoder and the bulk of the VLM are frozen, and only the expert LM action head, the state projection, the action-in / action-out linear layers, and the time-MLPs are trainable (`freeze_for_partial_finetune`, lines 197–212). The reward-weighted regression loop (Peng et al., 2019; Peters and Schaal, 2007) proceeds as:

1. Run an episode (≤ 200 steps) in `IsaacSO100Env`, sampling actions $a_t = \pi_\theta(o_t) + \epsilon_t$ with Gaussian exploration noise of standard deviation 0.1 on the arm dimensions and a 5 % Bernoulli flip on the gripper bit;
2. Record `(state, image, action, dense_reward)` transitions into a replay buffer of capacity 20 000;
3. Every five episodes, draw 10 minibatches of size 4 and compute per-sample loss with importance weights $w_i \propto \exp(\beta \cdot \tilde{R}_i)$, where $\tilde{R}_i$ is the within-batch standardized return and $\beta = 2.0$; the weights are clipped at 20 and normalized;
4. Apply a single AdamW step (lr = 1e-4, grad clip 5.0) per minibatch.

Checkpoints are saved every five episodes under `outputs/smolvla_rl/<timestamp>/checkpoints/`, and a per-step JSONL trace is written for all rollouts.

### 4.3 Thread C — Real-Robot Deployment on SO-101 (`robot/run_smolvla.py`)

A SmolVLA checkpoint (initially the base checkpoint `lerobot/smolvla_base`) was deployed on a physical SO-101 follower with the instruction `"pick up the red cube"`. The camera was an OpenCV USB camera and the arm was driven over a single serial connection. The first runs were performed in `--dry_run` mode to verify the observation pipeline, after which the gating flag was removed to issue actual joint commands.

---

## 5. Results and Observations

Three concrete findings emerged from the threads above. Two are negative and inform the future direction described in §6.

### 5.1 OpenVLA Zero-Shot Output Is Degenerate on the SO-100 Embodiment

When OpenVLA was queried in closed loop on the simulated SO-100 cube-lifting scene, every step produced a *bit-exact identical* 7-DoF action vector across 80 consecutive frames (recorded under `outputs/standalone_so100/...`), and the binary gripper bit never crossed its threshold despite instructions such as `"open the claw"`. The action and image-hash logging added during debugging confirmed that (i) the camera was producing distinct frames at every step and (ii) OpenVLA was being re-invoked with the up-to-date image and instruction; the model itself collapses to a constant output under this scene/instruction/embodiment distribution. We attribute this to severe out-of-distribution behavior of the BridgeData-trained action head with respect to the SO-100 visual and embodiment statistics, compounded by the small (224 × 224) input resolution. This motivated the switch to SmolVLA, which we expected to be more robust under the partial fine-tuning regime described in §4.2.

### 5.2 SmolVLA RL Fine-Tuning Failed to Converge in Practice

In Thread B, several training runs were attempted (nine timestamped runs in `outputs/smolvla_rl/` between 2026-05-14 11:34 and 12:14). None produced a success rate measurably above zero by the end of training. The fundamental difficulty was the joint quality of the *rollout data*: with random + light-noise exploration starting from a SmolVLA base policy that itself acts approximately constantly on this scene (a milder version of the OpenVLA pathology), almost no episodes ever lifted the cube, so the reward-weighted objective received vanishingly little high-return signal. Visual inspection of step traces confirmed that within ~30–50 control steps the arm saturated against its joint limits (clamping at `[-2.0, +3.5, -π, -2.5, +π]`), and from then on the policy was effectively asking for the maximum delta in every dimension at every step. The combination of (i) a bootstrap policy that does not solve the task even occasionally and (ii) a sparse-ish dense reward that requires actually lifting the cube produced an exploration ceiling that the present training loop cannot break through.

### 5.3 SmolVLA on the Real SO-101 Was Not Useful Either

The real-robot deployment (Thread C) confirmed the simulation observation. The base SmolVLA checkpoint produced motion on the hardware, but the motion was not goal-directed: the arm drifted toward joint limits in patterns broadly similar to those seen in simulation, and no attempt at the `"pick up the red cube"` task succeeded. This was an important sanity check: the *infrastructure* — camera capture, calibration, observation packing, action issuing, the serial-bus safety clamp, and post-run trace logging — all worked correctly, but the *policy* did not transfer. Because the base SmolVLA checkpoint is trained predominantly on demonstrations on other arms, this is consistent with the expectations of §2.1, but it also rules out the trivially optimistic scenario in which the base model would "mostly work" and only needed sim fine-tuning for the last 10 %.

### 5.4 Engineering Observations

Independent of the policy results, the project produced a number of replicable engineering artifacts: a single YAML-driven SO-100 scene configuration consumed by every entry point, a robust URDF→Isaac-Sim import that authors the root pose explicitly to avoid the silent zero-offset Stage-UI bug, a vendored LeRobot checkout that loads cleanly inside Isaac Sim's bundled Python 3.11 (a system-wide LeRobot is broken by an unrelated dataclass-ordering bug in the GR00T policy), and a per-step JSONL + PNG trace format shared between the simulator and the hardware runner. Several non-obvious failure modes were discovered and fixed during this work: OpenVLA's expectation of `transformers==4.40.1` symbols inside a newer transformers bundled by Isaac Sim; SmolVLA's expectation of the three named image keys `camera1`/`camera2`/`camera3` in the base checkpoint (rather than the `front` key used in some fine-tuned variants); the dotted vs. underscored language-token key (`observation.language.tokens`); and a benign-but-noisy Isaac Sim 5.1 atexit-time segfault inside `libomni.graph.core` that masked real Python tracebacks until forced `os._exit(0)` was added after `world.stop()`.

---

## 6. Discussion and Future Work

The most honest summary of the present state of the project is: **the *infrastructure* for VLA evaluation and online fine-tuning on the SO-100/SO-101 is in place and validated end-to-end, but neither of the two policies actually evaluated solves the cube-lift task in either simulation or hardware, and the present reward-weighted RL fine-tuning loop does not appear able to bootstrap a useful policy from the SmolVLA base on this embodiment.** Three directions are planned for the next phase.

### 6.1 Better Bootstrap Data Before RL

The empirical observation of §5.2 is that reward-weighted regression without high-return demonstrations is futile. The plan is therefore to invert the order of operations: first collect a small (~20–50 episode) imitation dataset of successful cube lifts in Isaac Sim — by scripted IK trajectories, by teleoperation through the LeRobot keyboard/gamepad interface, or by a heuristic gripper-above-cube controller — then warm-start SmolVLA with behavioral cloning on this dataset, and only then run the reward-weighted online phase as a refinement loop. The DAGGER-style collector and the HF-dataset offline training script already in the repository (`scripts/train_dagger_collect.py`, `scripts/train_hf_so_dataset.py`) provide the scaffolding.

### 6.2 Sim-to-Real Hardening Before Hardware Re-Deployment

The negative real-robot result of §5.3 was *expected* — the base SmolVLA checkpoint has no reason to generalize to this arm and scene — but it also exposed that the simulation visual distribution is currently far from the camera the real arm sees: lighting, background, focal length, and the precise extrinsic pose of the camera in the simulated cell were not deliberately matched to the hardware. Future work will (i) measure the actual real-robot camera intrinsics and extrinsics and reproduce them in the YAML scene config, and (ii) add basic domain randomization (lighting, table texture, cube color jitter) to the simulator.

### 6.3 Toward a Learned World Model

The longer-term direction — which gives the repository its name — is to train a learned world model on the simulator rollouts collected during §6.1–6.2 and then to compare:

1. **Model-free**: continue the SmolVLA reward-weighted fine-tuning loop on real environment rollouts;
2. **Model-based (imagination)**: roll out the policy *inside* the learned world model (à la Dreamer-V3) and compute the policy-gradient update on those imagined trajectories, only periodically re-grounding against the simulator;
3. **Model-based (planning)**: at inference time, use MPC over short horizons in the world model to refine the action proposed by SmolVLA, rather than fine-tuning the policy directly.

A concrete deliverable in this phase will be quantitative comparison of these three regimes on success rate and sample efficiency at the SO-100 cube-lift task, with the same simulator and the same final checkpoint format that can be deployed on the SO-101 hardware via the existing `run_smolvla.py` runner. The infrastructure described in §3 was deliberately built to make this comparison a matter of swapping the update rule, not the surrounding plumbing.

### 6.4 Architectural Variants

Once a working policy exists for the cube-lift task, two further directions are natural: (i) compare different VLA backbones (SmolVLA, π0, OpenVLA-LoRA, ACT trained on the same dataset) on the same evaluation harness, and (ii) compare action representations (joint deltas vs. delta-pose IK vs. action chunking with diffusion expert heads). The harness already supports the first comparison through LeRobot's policy registry; the second will require small changes to the action-application stage of `so100_gym_wrapper.py`.

---

## 7. Conclusion

This independent study delivered a complete vertical slice of infrastructure for evaluating VLA policies on the SO-100/SO-101 platform — from URDF-driven Isaac Sim scene construction through per-step trace logging to a real-robot LeRobot deployment path — and used that infrastructure to evaluate two contemporary VLAs (OpenVLA and SmolVLA) on a simple cube-lifting task. Both policies failed at the task in their zero-shot or lightly-fine-tuned form on this embodiment, and a reward-weighted online RL fine-tuning loop did not bootstrap a useful policy in the absence of demonstration data. These negative results — together with the engineering substrate that made them reliably reproducible — are the foundation for the planned model-based extension that motivates the project's name.

---

## Code Availability

The full source — entry-point scripts (`scripts/`), the project package (`vla_world_model_control/`), the Isaac Sim scene and Gym wrapper (`sim/isaac/`), the SO-100 assets, the YAML scene/policy configs, and the per-step JSONL/PNG traces from the runs reported in §5 — is available at:

**https://github.com/seokw-Ryan/vla-world-model-control**

The repository [`README.md`](README.md) at the root contains the **exact reproduction commands** for every result discussed in this report, organized into a *Quick Start — Simulation* section (scene viewer, closed-loop VLA control, online RL fine-tuning) and a *Quick Start — Real Hardware* section (dry-run and live execution on the SO-101 follower). This document (`report.md`) is also committed at the repository root for future reference.

---

## References

Cadène, R., Soare, A. & the LeRobot team (2024). *LeRobot: State-of-the-art AI for real-world robotics in PyTorch.* https://github.com/huggingface/lerobot

Ha, D., & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122.

Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models (Dreamer-V3).* arXiv:2301.04104.

Hugging Face TB Research (2024). *SmolVLM and SmolVLM2: Compact Vision-Language Models.* https://huggingface.co/HuggingFaceTB

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., & Finn, C. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246.

LeRobot Team (2024). *SmolVLA: A Compact Vision-Language-Action Model.* https://huggingface.co/lerobot/smolvla_base

Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., Yuan, J. L., Singh, R., Guo, Y., Mazhar, H., Mandlekar, A., Babich, B., Birchfield, S., Hutter, M., & Garg, A. (2023). *Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments (now Isaac Lab).* IEEE Robotics and Automation Letters, 8(6), 3740–3747.

Open X-Embodiment Collaboration (2024). *Open X-Embodiment: Robotic Learning Datasets and RT-X Models.* arXiv:2310.08864.

Peng, X. B., Kumar, A., Zhang, G., & Levine, S. (2019). *Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning.* arXiv:1910.00177.

Peters, J., & Schaal, S. (2007). *Reinforcement Learning by Reward-Weighted Regression for Operational Space Control.* ICML.

Wu, P., Escontrela, A., Hafner, D., Goldberg, K., & Abbeel, P. (2022). *DayDreamer: World Models for Physical Robot Learning.* CoRL.
