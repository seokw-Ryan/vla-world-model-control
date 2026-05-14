# SO100 SmolVLA + RL Plan

## Goal

Train a policy for the SO100 to pick up the red cube, with a workflow that:

1. starts from VLA-style behavior,
2. improves through reward-driven learning in simulation,
3. runs on the local RTX 5060 Ti 16 GB GPU.

## Decision

Use **SmolVLA**, not XVLA.

### Why SmolVLA

- The target is a **single embodiment**: SO100.
- The target is a **single task family**: pick up the red cube.
- We want a model that is more realistic to fine-tune locally with RL.
- SmolVLA is a better fit for **consumer GPU** iteration than XVLA for this project.

### Why Not XVLA

- XVLA is strongest when cross-embodiment transfer matters.
- That is not the main requirement here.
- For local RL fine-tuning, a smaller policy is the more pragmatic choice.

## What Exists Already

### In This Repo

- SO100 Isaac Sim standalone environment:
  - `sim/isaac/so100_gym_wrapper.py`
- SO100 Isaac Lab Arena environment:
  - `isaaclab_arena_vla/environments/so100_pick_and_place.py`
- SO100 standalone data collection:
  - `scripts/collect_so100_data.py`
- SO100 policy evaluation for LeRobot-style checkpoints:
  - `scripts/eval_so100_policy.py`
- Existing VLA RL prototype:
  - `scripts/train_rl_lora.py`
- Isaac Lab PPO baseline:
  - `scripts/train_rl.py`

### Verified Locally

- The SO100 Isaac Lab Arena env starts and steps successfully with:
  - `scripts/test_arena_env.py`
- Local GPU:
  - `NVIDIA GeForce RTX 5060 Ti`
  - `16 GB VRAM`

## Main Constraint

There is a Python version split:

- Isaac Sim / Isaac Lab runtime here is **Python 3.11**
- Vendored LeRobot in this repo is **Python 3.12-oriented**

This is the current blocker for directly importing the full modern LeRobot SmolVLA stack inside Isaac Sim.

Confirmed issue:

- importing vendored LeRobot SmolVLA under Isaac Sim 3.11 fails on Python 3.12-only syntax
- example:
  - `lerobot/src/lerobot/datasets/streaming_dataset.py`
  - `class Backtrackable[T]:`

## Intended Learning Loop

The desired loop is:

1. initialize from a pretrained SmolVLA checkpoint,
2. run episodes in simulation,
3. observe failure/success on red cube pickup,
4. compute reward,
5. update only a small trainable subset of the policy,
6. repeat.

This is **not PPO**.

This should be **reward fine-tuning of SmolVLA**, ideally with PEFT / LoRA-style updates rather than full-model training.

## Recommended Architecture

### Phase 1. Make SmolVLA Work For SO100 Offline

Objective:

- prove that SmolVLA can consume SO100 observations and emit usable SO100 actions

Tasks:

1. collect a small SO100 dataset with:
   - `scripts/collect_so100_data.py`
2. install SmolVLA dependencies in the Python 3.12 `.venv`
3. fine-tune SmolVLA offline first
4. verify inference on SO100 observations outside Isaac Sim if needed

Reason:

- this de-risks the action mapping and preprocessing before adding RL complexity

### Phase 2. Bridge SmolVLA Into Simulation

Objective:

- run SmolVLA inside the sim loop for SO100 episodes

Options:

1. **Patch vendored LeRobot for Python 3.11 compatibility**
   - best if we want direct Isaac Sim integration
2. **Split-process architecture**
   - Isaac Sim runs env
   - Python 3.12 process runs SmolVLA
   - communicate over IPC / socket / local RPC
   - safer if we want to avoid patching large parts of LeRobot immediately

Recommended first:

- start with **split-process** if compatibility patching becomes too wide
- start with **local patching** if the incompatible syntax surface is small enough

### Phase 3. Add Reward Fine-Tuning

Objective:

- update SmolVLA weights from reward during simulated rollouts

Approach:

1. load pretrained SmolVLA
2. wrap with PEFT adapters
3. freeze base weights
4. run episodes in SO100 env
5. compute trajectory return
6. apply policy-gradient-style update to adapter parameters only

Initial algorithm:

- simple REINFORCE-style loop
- optional baseline / EMA for variance reduction

Reason:

- easiest path to get failure-to-learning behavior without introducing PPO

### Phase 4. Improve Stability

After the first end-to-end reward loop works:

1. add batch updates over multiple episodes
2. add reward normalization / baseline
3. add checkpointing and evaluation rollouts
4. add rollout videos
5. optionally mix demonstration loss with RL loss

This last point is important:

- **behavior cloning + RL fine-tuning** is likely the best practical recipe here

## Concrete Implementation Plan

### Track A. Compatibility

1. Audit vendored LeRobot for Python 3.12-only syntax.
2. Patch the minimal set needed for SmolVLA imports under Python 3.11.
3. Verify:
   - `SmolVLAPolicy` import
   - checkpoint loading
   - `select_action` on a fake SO100 batch

### Track B. SmolVLA RL Script

Create a new script, likely:

- `scripts/train_smolvla_rl.py`

Responsibilities:

1. start Isaac Sim
2. create `IsaacSO100Env`
3. load SmolVLA checkpoint
4. wrap policy with PEFT
5. preprocess camera/state/task into SmolVLA input batch
6. map SmolVLA output to SO100 env action
7. run rollouts
8. compute returns
9. update trainable adapters
10. save checkpoints

### Track C. Observation / Action Wiring

Need a clean mapping for:

- image:
  - `observation.images.front`
- state:
  - `observation.state`
- language:
  - task string like:
    - `"pick up the red cube"`
- action:
  - SmolVLA action output to SO100 6D joint command

### Track D. Evaluation

Create or extend a script to:

1. load a trained SmolVLA adapter/checkpoint
2. run evaluation episodes
3. record success rate
4. optionally render video

## Immediate Next Steps

### Step 1

Patch the vendored LeRobot compatibility issues that block SmolVLA import under Isaac Sim Python 3.11.

### Step 2

Create `scripts/train_smolvla_rl.py` by adapting the existing structure of:

- `scripts/train_rl_lora.py`

but replacing OpenVLA-specific logic with SmolVLA policy loading and PEFT-based updates.

### Step 3

Run a smoke test:

- 1 environment
- 5 to 10 episodes
- very small update loop
- checkpoint save

### Step 4

Scale up:

- 20 to 50 episodes
- reward plots
- rollout video

## Important Notes

- We are **not using PPO**.
- We are aiming for **SmolVLA + reward fine-tuning**.
- The most likely best final recipe is:
  - **demo pretraining / fine-tuning first**
  - then **RL improvement in sim**

## Practical Recommendation

Do this in order:

1. patch SmolVLA compatibility under Isaac Sim Python 3.11
2. wire `train_smolvla_rl.py`
3. make the arm fail/succeed in simulation with reward updates
4. add mixed imitation + RL if pure reward tuning is too unstable

## Current Status

Planning document written.

Next coding task should be:

- **make SmolVLA importable in the Isaac Sim runtime and then wire the RL script**
