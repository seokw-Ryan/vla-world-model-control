# SO100 Training On Local GPU

This repository currently supports two different workflows for the SO100:

1. `Isaac Lab + RL` for direct PPO-style training in simulation.
2. `LeRobot + offline VLA training` for ACT / SmolVLA / X-VLA style policy training from demonstrations.

These are not the same thing. If you want a real VLA policy, use the offline LeRobot path. If you want online training inside Isaac Lab today, use PPO.

## What Works In This Repo Right Now

- The SO100 Isaac Lab Arena environment starts locally:
  - `scripts/test_arena_env.py --headless --num_envs 1 --num_steps 5`
- The repository already contains:
  - SO100 robot asset and Arena embodiment
  - Isaac Lab Arena environment for SO100 pick-and-place
  - Isaac Sim standalone SO100 data collection and evaluation scripts
  - LeRobot-compatible dataset collection for SO100

## Local Machine Notes

- GPU: `NVIDIA GeForce RTX 5060 Ti` with `16 GB` VRAM
- Isaac Sim Python: `3.11`
- Repo `.venv` Python: `3.12`

This version split matters:

- Use `Isaac Sim python.sh` for simulation, collection, and Arena smoke tests.
- Use the repo `.venv` for modern LeRobot training (`lerobot` in this repo requires Python 3.12).

## Recommended Path For A VLA Policy

Use this path if your goal is: "train a VLA-like policy for SO100 and run it back in sim."

### 1. Isaac Sim / Isaac Lab side

```bash
export ISAACSIM=~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64

$ISAACSIM/python.sh -m pip install -r requirements-lerobot.txt
$ISAACSIM/python.sh -m pip install 'qpsolvers[open_source_solvers]'
```

Smoke test the SO100 Arena env:

```bash
$ISAACSIM/python.sh scripts/test_arena_env.py --headless --num_envs 1 --num_steps 5
```

Optional GUI check:

```bash
$ISAACSIM/python.sh scripts/view_so100.py
```

### 2. Collect SO100 demonstrations

Current collection is implemented with the standalone SO100 environment, not the Arena env:

```bash
$ISAACSIM/python.sh scripts/collect_so100_data.py \
  --repo_id yourname/so100_pick_cube \
  --num_episodes 100 \
  --max_steps 200 \
  --headless
```

The dataset will be written under:

```bash
datasets/yourname_so100_pick_cube
```

### 3. Train the policy in `.venv`

Activate the Python 3.12 environment:

```bash
source .venv/bin/activate
pip install -e ./lerobot[xvla]
```

For a smaller first run on a 16 GB GPU, start with `xvla` rather than OpenVLA 7B.

Example:

```bash
lerobot-train \
  --dataset.repo_id=yourname/so100_pick_cube \
  --dataset.root=./datasets/yourname_so100_pick_cube \
  --output_dir=./outputs/xvla_so100 \
  --job_name=xvla_so100 \
  --policy.path=lerobot/xvla-base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.action_mode=auto \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --batch_size=2 \
  --steps=5000 \
  --save_freq=1000 \
  --eval_freq=0
```

Notes:

- `--batch_size=2` is the safe starting point on a 16 GB GPU.
- Increase to `4` only after checking memory usage.
- `policy.action_mode=auto` is the right choice for SO100's 6D joint action dataset.

### 4. Evaluate

For ACT-style checkpoints, the repo already has `scripts/eval_so100_policy.py`.

For XVLA / SmolVLA, this repo does not yet include a dedicated SO100 Isaac Sim evaluation bridge. You will need either:

- a small evaluation script that loads the LeRobot policy and maps its outputs into `IsaacSO100Env`, or
- an IsaacLab-Arena evaluation path built around LeRobot's `isaaclab_arena` env support.

## If You Want Online Isaac Lab Training Instead

Use PPO:

```bash
$ISAACSIM/python.sh scripts/train_rl.py \
  --headless \
  --num_envs 4 \
  --total_timesteps 500000
```

This is the path that actually trains *inside Isaac Lab* today, but it is not a VLA.

## What Is Not Ready Yet

- There is no finished "LeRobot VLA training directly inside Isaac Lab" pipeline in this repo.
- `isaaclab_arena_vla/policies/openvla_policy.py` is an inference wrapper, not a full trainer.
- `scripts/train_rl_lora.py` is experimental and not the best first path on a 16 GB local GPU.

## Practical Recommendation

For this machine, the highest-probability path is:

1. Collect SO100 demos in Isaac Sim.
2. Train `XVLA` in `.venv` on the RTX 5060 Ti.
3. Add a small SO100 evaluation bridge for XVLA.

If you only want something end-to-end today with minimal extra work, use:

1. SO100 demo collection
2. `ACT` training
3. `scripts/eval_so100_policy.py`
