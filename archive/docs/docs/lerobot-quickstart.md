# LeRobot + Isaac Sim — Quickstart Guide

End-to-end pipeline: collect demonstrations in Isaac Sim, train an ACT policy with LeRobot, evaluate in sim.

## Prerequisites

- Isaac Sim 5.1.0 installed at `~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/`
- NVIDIA GPU (tested on RTX 5060 Ti 16GB)

Set this alias to save typing:

```bash
export ISAACSIM=~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64
```

## Step 1: Install Dependencies

```bash
$ISAACSIM/python.sh -m pip install -r requirements-lerobot.txt
```

> **Important:** lerobot pulls in numpy 2.x which breaks Isaac Sim. If that happens, downgrade:
> ```bash
> $ISAACSIM/python.sh -m pip install "numpy<2.0"
> ```

## Step 2: Collect Demonstrations

Runs a scripted pick-and-place policy in Isaac Sim and saves episodes in LeRobot dataset format (Parquet + images).

```bash
$ISAACSIM/python.sh scripts/collect_lerobot_data.py \
    --repo_id myuser/franka_pick_cube \
    --num_episodes 50 \
    --max_steps 200 \
    --headless
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--repo_id` | *(required)* | Dataset name (e.g. `myuser/franka_pick_cube`) |
| `--root` | `./datasets` | Local directory to save dataset |
| `--num_episodes` | `50` | Number of demo episodes to collect |
| `--max_steps` | `200` | Max steps per episode |
| `--fps` | `5` | Control frequency (matches sim's 5Hz default) |
| `--headless` | off | Run without GUI (faster) |
| `--push_to_hub` | off | Upload dataset to HuggingFace Hub when done |
| `--task` | `"pick up the red cube"` | Natural language task description |
| `--sim_config` | `configs/sim/isaac_standalone.yaml` | Scene config path |

**Output:** Dataset saved to `./datasets/myuser_franka_pick_cube/`

To verify the dataset:

```bash
$ISAACSIM/python.sh -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('myuser/franka_pick_cube', root='./datasets/myuser_franka_pick_cube')
print(f'{len(ds)} frames, {ds.fps} FPS, features: {list(ds.features.keys())}')
"
```

## Step 3: Train ACT Policy

Training uses LeRobot's CLI and does **not** need Isaac Sim — run with regular Python.

```bash
pip install lerobot  # if not already in your base env

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=myuser/franka_pick_cube \
    --dataset.root=./datasets/myuser_franka_pick_cube \
    --output_dir=outputs/train/act_franka
```

**Key training parameters** (see `configs/lerobot/act_franka.yaml` for full reference):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--policy.type` | `act` | Policy architecture (also: `diffusion`, `vqbet`) |
| `--training.batch_size` | `8` | Batch size (lower if OOM) |
| `--training.steps` | `100000` | Total training steps |
| `--training.lr` | `1e-5` | Learning rate |
| `--training.save_freq` | `10000` | Checkpoint save interval |
| `--training.eval_freq` | `10000` | Evaluation interval |

**Monitor with W&B** (optional):

```bash
wandb login
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=myuser/franka_pick_cube \
    --dataset.root=./datasets/myuser_franka_pick_cube \
    --output_dir=outputs/train/act_franka \
    --wandb.enable=true \
    --wandb.project=lerobot-franka
```

**Output:** Checkpoints saved to `outputs/train/act_franka/checkpoints/`

## Step 4: Evaluate in Isaac Sim

```bash
$ISAACSIM/python.sh scripts/eval_lerobot_policy.py \
    --policy_path outputs/train/act_franka/checkpoints/last/pretrained_model \
    --num_episodes 50 \
    --headless
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--policy_path` | *(required)* | Path to trained policy (local dir or HF hub ID) |
| `--num_episodes` | `50` | Number of evaluation episodes |
| `--max_steps` | `200` | Max steps per episode |
| `--headless` | off | Run without GUI |
| `--device` | `cuda` | Inference device |

**Output:** Prints per-episode results and final success rate.

To watch the robot visually, omit `--headless`:

```bash
$ISAACSIM/python.sh scripts/eval_lerobot_policy.py \
    --policy_path outputs/train/act_franka/checkpoints/last/pretrained_model \
    --num_episodes 10
```

## Project Structure

```
scripts/
  collect_lerobot_data.py    # Step 2: data collection
  eval_lerobot_policy.py     # Step 4: evaluation
sim/isaac/
  gym_wrapper.py             # Gymnasium env wrapping Isaac Sim Franka scene
  standalone_vla.py          # Original OpenVLA standalone script
configs/
  lerobot/act_franka.yaml    # ACT training config reference
  sim/isaac_standalone.yaml  # Isaac Sim scene config
datasets/                    # Collected datasets (git-ignored)
outputs/                     # Training outputs (git-ignored)
```

## Troubleshooting

**`numpy` version conflict**
lerobot requires numpy 2.x but Isaac Sim needs numpy <2. Install lerobot first, then downgrade numpy:
```bash
$ISAACSIM/python.sh -m pip install "numpy<2.0"
```
Training (Step 3) runs outside Isaac Sim, so numpy 2.x is fine there.

**Camera returns black images**
The gym wrapper warms up the camera for 20 frames on scene build. If images are still black, increase warm-up by editing `gym_wrapper.py` line ~190 (`range(20)` → `range(50)`).

**IK failures during data collection**
Some random actions will fail IK (the arm can't reach the target). This is expected — the scripted policy uses small deltas that mostly succeed. If you see many IK failures, check that the cube spawns within the robot's workspace in `configs/sim/isaac_standalone.yaml`.

**OOM during training**
Lower `--training.batch_size` to 4 or 2. ACT with ResNet-18 backbone at 256x256 uses ~8GB VRAM with batch size 8.
