"""Collect demonstration data from Isaac Sim into LeRobot dataset format.

Runs scripted pick-and-place demonstrations and saves them as a
LeRobotDataset v3 (Parquet + MP4), ready for training with LeRobot.

Usage:
    <ISAAC_SIM>/python.sh scripts/collect_lerobot_data.py \
        --repo_id user/franka_pick_cube \
        --num_episodes 50 \
        --headless

IMPORTANT: Must be run with Isaac Sim's python.sh.
"""

from __future__ import annotations

import argparse
import sys
import os

# ─── Parse args before any Isaac Sim imports ──────────────────────────────

parser = argparse.ArgumentParser(description="Collect LeRobot demonstrations")
parser.add_argument("--repo_id", type=str, required=True,
                    help="HuggingFace dataset repo ID (e.g. user/franka_pick_cube)")
parser.add_argument("--root", type=str, default="./datasets",
                    help="Local dataset root directory")
parser.add_argument("--num_episodes", type=int, default=50,
                    help="Number of demonstration episodes to collect")
parser.add_argument("--max_steps", type=int, default=200,
                    help="Max steps per episode")
parser.add_argument("--fps", type=int, default=5,
                    help="Control frequency (should match sim control rate)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--push_to_hub", action="store_true",
                    help="Push dataset to HuggingFace Hub after collection")
parser.add_argument("--sim_config", type=str,
                    default="configs/sim/isaac_standalone.yaml")
parser.add_argument("--task", type=str, default="pick up the red cube",
                    help="Task description for the dataset")
args = parser.parse_args()


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# ─── Create SimulationApp ────────────────────────────────────────────────

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# ─── Now import everything else ──────────────────────────────────────────

from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sim.isaac.gym_wrapper import IsaacFrankaEnv

# ─── Scripted demonstration policy ──────────────────────────────────────


class ScriptedPickPolicy:
    """Simple scripted policy that moves to cube, closes gripper, lifts.

    Phases:
        0: Move above cube
        1: Lower to grasp height
        2: Close gripper
        3: Lift
    """

    def __init__(self):
        self._phase = 0
        self._grasp_steps = 0

    def reset(self):
        self._phase = 0
        self._grasp_steps = 0

    def act(self, obs: dict, info: dict) -> np.ndarray:
        """Return (7,) action: [dx, dy, dz, drx, dry, drz, gripper]."""
        ee_pos = info.get("ee_pos", np.zeros(3))
        cube_pos = info.get("cube_pos", np.array([0.5, 0.0, 0.525]))

        action = np.zeros(7, dtype=np.float32)
        # Keep gripper open during approach
        action[6] = 1.0

        if self._phase == 0:
            # Move above cube
            target = cube_pos.copy()
            target[2] += 0.10  # 10cm above
            delta = target - ee_pos
            if np.linalg.norm(delta) < 0.02:
                self._phase = 1
            action[:3] = np.clip(delta * 3.0, -1.0, 1.0)

        elif self._phase == 1:
            # Lower to grasp
            target = cube_pos.copy()
            target[2] += 0.02  # just above cube
            delta = target - ee_pos
            if np.linalg.norm(delta) < 0.015:
                self._phase = 2
                self._grasp_steps = 0
            action[:3] = np.clip(delta * 3.0, -1.0, 1.0)

        elif self._phase == 2:
            # Close gripper
            action[6] = -1.0  # close
            self._grasp_steps += 1
            if self._grasp_steps > 10:
                self._phase = 3

        elif self._phase == 3:
            # Lift
            action[2] = 0.8  # move up
            action[6] = -1.0  # keep closed

        return action


# ─── Build dataset ───────────────────────────────────────────────────────

from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_H, IMAGE_W = 256, 256

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "joint_0", "joint_1", "joint_2", "joint_3",
                "joint_4", "joint_5", "joint_6",
            ]
        },
    },
    "observation.images.front": {
        "dtype": "image",
        "shape": (3, IMAGE_H, IMAGE_W),
        "names": None,
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "delta_x", "delta_y", "delta_z",
                "delta_rx", "delta_ry", "delta_rz",
                "gripper",
            ]
        },
    },
}

dataset_root = Path(args.root) / args.repo_id.replace("/", "_")
dataset = LeRobotDataset.create(
    repo_id=args.repo_id,
    root=dataset_root,
    features=features,
    fps=args.fps,
)

# ─── Create env ──────────────────────────────────────────────────────────

env = IsaacFrankaEnv(
    simulation_app=simulation_app,
    sim_config_path=args.sim_config,
    image_height=IMAGE_H,
    image_width=IMAGE_W,
    max_episode_steps=args.max_steps,
    cube_randomize=True,
)

policy = ScriptedPickPolicy()

# ─── Collection loop ─────────────────────────────────────────────────────

log(f"[Collect] Starting collection of {args.num_episodes} episodes...")

successful_episodes = 0
total_episodes = 0

for ep in range(args.num_episodes):
    obs, info = env.reset()
    policy.reset()
    total_episodes += 1

    episode_success = False

    for t in range(args.max_steps):
        action = policy.act(obs, info)

        # Save frame to dataset
        frame = {
            "observation.state": obs["observation.state"],
            "observation.images.front": obs["observation.images.front"],
            "action": action,
            "task": args.task,
        }
        dataset.add_frame(frame)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            episode_success = True
            break
        if truncated:
            break

    dataset.save_episode()

    if episode_success:
        successful_episodes += 1

    log(f"[Collect] Episode {ep + 1}/{args.num_episodes} — "
        f"{'SUCCESS' if episode_success else 'FAIL'} — "
        f"steps: {t + 1} — success rate: {successful_episodes}/{total_episodes}")

# ─── Finalize ────────────────────────────────────────────────────────────

dataset.finalize()
log(f"[Collect] Dataset saved to {dataset_root}")
log(f"[Collect] Total: {successful_episodes}/{total_episodes} successful episodes")

if args.push_to_hub:
    log("[Collect] Pushing to HuggingFace Hub...")
    dataset.push_to_hub()
    log(f"[Collect] Pushed to https://huggingface.co/datasets/{args.repo_id}")

# ─── Cleanup ─────────────────────────────────────────────────────────────

env.close()
simulation_app.close()
log("[Collect] Done.")
