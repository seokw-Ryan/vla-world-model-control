#!/usr/bin/env python3
"""Option 3: RL-guided data collection with OpenVLA for offline fine-tuning.

Runs OpenVLA in Isaac Sim (inference only, no backprop), collects episodes,
filters successful ones, and saves them as a LeRobot-compatible dataset.
Then you fine-tune OpenVLA offline on the successful demonstrations.

This is a two-phase approach:
  Phase A (this script): Collect data — run OpenVLA, keep the good episodes
  Phase B (lerobot train): Fine-tune OpenVLA on the collected dataset

Usage (run with Isaac Sim's python.sh):
    # Phase A: collect demonstrations
    <ISAAC_SIM>/python.sh scripts/train_dagger_collect.py --robot so100 --total_episodes 200

    # Phase B: fine-tune with LeRobot (after collection)
    python -m lerobot.scripts.train \
        --policy.type=vla \
        --dataset.repo_id=user/openvla_rl_demos \
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Argument parsing — BEFORE SimulationApp
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect OpenVLA demonstrations in Isaac Sim for offline fine-tuning"
    )

    # Environment
    p.add_argument("--robot", type=str, default="so100", choices=["franka", "so100"])
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--instruction", type=str, default="pick up the red cube")

    # VLA
    p.add_argument("--model_path", type=str, default="openvla/openvla-7b")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Sampling temperature for action diversity (0=greedy, higher=more random)")
    p.add_argument("--lora_path", type=str, default=None,
                   help="Path to LoRA weights from train_rl_lora.py (optional)")

    # Collection
    p.add_argument("--total_episodes", type=int, default=200)
    p.add_argument("--keep_mode", type=str, default="successful",
                   choices=["successful", "all", "top_k"],
                   help="Which episodes to keep: successful only, all, or top K%% by reward")
    p.add_argument("--top_k_pct", type=float, default=25.0,
                   help="Keep top K%% of episodes by reward (only for --keep_mode top_k)")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save the dataset")
    p.add_argument("--dataset_name", type=str, default="openvla_rl_demos",
                   help="Name for the dataset")
    p.add_argument("--fps", type=int, default=5, help="Control frequency / dataset FPS")
    p.add_argument("--push_to_hub", action="store_true", default=False)
    p.add_argument("--hub_repo_id", type=str, default=None,
                   help="HuggingFace repo ID for push (e.g. user/openvla_rl_demos)")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap
# ---------------------------------------------------------------------------

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Environment & VLA setup
# ---------------------------------------------------------------------------


def make_env(args):
    if args.robot == "franka":
        from sim.isaac.gym_wrapper import IsaacFrankaEnv
        return IsaacFrankaEnv(
            simulation_app=simulation_app,
            sim_config_path=os.path.join(PROJECT_ROOT, "configs/sim/isaac_standalone.yaml"),
            max_episode_steps=args.max_steps,
        )
    else:
        from sim.isaac.so100_gym_wrapper import IsaacSO100Env
        return IsaacSO100Env(
            simulation_app=simulation_app,
            sim_config_path=os.path.join(PROJECT_ROOT, "configs/sim/so100_standalone.yaml"),
            max_episode_steps=args.max_steps,
        )


def load_vla(args):
    """Load OpenVLA for inference (optionally with LoRA weights)."""
    from vla_world_model_control.shared import OpenVLAConfig, OpenVLAWrapper

    config = OpenVLAConfig(model_path=args.model_path)
    vla = OpenVLAWrapper(config).load()

    # Load LoRA weights if provided (from train_rl_lora.py output)
    if args.lora_path:
        log(f"[collect] Loading LoRA weights from {args.lora_path}")
        from peft import PeftModel
        vla._model = PeftModel.from_pretrained(vla._model, args.lora_path)
        log("[collect] LoRA weights loaded.")

    return vla


def vla_action_to_env_action(raw_action: np.ndarray, robot: str) -> np.ndarray:
    """Map VLA 7D output to env action space."""
    if robot == "franka":
        return np.clip(raw_action, -1.0, 1.0).astype(np.float32)
    else:
        action = np.zeros(6, dtype=np.float32)
        scale = 0.5
        action[0] = np.clip(raw_action[0] * scale, -1.0, 1.0)
        action[1] = np.clip(raw_action[1] * scale, -1.0, 1.0)
        action[2] = np.clip(raw_action[2] * scale, -1.0, 1.0)
        action[3] = np.clip(raw_action[3] * scale, -1.0, 1.0)
        action[4] = np.clip(raw_action[4] * scale, -1.0, 1.0)
        action[5] = 1.0 if raw_action[6] > 0.5 else -1.0
        return action


# ---------------------------------------------------------------------------
# Episode data structure
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Records a single episode's observations, actions, and metadata."""

    def __init__(self):
        self.images = []      # list of (3, H, W) uint8
        self.states = []      # list of (N,) float32
        self.actions = []     # list of (M,) float32
        self.rewards = []     # list of float
        self.success = False
        self.total_reward = 0.0

    def add_step(self, image, state, action, reward):
        self.images.append(image.copy())
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.total_reward += reward

    def __len__(self):
        return len(self.actions)


# ---------------------------------------------------------------------------
# Save as LeRobot dataset
# ---------------------------------------------------------------------------


def save_lerobot_dataset(episodes: list[EpisodeRecorder], args, output_dir: str):
    """Save episodes in LeRobot v2 format (Parquet + images).

    Creates the directory structure LeRobot expects:
        output_dir/
            meta/
                info.json
                episodes.jsonl
                tasks.jsonl
            data/
                chunk-000/
                    episode_000000.parquet
                    ...
            videos/  (or images/)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = os.path.join(output_dir, "data", "chunk-000")
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    state_dim = len(episodes[0].states[0])
    action_dim = len(episodes[0].actions[0])
    image_shape = episodes[0].images[0].shape  # (C, H, W)
    total_frames = sum(len(ep) for ep in episodes)

    log(f"[collect] Saving {len(episodes)} episodes ({total_frames} frames) to {output_dir}")

    # Save each episode as a parquet file
    episodes_meta = []
    frame_idx = 0

    for ep_idx, ep in enumerate(episodes):
        ep_len = len(ep)

        # Build columns
        timestamps = [i / args.fps for i in range(ep_len)]
        frame_indices = list(range(frame_idx, frame_idx + ep_len))
        ep_indices = [ep_idx] * ep_len
        task_indices = [0] * ep_len  # single task

        # State and action as lists of lists
        states = [s.tolist() for s in ep.states]
        actions = [a.tolist() for a in ep.actions]

        # Images: save as numpy files for simplicity
        # (LeRobot v2 can use various formats)
        image_paths = []
        img_dir = os.path.join(output_dir, "images", "front", f"episode_{ep_idx:06d}")
        os.makedirs(img_dir, exist_ok=True)
        for step_idx, img in enumerate(ep.images):
            # Save as CHW uint8 numpy (compact)
            img_path = os.path.join(img_dir, f"frame_{step_idx:06d}.npy")
            np.save(img_path, img)
            image_paths.append(
                f"images/front/episode_{ep_idx:06d}/frame_{step_idx:06d}.npy"
            )

        # Create parquet table
        table = pa.table({
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": ep_indices,
            "task_index": task_indices,
            "observation.state": states,
            "action": actions,
            "observation.images.front": image_paths,
            "next.reward": ep.rewards,
            "next.done": [False] * (ep_len - 1) + [True],
            "next.success": [False] * (ep_len - 1) + [ep.success],
        })

        pq.write_table(
            table,
            os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet"),
        )

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [args.instruction],
            "length": ep_len,
        })

        frame_idx += ep_len

    # Save metadata
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot,
        "fps": args.fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "observation.images.front": {
                "dtype": "image",
                "shape": list(image_shape),
                "names": ["channel", "height", "width"],
            },
            "action": {"dtype": "float32", "shape": [action_dim]},
        },
        "task": args.instruction,
        "source_model": args.model_path,
        "lora_path": args.lora_path,
        "collection_date": datetime.now().isoformat(),
    }

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Tasks file
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.instruction}) + "\n")

    # Episodes file
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")

    log(f"[collect] Dataset saved: {output_dir}")
    log(f"[collect]   Episodes: {len(episodes)}")
    log(f"[collect]   Frames:   {total_frames}")
    log(f"[collect]   Success:  {sum(ep.success for ep in episodes)}/{len(episodes)}")

    return output_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    np.random.seed(args.seed)

    log(f"[collect] Robot: {args.robot}, Episodes: {args.total_episodes}, "
        f"Keep: {args.keep_mode}")

    # --- Environment ---
    env = make_env(args)
    log("[collect] Environment ready.")

    # --- VLA ---
    vla = load_vla(args)
    log("[collect] VLA loaded.")

    # --- Output dir ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, f"datasets/{args.dataset_name}_{timestamp}"
    )

    # --- Collect episodes ---
    all_episodes: list[EpisodeRecorder] = []
    successes = 0
    start_time = time.time()

    for ep_num in range(1, args.total_episodes + 1):
        obs, info = env.reset()
        recorder = EpisodeRecorder()

        for step in range(args.max_steps):
            image = obs["observation.images.front"]  # (3, 256, 256) uint8
            state = obs["observation.state"]

            # VLA inference
            # Convert CHW -> HWC for the VLA wrapper
            image_hwc = np.transpose(image, (1, 2, 0))
            vla_action = vla.predict(image_hwc, args.instruction)
            raw = vla_action.raw  # (7,)

            # Map to env action
            env_action = vla_action_to_env_action(raw, args.robot)

            # Step env
            obs, reward, terminated, truncated, info = env.step(env_action)

            # Record observation-action-reward tuple
            recorder.add_step(image, state, env_action, reward)

            if terminated:
                recorder.success = True
                successes += 1
                break
            if truncated:
                break

        all_episodes.append(recorder)

        # Progress
        if ep_num % 10 == 0 or ep_num == args.total_episodes:
            elapsed = time.time() - start_time
            eps_per_hour = ep_num / max(elapsed / 3600, 1e-6)
            success_rate = successes / ep_num
            log(
                f"[collect] ep={ep_num}/{args.total_episodes}  "
                f"success={success_rate:.1%} ({successes}/{ep_num})  "
                f"eps/hr={eps_per_hour:.0f}  "
                f"last_reward={recorder.total_reward:.2f}  "
                f"last_steps={len(recorder)}"
            )

    # --- Filter episodes ---
    if args.keep_mode == "successful":
        kept = [ep for ep in all_episodes if ep.success]
        log(f"[collect] Keeping {len(kept)} successful episodes out of {len(all_episodes)}")
    elif args.keep_mode == "top_k":
        sorted_eps = sorted(all_episodes, key=lambda ep: ep.total_reward, reverse=True)
        k = max(1, int(len(sorted_eps) * args.top_k_pct / 100))
        kept = sorted_eps[:k]
        log(f"[collect] Keeping top {k} episodes (top {args.top_k_pct}%) by reward")
    else:
        kept = all_episodes
        log(f"[collect] Keeping all {len(kept)} episodes")

    if not kept:
        log("[collect] WARNING: No episodes to save! OpenVLA never succeeded.")
        log("[collect] Try: --keep_mode top_k --top_k_pct 50  to keep the best attempts")
        log("[collect] Or: --keep_mode all  to keep everything for DAgger-style training")
        env.close()
        simulation_app.close()
        return

    # --- Save dataset ---
    save_lerobot_dataset(kept, args, output_dir)

    # --- Summary ---
    elapsed = time.time() - start_time
    total_success = sum(ep.success for ep in all_episodes)
    log(f"\n[collect] Collection complete!")
    log(f"  Total episodes:    {len(all_episodes)}")
    log(f"  Successes:         {total_success} ({total_success/len(all_episodes):.1%})")
    log(f"  Kept episodes:     {len(kept)}")
    log(f"  Kept frames:       {sum(len(ep) for ep in kept)}")
    log(f"  Time:              {elapsed/60:.1f} min")
    log(f"  Dataset:           {output_dir}")
    log(f"\nNext step: fine-tune OpenVLA on this dataset with LeRobot:")
    log(f"  python -m lerobot.scripts.train \\")
    log(f"    --dataset.repo_id={output_dir} \\")
    log(f"    --policy.type=vla")

    if args.push_to_hub and args.hub_repo_id:
        log(f"\n[collect] Pushing to HuggingFace Hub: {args.hub_repo_id}")
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            repo_id=args.hub_repo_id,
            repo_type="dataset",
        )
        log(f"[collect] Pushed to: https://huggingface.co/datasets/{args.hub_repo_id}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
