"""Evaluate a trained LeRobot policy on the SO-100 in Isaac Sim.

Loads a LeRobot policy checkpoint and runs closed-loop evaluation
in the Isaac Sim SO-100 environment.

Usage:
    <ISAAC_SIM>/python.sh scripts/eval_so100_policy.py \
        --policy_path outputs/train/act_so100/checkpoints/last/pretrained_model \
        --num_episodes 50 \
        --headless

IMPORTANT: Must be run with Isaac Sim's python.sh.
"""

from __future__ import annotations

import argparse
import sys
import os

# ─── Parse args before any Isaac Sim imports ──────────────────────────────

parser = argparse.ArgumentParser(description="Evaluate LeRobot policy on SO-100")
parser.add_argument("--policy_path", type=str, required=True,
                    help="Path to trained policy (local dir or HF hub ID)")
parser.add_argument("--num_episodes", type=int, default=50,
                    help="Number of evaluation episodes")
parser.add_argument("--max_steps", type=int, default=200,
                    help="Max steps per episode")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--sim_config", type=str,
                    default="configs/sim/so100_standalone.yaml")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device for policy inference")
args = parser.parse_args()


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# ─── Create SimulationApp ────────────────────────────────────────────────

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# ─── Imports ─────────────────────────────────────────────────────────────

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sim.isaac.so100_gym_wrapper import IsaacSO100Env

# ─── Load policy ─────────────────────────────────────────────────────────

from lerobot.policies.pretrained import PreTrainedPolicy

log(f"[Eval] Loading policy from {args.policy_path}...")
policy = PreTrainedPolicy.from_pretrained(args.policy_path)
policy.to(args.device)
policy.eval()
log("[Eval] Policy loaded.")

# ─── Create env ──────────────────────────────────────────────────────────

env = IsaacSO100Env(
    simulation_app=simulation_app,
    sim_config_path=args.sim_config,
    max_episode_steps=args.max_steps,
    cube_randomize=True,
)

# ─── Evaluation loop ─────────────────────────────────────────────────────

log(f"[Eval] Running {args.num_episodes} episodes...")

successes = 0
total_steps = 0

for ep in range(args.num_episodes):
    obs, info = env.reset()
    policy.reset()
    episode_success = False
    ep_steps = 0

    for t in range(args.max_steps):
        # Convert obs to policy input format (batched tensors)
        policy_obs = {
            "observation.state": torch.from_numpy(
                obs["observation.state"]
            ).unsqueeze(0).to(args.device),
            "observation.images.front": torch.from_numpy(
                obs["observation.images.front"]
            ).unsqueeze(0).float().to(args.device),
        }

        with torch.inference_mode():
            action_tensor = policy.select_action(policy_obs)

        action = action_tensor.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_steps += 1

        if terminated:
            episode_success = True
            break
        if truncated:
            break

    total_steps += ep_steps
    if episode_success:
        successes += 1

    log(f"[Eval] Episode {ep + 1}/{args.num_episodes} — "
        f"{'SUCCESS' if episode_success else 'FAIL'} — "
        f"steps: {ep_steps} — "
        f"running success: {successes}/{ep + 1} "
        f"({100 * successes / (ep + 1):.1f}%)")

# ─── Summary ─────────────────────────────────────────────────────────────

success_rate = 100 * successes / args.num_episodes
avg_steps = total_steps / args.num_episodes

log(f"\n[Eval] ═══ Results ═══")
log(f"[Eval] Episodes:     {args.num_episodes}")
log(f"[Eval] Successes:    {successes}")
log(f"[Eval] Success rate: {success_rate:.1f}%")
log(f"[Eval] Avg steps:    {avg_steps:.1f}")
log(f"[Eval] Policy:       {args.policy_path}")

# ─── Cleanup ─────────────────────────────────────────────────────────────

env.close()
simulation_app.close()
log("[Eval] Done.")
