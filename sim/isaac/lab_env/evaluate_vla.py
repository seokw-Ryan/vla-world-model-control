"""Isaac Lab VLA evaluation script.

Runs a VLA policy in the FrankaVLAEnv for multiple episodes and reports
success rate and per-episode metrics.

Usage:
    python sim/isaac/lab_env/evaluate_vla.py --headless --enable_cameras --num_episodes 5
    python sim/isaac/lab_env/evaluate_vla.py --instruction "pick up the red cube" --model_path openvla/openvla-7b
"""

from __future__ import annotations

import argparse

# ─── Phase 1: Parse args (AppLauncher pattern — args before everything) ──────

parser = argparse.ArgumentParser(description="Evaluate VLA policy in Isaac Lab")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--instruction", type=str, default="pick up the red cube",
                    help="Natural language instruction")
parser.add_argument("--model_path", type=str, default=None, help="Override VLA model path")
parser.add_argument("--vla_config", type=str, default="configs/vla/openvla_default.yaml",
                    help="Path to VLA config YAML")

# ─── Phase 2: AppLauncher (creates SimulationApp internally) ─────────────────

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ─── Phase 3: Import everything else (after SimulationApp is created) ────────

import numpy as np
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from vla_world_model_control.shared import add_project_root_to_sys_path, build_openvla_config

project_root = add_project_root_to_sys_path()
from vla_world_model_control.shared import OpenVLAWrapper
from sim.isaac.lab_env.franka_vla_env_cfg import FrankaVLAEnvCfg

# ─── Phase 4: Load VLA config ───────────────────────────────────────────────
vla_config = build_openvla_config(args.vla_config, model_path=args.model_path)

# ─── Phase 5: Create environment ────────────────────────────────────────────

env_cfg = FrankaVLAEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)

# ─── Phase 6: Load VLA model ────────────────────────────────────────────────

print("[Evaluate] Loading VLA model...")
vla = OpenVLAWrapper(vla_config).load()

# ─── Phase 7: Episode loop ──────────────────────────────────────────────────

results = []
max_steps_per_episode = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

print(f"[Evaluate] Running {args.num_episodes} episodes, "
      f"max {max_steps_per_episode} steps each.")
print(f"[Evaluate] Instruction: '{args.instruction}'")

for episode in range(args.num_episodes):
    obs, info = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False

    while not done and episode_steps < max_steps_per_episode:
        # Extract RGB image from observation
        # obs["image"]["image"] is a tensor of shape (num_envs, H, W, C) or (num_envs, C, H, W)
        image_tensor = obs["image"]["image"]
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor[0].cpu().numpy()  # first env
        else:
            image_np = np.array(image_tensor[0])

        # Handle CHW vs HWC format
        if image_np.ndim == 3 and image_np.shape[0] in (3, 4):
            image_np = np.transpose(image_np, (1, 2, 0))  # CHW → HWC

        # Convert to uint8 if needed
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        # VLA inference
        vla_action = vla.predict(image_np, args.instruction)

        # Convert VLAAction to env action tensor
        # Arm action: 6D (delta_pos(3) + delta_rot(3)), Gripper: 1D binary
        arm_action = np.concatenate([vla_action.delta_pos, vla_action.delta_rot])
        gripper_binary = 1.0 if vla_action.gripper > 0.5 else 0.0  # 1=open, 0=close

        # Construct full action: [arm(6), gripper(1)]
        action_np = np.concatenate([arm_action, [gripper_binary]])
        action_tensor = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_tensor)

        episode_reward += reward[0].item()
        episode_steps += 1
        done = terminated[0].item() or truncated[0].item()

    success = episode_reward > 0.0
    results.append({
        "episode": episode,
        "steps": episode_steps,
        "reward": episode_reward,
        "success": success,
    })

    print(f"  Episode {episode + 1}/{args.num_episodes}: "
          f"steps={episode_steps}, reward={episode_reward:.3f}, "
          f"success={success}")

# ─── Phase 8: Summary ───────────────────────────────────────────────────────

num_success = sum(1 for r in results if r["success"])
success_rate = num_success / len(results) if results else 0.0
avg_reward = np.mean([r["reward"] for r in results]) if results else 0.0
avg_steps = np.mean([r["steps"] for r in results]) if results else 0.0

print("\n" + "=" * 60)
print(f"[Evaluate] Results over {len(results)} episodes:")
print(f"  Success rate: {success_rate:.1%} ({num_success}/{len(results)})")
print(f"  Avg reward:   {avg_reward:.3f}")
print(f"  Avg steps:    {avg_steps:.1f}")
print("=" * 60)

# ─── Phase 9: Cleanup ───────────────────────────────────────────────────────

env.close()
simulation_app.close()
print("[Evaluate] Done.")
