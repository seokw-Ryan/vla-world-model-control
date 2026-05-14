#!/usr/bin/env python3
"""Option 2: LoRA + RL fine-tuning of OpenVLA in Isaac Sim.

Loads OpenVLA in 4-bit, adds LoRA adapters (~2M trainable params), and
trains them with REINFORCE (policy gradient) using environment rewards.
The full VLA processes camera images + instruction → actions, but only
the LoRA weights are updated.

Usage (run with Isaac Sim's python.sh):
    <ISAAC_SIM>/python.sh scripts/train_rl_lora.py --robot so100
    <ISAAC_SIM>/python.sh scripts/train_rl_lora.py --robot franka --total_episodes 500

Requires:
    pip install peft accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Argument parsing — BEFORE SimulationApp
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA + RL fine-tuning of OpenVLA")

    # Environment
    p.add_argument("--robot", type=str, default="so100", choices=["franka", "so100"])
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    p.add_argument("--instruction", type=str, default="pick up the red cube")

    # VLA model
    p.add_argument("--model_path", type=str, default="openvla/openvla-7b")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # RL (REINFORCE)
    p.add_argument("--total_episodes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--baseline_ema", type=float, default=0.99,
                   help="EMA decay for reward baseline (variance reduction)")
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--update_every", type=int, default=1,
                   help="Update LoRA weights every N episodes")

    # Reward
    p.add_argument("--success_bonus", type=float, default=10.0)
    p.add_argument("--lift_weight", type=float, default=2.0)
    p.add_argument("--reach_weight", type=float, default=1.0)

    # Logging
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--checkpoint_freq", type=int, default=50,
                   help="Save LoRA weights every N episodes")
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap
# ---------------------------------------------------------------------------

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import time  # noqa: E402
from collections import deque  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# OpenVLA with LoRA
# ---------------------------------------------------------------------------


class OpenVLALoRA:
    """OpenVLA model with LoRA adapters for RL fine-tuning.

    Wraps the HuggingFace OpenVLA model, adds LoRA to the language model
    layers, and provides methods to:
      - predict actions with log probabilities (for REINFORCE)
      - update LoRA weights given rewards
      - save/load LoRA adapters
    """

    def __init__(self, model_path: str, lora_rank: int, lora_alpha: int,
                 lora_dropout: float, lr: float):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        log(f"[VLA-LoRA] Loading {model_path} in 4-bit...")
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        log(f"[VLA-LoRA] Adding LoRA (rank={lora_rank}, alpha={lora_alpha})...")
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Optimizer — only LoRA params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # Image preprocessing
        self.image_size = 224
        self.unnorm_key = "bridge_orig"

        log("[VLA-LoRA] Ready.")

    def predict_with_log_prob(
        self, image: np.ndarray, instruction: str
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Run VLA inference and return (action_7d, log_prob).

        Uses teacher-forced generation to get per-token log probabilities
        for the predicted action tokens.
        """
        from PIL import Image

        # Preprocess image
        if image.ndim == 3 and image.shape[0] in (3, 4):
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_image = Image.fromarray(image, "RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )

        prompt = f"In: {instruction}\nOut:"
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)

        # Step 1: Generate action tokens (with sampling for exploration)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                output_scores=False,
            )

        # Get the raw action from the model's built-in decoding
        with torch.no_grad():
            raw_action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=True,
            )

        # Step 2: Teacher-force the generated tokens to get log probs
        # Extract only the generated (action) tokens
        input_len = inputs["input_ids"].shape[1]
        action_token_ids = generated[:, input_len:]

        # Forward pass with action tokens to get logits
        self.model.train()
        full_ids = generated
        labels = full_ids.clone()
        labels[:, :input_len] = -100  # Mask input tokens in loss

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            labels=labels,
        )

        # Per-token log probs for the action tokens
        logits = outputs.logits[:, input_len - 1:-1, :]  # Shifted by 1
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, action_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probs across action tokens
        total_log_prob = token_log_probs.sum()

        return np.array(raw_action, dtype=np.float64), total_log_prob

    def update(self, log_probs: list[torch.Tensor], advantages: list[float],
               entropy_coef: float, max_grad_norm: float) -> float:
        """REINFORCE update: loss = -sum(log_prob * advantage)."""
        self.model.train()

        policy_loss = torch.tensor(0.0, device=self.device)
        for lp, adv in zip(log_probs, advantages):
            policy_loss += -lp * adv

        policy_loss = policy_loss / max(len(log_probs), 1)

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        self.optimizer.step()

        return policy_loss.item()

    def save_lora(self, path: str):
        """Save only the LoRA adapter weights."""
        self.model.save_pretrained(path)
        log(f"[VLA-LoRA] LoRA weights saved to: {path}")

    def load_lora(self, path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model.base_model, path)
        log(f"[VLA-LoRA] LoRA weights loaded from: {path}")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_env(args):
    """Create the Isaac Sim gym environment."""
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


def vla_action_to_env_action(raw_action: np.ndarray, robot: str) -> np.ndarray:
    """Map VLA 7D output to environment action space.

    VLA outputs: [dx, dy, dz, drx, dry, drz, gripper]
    Franka env expects: [dx, dy, dz, drx, dry, drz, gripper] (7D) — direct match
    SO-100 env expects: [shoulder_pan, shoulder_lift, elbow_flex,
                         wrist_flex, wrist_roll, gripper] (6D)
    """
    if robot == "franka":
        return np.clip(raw_action, -1.0, 1.0).astype(np.float32)
    else:
        # Map VLA 7D -> SO-100 6D (same mapping as openvla_policy.py)
        action = np.zeros(6, dtype=np.float32)
        scale = 0.5  # Scale VLA deltas to [-1, 1] range
        action[0] = np.clip(raw_action[0] * scale, -1.0, 1.0)  # shoulder_pan
        action[1] = np.clip(raw_action[1] * scale, -1.0, 1.0)  # shoulder_lift
        action[2] = np.clip(raw_action[2] * scale, -1.0, 1.0)  # elbow_flex
        action[3] = np.clip(raw_action[3] * scale, -1.0, 1.0)  # wrist_flex
        action[4] = np.clip(raw_action[4] * scale, -1.0, 1.0)  # wrist_roll
        action[5] = 1.0 if raw_action[6] > 0.5 else -1.0       # gripper
        return action


def compute_step_reward(env_reward: float, info: dict, args) -> float:
    """Dense per-step reward."""
    reward = env_reward
    if info.get("is_success", False):
        reward += args.success_bonus
    cube_pos = info.get("cube_pos")
    if cube_pos is not None:
        lift = max(0.0, float(cube_pos[2]) - 0.525)
        reward += args.lift_weight * lift
    ee_pos = info.get("ee_pos")
    if ee_pos is not None and cube_pos is not None:
        dist = np.linalg.norm(ee_pos - cube_pos)
        reward -= args.reach_weight * dist
    return reward


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(f"[train_rl_lora] Robot: {args.robot}, Episodes: {args.total_episodes}")

    # --- Environment ---
    env = make_env(args)
    log("[train_rl_lora] Environment ready.")

    # --- VLA + LoRA ---
    vla = OpenVLALoRA(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
    )

    # VRAM check
    allocated = torch.cuda.memory_allocated() / 1e9
    log(f"[train_rl_lora] GPU memory after model load: {allocated:.1f} GB")

    # --- Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        PROJECT_ROOT, f"outputs/rl_lora/{args.robot}_{timestamp}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tb"))

    # --- Training state ---
    reward_baseline = 0.0  # EMA baseline for variance reduction
    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    best_success_rate = 0.0
    start_time = time.time()

    # Accumulate episodes for batched updates
    episode_log_probs = []
    episode_advantages = []

    log(f"[train_rl_lora] Starting RL fine-tuning for {args.total_episodes} episodes")
    log(f"[train_rl_lora] Checkpoints: {checkpoint_dir}")

    for ep in range(1, args.total_episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_success = False
        step_log_probs = []
        step_rewards = []

        for step in range(args.max_steps):
            # Get camera image (CHW uint8 -> HWC for VLA)
            image = obs["observation.images.front"]  # (3, 256, 256) uint8

            # VLA inference with log prob
            raw_action, log_prob = vla.predict_with_log_prob(image, args.instruction)

            # Map to env action space
            env_action = vla_action_to_env_action(raw_action, args.robot)

            # Step environment
            obs, env_reward, terminated, truncated, info = env.step(env_action)

            # Dense reward
            reward = compute_step_reward(env_reward, info, args)
            step_rewards.append(reward)
            step_log_probs.append(log_prob)

            ep_reward += reward
            ep_success = ep_success or terminated

            if terminated or truncated:
                break

        # --- Compute discounted returns & advantages ---
        returns = []
        G = 0.0
        for r in reversed(step_rewards):
            G = r + args.gamma * G
            returns.insert(0, G)

        ep_return = returns[0] if returns else 0.0

        # Update baseline (EMA)
        reward_baseline = args.baseline_ema * reward_baseline + (1 - args.baseline_ema) * ep_return

        # Advantages = return - baseline (per-step)
        for lp, R in zip(step_log_probs, returns):
            episode_log_probs.append(lp)
            episode_advantages.append(R - reward_baseline)

        recent_rewards.append(ep_reward)
        recent_successes.append(float(ep_success))

        # --- Update LoRA weights ---
        if ep % args.update_every == 0 and episode_log_probs:
            loss = vla.update(
                episode_log_probs, episode_advantages,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
            )
            episode_log_probs = []
            episode_advantages = []

            writer.add_scalar("train/policy_loss", loss, ep)

        # --- Logging ---
        if ep % args.log_freq == 0:
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            success_rate = np.mean(recent_successes) if recent_successes else 0.0
            elapsed = time.time() - start_time
            eps_per_hour = ep / max(elapsed / 3600, 1e-6)

            log(
                f"[ep {ep:>5}/{args.total_episodes}] "
                f"reward={mean_reward:>7.2f}  "
                f"success={success_rate:>5.1%}  "
                f"baseline={reward_baseline:>7.2f}  "
                f"eps/hr={eps_per_hour:>.0f}  "
                f"vram={torch.cuda.memory_allocated()/1e9:.1f}GB"
            )

            writer.add_scalar("train/mean_reward", mean_reward, ep)
            writer.add_scalar("train/success_rate", success_rate, ep)
            writer.add_scalar("train/reward_baseline", reward_baseline, ep)
            writer.add_scalar("train/ep_reward", ep_reward, ep)

        # --- Checkpoint ---
        if ep % args.checkpoint_freq == 0:
            lora_path = os.path.join(checkpoint_dir, f"lora_ep{ep}")
            vla.save_lora(lora_path)

            success_rate = np.mean(recent_successes) if recent_successes else 0.0
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_path = os.path.join(checkpoint_dir, "best_lora")
                vla.save_lora(best_path)
                log(f"[train_rl_lora] New best (success={best_success_rate:.1%})")

    # --- Final save ---
    final_path = os.path.join(checkpoint_dir, "final_lora")
    vla.save_lora(final_path)

    elapsed = time.time() - start_time
    final_success = np.mean(recent_successes) if recent_successes else 0.0
    log(f"\n[train_rl_lora] Done! episodes={args.total_episodes} "
        f"time={elapsed/3600:.1f}h success={final_success:.1%}")
    log(f"[train_rl_lora] LoRA weights: {checkpoint_dir}")

    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
