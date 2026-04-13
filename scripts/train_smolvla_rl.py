#!/usr/bin/env python3
"""SmolVLA reward-weighted online fine-tuning for SO100 cube lifting.

This script runs the SO100 Isaac Sim environment, executes SmolVLA in the loop,
collects trajectories, and fine-tunes a small subset of SmolVLA parameters using
reward-weighted regression on the actions that produced higher returns.

It is intentionally narrow:
- robot: SO100 only
- task: pick up the red cube
- policy: SmolVLA only

Usage:
    <ISAAC_SIM>/python.sh scripts/train_smolvla_rl.py --headless
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA reward fine-tuning on SO100")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_config", type=str, default="configs/sim/so100_standalone.yaml")
    parser.add_argument("--instruction", type=str, default="pick up the red cube")
    parser.add_argument("--policy_path", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--image_key", type=str, default="observation.images.front")
    parser.add_argument("--state_key", type=str, default="observation.state")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--total_episodes", type=int, default=200)
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--gradient_updates", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_temperature", type=float, default=2.0)
    parser.add_argument("--weight_clip", type=float, default=20.0)
    parser.add_argument("--action_noise_std", type=float, default=0.10)
    parser.add_argument("--gripper_noise_prob", type=float, default=0.05)
    parser.add_argument("--success_bonus", type=float, default=10.0)
    parser.add_argument("--lift_weight", type=float, default=10.0)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=25)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


args = parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"

for path in (PROJECT_ROOT, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402
from sim.isaac.so100_gym_wrapper import IsaacSO100Env  # noqa: E402


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


@dataclass
class Transition:
    state: np.ndarray
    image: np.ndarray
    action: np.ndarray
    task: str
    reward: float
    episode_return: float = 0.0
    success: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_policy_config(device: str) -> SmolVLAConfig:
    return SmolVLAConfig(
        device=device,
        chunk_size=1,
        n_action_steps=1,
        n_obs_steps=1,
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            args.image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )


def freeze_for_partial_finetune(policy: SmolVLAPolicy) -> list[str]:
    trainable_prefixes = (
        "model.vlm_with_expert.lm_expert",
        "model.state_proj",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.action_time_mlp_in",
        "model.action_time_mlp_out",
    )
    trainable = []
    for name, param in policy.named_parameters():
        keep_trainable = name.startswith(trainable_prefixes)
        param.requires_grad_(keep_trainable)
        if keep_trainable:
            trainable.append(name)
    return trainable


def build_single_batch(
    obs: dict,
    task_tokens: torch.Tensor,
    task_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image = torch.from_numpy(obs[args.image_key]).float().unsqueeze(0).to(device) / 255.0
    state = torch.from_numpy(obs[args.state_key]).float().unsqueeze(0).to(device)
    return {
        args.image_key: image,
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: task_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: task_mask.to(dtype=torch.bool),
    }


def build_training_batch(
    transitions: list[Transition],
    tokenizer,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tasks = [t.task if t.task.endswith("\n") else f"{t.task}\n" for t in transitions]
    tokenized = tokenizer(
        tasks,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=48,
    )
    states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(device)
    images = torch.from_numpy(np.stack([t.image for t in transitions])).float().to(device) / 255.0
    actions = torch.from_numpy(np.stack([t.action for t in transitions])).float().unsqueeze(1).to(device)
    returns = torch.tensor([t.episode_return for t in transitions], dtype=torch.float32, device=device)
    batch = {
        OBS_STATE: states,
        args.image_key: images,
        ACTION: actions,
        OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].to(device=device, dtype=torch.bool),
    }
    return batch, returns


def compute_dense_reward(info: dict, success_bonus: float, lift_weight: float, cube_rest_height: float) -> float:
    cube_pos = info.get("cube_pos")
    dense = 0.0
    if cube_pos is not None:
        dense += lift_weight * max(0.0, float(cube_pos[2]) - cube_rest_height)
    if info.get("is_success", False):
        dense += success_bonus
    return dense


def add_exploration_noise(action: np.ndarray) -> np.ndarray:
    noisy = action.copy()
    noisy[:5] += np.random.normal(0.0, args.action_noise_std, size=5).astype(np.float32)
    if np.random.rand() < args.gripper_noise_prob:
        noisy[5] *= -1.0
    return np.clip(noisy, -1.0, 1.0)


def compute_sample_weights(returns: torch.Tensor) -> torch.Tensor:
    if returns.numel() == 1:
        return torch.ones_like(returns)
    normalized = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
    weights = torch.exp(args.weight_temperature * normalized)
    weights = torch.clamp(weights, max=args.weight_clip)
    return weights / (weights.mean() + 1e-6)


def sample_transitions(replay: deque[Transition], batch_size: int) -> list[Transition]:
    if len(replay) <= batch_size:
        return list(replay)
    indices = np.random.choice(len(replay), size=batch_size, replace=False)
    replay_list = list(replay)
    return [replay_list[i] for i in indices]


def save_checkpoint(policy: SmolVLAPolicy, checkpoint_root: Path, episode: int) -> Path:
    ckpt_dir = checkpoint_root / f"episode_{episode:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    return ckpt_dir


def main() -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "outputs" / "smolvla_rl" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    tokenizer = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct").tokenizer
    task_text = args.instruction if args.instruction.endswith("\n") else f"{args.instruction}\n"
    task_tokens = tokenizer(
        [task_text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=48,
    )
    task_tokens = task_tokens["input_ids"].to(device)
    task_mask = tokenizer(
        [task_text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=48,
    )["attention_mask"].to(device=device, dtype=torch.bool)

    env = None
    try:
        env = IsaacSO100Env(
            simulation_app=simulation_app,
            sim_config_path=str(PROJECT_ROOT / args.sim_config),
            max_episode_steps=args.max_steps,
            cube_randomize=True,
        )
        cube_rest_height = env.cube_rest_height

        log("[SmolVLA-RL] Loading policy...")
        config = make_policy_config(device.type)
        policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=config, strict=False)
        policy.to(device)
        trainable_names = freeze_for_partial_finetune(policy)
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        log(f"[SmolVLA-RL] Policy ready. Trainable params: {len(trainable_names)} tensors")

        replay: deque[Transition] = deque(maxlen=args.buffer_size)
        recent_returns: deque[float] = deque(maxlen=20)
        recent_successes: deque[float] = deque(maxlen=20)
        global_step = 0

        for episode in range(1, args.total_episodes + 1):
            obs, _ = env.reset()
            policy.reset()
            episode_transitions: list[Transition] = []
            episode_return = 0.0
            episode_success = False
            max_cube_height = 0.0

            for step_idx in range(args.max_steps):
                batch = build_single_batch(obs, task_tokens, task_mask, device)
                with torch.inference_mode():
                    action_tensor = policy.select_action(batch)
                action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                action = add_exploration_noise(action)

                next_obs, env_reward, terminated, truncated, info = env.step(action)
                dense_reward = float(env_reward) + compute_dense_reward(
                    info, args.success_bonus, args.lift_weight, cube_rest_height
                )
                episode_return += dense_reward
                max_cube_height = max(max_cube_height, float(info["cube_pos"][2]))

                episode_transitions.append(
                    Transition(
                        state=obs[args.state_key].copy(),
                        image=obs[args.image_key].copy(),
                        action=action.copy(),
                        task=args.instruction,
                        reward=dense_reward,
                    )
                )

                obs = next_obs
                global_step += 1

                if terminated:
                    episode_success = True
                    break
                if truncated:
                    break

            for transition in episode_transitions:
                transition.episode_return = episode_return
                transition.success = episode_success
                replay.append(transition)

            recent_returns.append(episode_return)
            recent_successes.append(1.0 if episode_success else 0.0)

            writer.add_scalar("train/episode_return", episode_return, episode)
            writer.add_scalar("train/episode_length", len(episode_transitions), episode)
            writer.add_scalar("train/success", float(episode_success), episode)
            writer.add_scalar("train/max_cube_height", max_cube_height, episode)
            writer.add_scalar("train/replay_size", len(replay), episode)
            writer.add_scalar("train/rolling_return_20", float(np.mean(recent_returns)), episode)
            writer.add_scalar("train/rolling_success_20", float(np.mean(recent_successes)), episode)

            if episode % args.log_freq == 0:
                log(
                    f"[SmolVLA-RL] Episode {episode}/{args.total_episodes} "
                    f"return={episode_return:.3f} success={int(episode_success)} "
                    f"len={len(episode_transitions)} max_cube_z={max_cube_height:.3f} "
                    f"replay={len(replay)}"
                )

            if len(replay) >= args.batch_size and episode % args.update_every == 0:
                policy.train()
                update_losses = []
                update_weight_means = []
                for update_idx in range(args.gradient_updates):
                    sampled = sample_transitions(replay, args.batch_size)
                    train_batch, returns = build_training_batch(sampled, tokenizer, device)
                    per_sample_loss, loss_dict = policy.forward(train_batch, reduction="none")
                    weights = compute_sample_weights(returns)
                    loss = (per_sample_loss * weights).sum() / (weights.sum() + 1e-6)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                    optimizer.step()

                    update_losses.append(float(loss.item()))
                    update_weight_means.append(float(weights.mean().item()))
                    writer.add_scalar("train/update_loss", float(loss.item()), global_step + update_idx)
                    writer.add_scalar(
                        "train/update_weight_mean", float(weights.mean().item()), global_step + update_idx
                    )
                    if "loss" in loss_dict:
                        writer.add_scalar("train/model_loss_raw", float(loss_dict["loss"]), global_step + update_idx)
                policy.eval()
                writer.add_scalar("train/update_loss_mean", float(np.mean(update_losses)), episode)
                writer.add_scalar("train/update_weight_mean_epoch", float(np.mean(update_weight_means)), episode)

            if episode % args.checkpoint_freq == 0:
                ckpt_dir = save_checkpoint(policy, checkpoint_dir, episode)
                log(f"[SmolVLA-RL] Checkpoint saved to {ckpt_dir}")

        final_ckpt = save_checkpoint(policy, checkpoint_dir, args.total_episodes)
        log(f"[SmolVLA-RL] Final checkpoint saved to {final_ckpt}")
        log(f"[SmolVLA-RL] TensorBoard logdir: {log_dir}")
    finally:
        writer.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
