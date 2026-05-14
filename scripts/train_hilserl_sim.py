#!/usr/bin/env python3
"""Create and optionally launch LeRobot HILSERL sim training.

This follows the Hugging Face LeRobot HILSERL simulation guide:
https://huggingface.co/docs/lerobot/hilserl_sim

The documented workflow trains a SAC policy in `gym_hil` using two processes:
`lerobot.rl.learner` and `lerobot.rl.actor`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run HILSERL sim SAC training")
    default_job_name = "franka_sim_sac"
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lerobot" / "gym_hil_train_config.json",
        help="Where to write the generated LeRobot JSON config.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="aractingi/franka_sim_pick_lift_5",
        help="Offline dataset repo used to bootstrap SAC, matching the upstream example by default.",
    )
    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default="aractingi/franka_sim_pick_lift",
        help="Repo id metadata for the trained SAC policy config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional training run output directory. If omitted, LeRobot picks one at runtime.",
    )
    parser.add_argument("--job-name", type=str, default=default_job_name)
    parser.add_argument("--task", type=str, default="PandaPickCubeGamepad-v0")
    parser.add_argument("--control-mode", type=str, choices=("gamepad", "keyboard"), default="gamepad")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--learner-host", type=str, default="127.0.0.1")
    parser.add_argument("--learner-port", type=int, default=50051)
    parser.add_argument("--wandb-project", type=str, default="franka_sim")
    parser.add_argument("--wandb-enable", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start learner first, then actor, using the generated config.",
    )
    parser.add_argument(
        "--actor-delay-seconds",
        type=float,
        default=5.0,
        help="Delay after launching learner before starting actor.",
    )
    return parser.parse_args()


def make_train_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "job_name": args.job_name,
        "resume": False,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "eval_freq": 20000,
        "log_freq": 1,
        "save_checkpoint": True,
        "save_freq": 20000,
        "use_policy_training_preset": True,
        "optimizer": None,
        "scheduler": None,
        "wandb": {
            "enable": args.wandb_enable,
            "project": args.wandb_project,
            "disable_artifact": True,
        },
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "use_imagenet_stats": False,
        },
        "env": {
            "type": "gym_manipulator",
            "name": "gym_hil",
            "task": args.task,
            "fps": args.fps,
            "robot": None,
            "teleop": None,
            "processor": {
                "control_mode": args.control_mode,
                "gripper": {
                    "use_gripper": True,
                    "gripper_penalty": -0.02,
                },
                "reset": {
                    "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
                    "reset_time_s": 2.0,
                    "control_time_s": 15.0,
                    "terminate_on_success": True,
                },
            },
            "features": {
                "observation.images.front": {"type": "VISUAL", "shape": [3, 128, 128]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 128, 128]},
                "observation.state": {"type": "STATE", "shape": [18]},
                "action": {"type": "ACTION", "shape": [3]},
            },
            "features_map": {
                "observation.images.front": "observation.images.front",
                "observation.images.wrist": "observation.images.wrist",
                "observation.state": "observation.state",
                "action": "action",
            },
        },
        "policy": {
            "type": "sac",
            "n_obs_steps": 1,
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MIN_MAX",
                "ENV": "MIN_MAX",
                "ACTION": "MIN_MAX",
            },
            "input_features": {
                "observation.images.front": {"type": "VISUAL", "shape": [3, 128, 128]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 128, 128]},
                "observation.state": {"type": "STATE", "shape": [18]},
            },
            "output_features": {
                "action": {"type": "ACTION", "shape": [3]},
            },
            "device": args.device,
            "use_amp": False,
            "dataset_stats": {
                "observation.images.front": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "observation.images.wrist": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "observation.state": {
                    "min": [
                        -0.6897139549255371,
                        -1.1421763896942139,
                        -0.5745007991790771,
                        -2.97829008102417,
                        -0.2710767090320587,
                        1.3246592283248901,
                        -0.04057434946298599,
                        -0.21261805295944214,
                        -0.4548068344593048,
                        -0.6540042757987976,
                        -0.3644964098930359,
                        -1.1057522296905518,
                        -0.40768879652023315,
                        -0.2220114767551422,
                        0.0,
                        0.19176171720027924,
                        -0.3013063669204712,
                        0.00362197193317115,
                    ],
                    "max": [
                        0.5107022523880005,
                        0.5516204237937927,
                        0.5620884299278259,
                        -1.3330878019332886,
                        0.32758936285972595,
                        3.119610548019409,
                        1.8364211320877075,
                        0.25358933210372925,
                        0.36316126585006714,
                        0.14765967428684235,
                        0.49947625398635864,
                        0.144814133644104,
                        0.2820609211921692,
                        0.7382049560546875,
                        255.0,
                        0.6012658476829529,
                        0.3005995750427246,
                        0.5004003643989563,
                    ],
                },
                "action": {
                    "min": [-0.025, -0.025, -0.025],
                    "max": [0.025, 0.025, 0.025],
                },
            },
            "repo_id": args.policy_repo_id,
            "storage_device": "cpu",
            "vision_encoder_name": "helper2424/resnet10",
            "freeze_vision_encoder": True,
            "image_encoder_hidden_dim": 32,
            "shared_encoder": True,
            "online_steps": 1000000,
            "online_buffer_capacity": 100000,
            "offline_buffer_capacity": 100000,
            "online_step_before_learning": 100,
            "policy_update_freq": 1,
            "discount": 0.97,
            "temperature_init": 0.01,
            "num_critics": 2,
            "num_subsample_critics": None,
            "critic_lr": 3e-4,
            "actor_lr": 3e-4,
            "temperature_lr": 3e-4,
            "critic_target_update_weight": 0.005,
            "utd_ratio": 2,
            "state_encoder_hidden_dim": 256,
            "latent_dim": 64,
            "target_entropy": None,
            "use_backup_entropy": True,
            "grad_clip_norm": 10.0,
            "num_discrete_actions": 3,
            "critic_network_kwargs": {
                "hidden_dims": [256, 256],
                "activate_final": True,
                "final_activation": None,
            },
            "actor_network_kwargs": {
                "hidden_dims": [256, 256],
                "activate_final": True,
            },
            "policy_kwargs": {
                "use_tanh_squash": True,
                "std_min": 1e-5,
                "std_max": 5,
                "init_final": 0.05,
            },
            "actor_learner_config": {
                "learner_host": args.learner_host,
                "learner_port": args.learner_port,
                "policy_parameters_push_frequency": 4,
            },
            "concurrency": {
                "actor": "threads",
                "learner": "threads",
            },
        },
    }


def write_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    extra_paths = [str(PROJECT_ROOT), str(LEROBOT_SRC)]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([current_pythonpath] if current_pythonpath else []))
    return env


def launch_processes(config_path: Path, actor_delay_seconds: float) -> int:
    env = make_env()
    learner_cmd = [sys.executable, "-m", "lerobot.rl.learner", "--config_path", str(config_path)]
    actor_cmd = [sys.executable, "-m", "lerobot.rl.actor", "--config_path", str(config_path)]

    learner = subprocess.Popen(learner_cmd, cwd=PROJECT_ROOT, env=env)
    actor = None

    try:
        time.sleep(actor_delay_seconds)
        actor = subprocess.Popen(actor_cmd, cwd=PROJECT_ROOT, env=env)
        actor_rc = actor.wait()
        learner_rc = learner.wait()
        return actor_rc or learner_rc
    except KeyboardInterrupt:
        return_code = 130
        if actor is not None and actor.poll() is None:
            actor.send_signal(signal.SIGINT)
        if learner.poll() is None:
            learner.send_signal(signal.SIGINT)
        return return_code
    finally:
        if actor is not None and actor.poll() is None:
            actor.kill()
        if learner.poll() is None:
            learner.kill()


def main() -> int:
    args = parse_args()
    config = make_train_config(args)
    write_config(args.config_path, config)

    print(f"Wrote HILSERL sim config to {args.config_path}")
    print("This config follows the Hugging Face gym_hil SAC example for PandaPickCube.")

    if not args.launch:
        print("\nNext steps:")
        print(f"  PYTHONPATH={PROJECT_ROOT}:{LEROBOT_SRC}:$PYTHONPATH \\")
        print(f"    python -m lerobot.rl.learner --config_path {args.config_path}")
        print(f"  PYTHONPATH={PROJECT_ROOT}:{LEROBOT_SRC}:$PYTHONPATH \\")
        print(f"    python -m lerobot.rl.actor --config_path {args.config_path}")
        return 0

    return launch_processes(args.config_path, args.actor_delay_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
