#!/usr/bin/env python3
"""Run a trained SmolVLA checkpoint on a real SO-100 / SO-101 follower arm.

This script is intended to be run inside the LeRobot conda environment.

Example:
    conda run -n lerobot python scripts/run_smolvla.py \
        --policy_path outputs/smolvla_rl/20260424_120000/checkpoints/episode_00025 \
        --robot_type so100_follower \
        --robot_port /dev/ttyACM0 \
        --robot_id my_so100 \
        --task "pick up the red cube"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from vla_world_model_control.shared import PROJECT_ROOT, add_lerobot_import_paths_to_sys_path

add_lerobot_import_paths_to_sys_path()

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
    SO101Follower,
    SO101FollowerConfig,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.errors import DeviceNotConnectedError


SO_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
SO_ARM_JOINT_NAMES = SO_JOINT_NAMES[:5]
SO_ARM_LIMITS_LOWER = np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158], dtype=np.float32)
SO_ARM_LIMITS_UPPER = np.array([2.0, 3.5, 0.0, 1.2, 3.14158], dtype=np.float32)
SIM_ARM_DELTA_SCALE = 0.1
SIM_GRIPPER_OPEN = 1.5
SIM_GRIPPER_CLOSED = -0.1
HW_GRIPPER_OPEN = 100.0
HW_GRIPPER_CLOSED = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained SmolVLA checkpoint on a real SO follower arm.")
    parser.add_argument("--policy_path", required=True, help="Path to a SmolVLA checkpoint directory.")
    parser.add_argument(
        "--robot_type",
        choices=("so100_follower", "so101_follower"),
        default="so100_follower",
        help="LeRobot robot type for the physical follower arm.",
    )
    parser.add_argument("--robot_port", required=True, help="Serial port for the robot, for example /dev/ttyACM0.")
    parser.add_argument("--robot_id", required=True, help="Calibration id for the robot.")
    parser.add_argument("--task", default="pick up the red cube", help="Instruction passed to the policy.")
    parser.add_argument("--steps", type=int, default=200, help="Number of control loop iterations.")
    parser.add_argument("--fps", type=float, default=5.0, help="Control frequency in Hz. Defaults to the sim rate.")
    parser.add_argument("--log_every", type=int, default=10, help="Print action summaries every N steps.")
    parser.add_argument("--device", default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument(
        "--camera_type",
        choices=("opencv", "intelrealsense"),
        default="opencv",
        help="Camera backend to use on the real setup.",
    )
    parser.add_argument(
        "--camera_device",
        default=os.environ.get("SO100_CAMERA_DEVICE", "/dev/video1"),
        help="OpenCV camera index or path. The default matches the detected local OpenCV device.",
    )
    parser.add_argument(
        "--camera_serial",
        default=None,
        help="RealSense serial number or device name when using --camera_type intelrealsense.",
    )
    parser.add_argument("--camera_width", type=int, default=640, help="Camera width in pixels.")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera height in pixels.")
    parser.add_argument("--camera_fps", type=int, default=30, help="Camera stream FPS.")
    parser.add_argument(
        "--camera_rotation",
        type=int,
        choices=(0, 90, 180, -90),
        default=0,
        help="Rotation to apply to the real camera stream.",
    )
    parser.add_argument(
        "--max_relative_target",
        type=float,
        default=10.0,
        help="Safety clamp in LeRobot's follower driver, in degrees.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Read robot state and camera, run inference, but do not send actions.",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow LeRobot to prompt for calibration if needed.",
    )
    parser.add_argument(
        "--trace_steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a per-step JSONL trace with policy inputs and outputs.",
    )
    parser.add_argument(
        "--trace_image_interval",
        type=int,
        default=1,
        help="Save the real camera image every N steps to the trace directory. Disabled when 0.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Optional directory for per-step trace logs. Defaults to outputs/run_smolvla/<timestamp>.",
    )
    return parser.parse_args()


def parse_camera_device(raw: str) -> int | str:
    if raw.isdigit():
        return int(raw)
    return str(Path(raw))


def build_camera_config(args: argparse.Namespace):
    if args.camera_type == "opencv":
        return OpenCVCameraConfig(
            index_or_path=parse_camera_device(args.camera_device),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            color_mode="rgb",
            rotation=Cv2Rotation(args.camera_rotation),
            warmup_s=1,
            fourcc="YUYV",
        )

    if not args.camera_serial:
        raise ValueError("--camera_serial is required when using --camera_type intelrealsense.")
    return RealSenseCameraConfig(
        serial_number_or_name=args.camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        color_mode="rgb",
        rotation=Cv2Rotation(args.camera_rotation),
        use_depth=False,
        warmup_s=1,
    )


def summarize_image(image: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
    }


def maybe_save_trace_image(image: np.ndarray, image_dir: Path | None, step_idx: int) -> str | None:
    if image_dir is None:
        return None
    import cv2

    image_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_path = image_dir / f"step{step_idx:05d}.png"
    cv2.imwrite(str(image_path), image_bgr)
    return str(image_path)


def write_trace_line(trace_file, payload: dict[str, object]) -> None:
    trace_file.write(json.dumps(payload) + "\n")
    trace_file.flush()


def build_robot(args: argparse.Namespace):
    camera_config = {"front": build_camera_config(args)}
    if args.robot_type == "so100_follower":
        config = SO100FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras=camera_config,
            max_relative_target=args.max_relative_target,
            use_degrees=True,
        )
        return SO100Follower(config)

    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=camera_config,
        max_relative_target=args.max_relative_target,
        use_degrees=True,
    )
    return SO101Follower(config)


def find_existing_calibration_ids(calibration_dir: Path) -> list[str]:
    if not calibration_dir.exists():
        return []
    return sorted(path.stem for path in calibration_dir.glob("*.json"))


def load_policy(
    policy_path: str,
    device_override: str | None,
    camera_height: int,
    camera_width: int,
) -> tuple[SmolVLAPolicy, torch.device, AutoProcessor]:
    policy = SmolVLAPolicy.from_pretrained(policy_path, strict=False)
    policy.config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, camera_height, camera_width),
        ),
    }
    policy.config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    device = torch.device(device_override or policy.config.device)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
    return policy, device, processor


def observation_to_state(observation: dict[str, np.ndarray | float]) -> np.ndarray:
    joint_values_deg = np.array([float(observation[f"{joint}.pos"]) for joint in SO_ARM_JOINT_NAMES], dtype=np.float32)
    gripper_percent = float(observation["gripper.pos"])
    gripper_sim = np.float32(np.interp(gripper_percent, [0.0, 100.0], [SIM_GRIPPER_CLOSED, SIM_GRIPPER_OPEN]))
    return np.concatenate([np.deg2rad(joint_values_deg), np.array([gripper_sim], dtype=np.float32)])


def observation_to_image(observation: dict[str, np.ndarray | float]) -> np.ndarray:
    image = np.asarray(observation["front"])
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image from camera, got shape {image.shape}")
    return np.transpose(image, (2, 0, 1)).astype(np.uint8, copy=False)


def build_policy_batch(
    observation: dict[str, np.ndarray | float],
    task_tokens: torch.Tensor,
    task_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image = torch.from_numpy(observation_to_image(observation)).float().unsqueeze(0).to(device) / 255.0
    state = torch.from_numpy(observation_to_state(observation)).float().unsqueeze(0).to(device)
    return {
        "observation.images.front": image,
        "observation.state": state,
        OBS_LANGUAGE_TOKENS: task_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: task_mask.to(dtype=torch.bool),
    }


def policy_action_to_robot_action(policy_action: np.ndarray, current_state: np.ndarray) -> dict[str, float]:
    action = np.clip(np.asarray(policy_action, dtype=np.float32), -1.0, 1.0)
    current_arm = current_state[:5]
    target_arm = np.clip(
        current_arm + action[:5] * SIM_ARM_DELTA_SCALE,
        SO_ARM_LIMITS_LOWER,
        SO_ARM_LIMITS_UPPER,
    )
    target_arm_deg = np.rad2deg(target_arm)
    gripper_target = HW_GRIPPER_OPEN if float(action[5]) > 0.0 else HW_GRIPPER_CLOSED
    robot_action = {f"{joint}.pos": float(target_arm_deg[idx]) for idx, joint in enumerate(SO_ARM_JOINT_NAMES)}
    robot_action["gripper.pos"] = float(gripper_target)
    return robot_action


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    logging.info("Loading SmolVLA checkpoint from %s", args.policy_path)
    policy, device, processor = load_policy(args.policy_path, args.device, args.camera_height, args.camera_width)
    tokenizer = processor.tokenizer

    task_text = args.task if args.task.endswith("\n") else f"{args.task}\n"
    tokenized = tokenizer([task_text], return_tensors="pt", padding="max_length", truncation=True, max_length=48)
    task_tokens = tokenized["input_ids"].to(device)
    task_mask = tokenized["attention_mask"].to(device=device, dtype=torch.bool)

    robot = build_robot(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir is not None else PROJECT_ROOT / "outputs" / "run_smolvla" / timestamp
    trace_path = log_dir / "step_trace.jsonl"
    trace_image_dir = log_dir / "step_images" if args.trace_image_interval > 0 else None
    trace_file = None
    logging.info(
        "Robot=%s port=%s camera=%s %sx%s@%s",
        args.robot_type,
        args.robot_port,
        args.camera_type,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )
    if args.trace_steps:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Step trace: %s", trace_path)
        if trace_image_dir is not None:
            logging.info("Step images: %s", trace_image_dir)
    if args.dry_run:
        logging.warning("Dry-run enabled: actions will not be sent to the robot.")
    if not args.calibrate and not robot.calibration_fpath.is_file():
        known_ids = find_existing_calibration_ids(robot.calibration_dir)
        known_ids_msg = ", ".join(known_ids) if known_ids else "(none found)"
        logging.error(
            "Calibration file not found for robot_id=%s at %s. Existing calibration ids in %s: %s. "
            "Run again with --calibrate or use a robot_id that already has calibration.",
            args.robot_id,
            robot.calibration_fpath,
            robot.calibration_dir,
            known_ids_msg,
        )
        return 1

    step_duration_s = 1.0 / args.fps
    try:
        if args.trace_steps:
            trace_file = trace_path.open("a", encoding="utf-8")
        robot.connect(calibrate=args.calibrate)
        policy.reset()

        for step in range(args.steps):
            step_start = time.perf_counter()
            observation = robot.get_observation()
            current_state = observation_to_state(observation)
            image_hwc = np.asarray(observation["front"])
            batch = build_policy_batch(observation, task_tokens, task_mask, device)

            with torch.inference_mode():
                action_tensor = policy.select_action(batch)
            policy_action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            robot_action = policy_action_to_robot_action(policy_action, current_state)

            if trace_file is not None:
                image_path = None
                if args.trace_image_interval > 0 and (step % args.trace_image_interval == 0):
                    image_path = maybe_save_trace_image(image_hwc, trace_image_dir, step)
                write_trace_line(
                    trace_file,
                    {
                        "step": step,
                        "task": args.task,
                        "state_radians": current_state.astype(float).tolist(),
                        "state_degrees": np.rad2deg(current_state[:5]).astype(float).tolist(),
                        "gripper_sim": float(current_state[5]),
                        "image": summarize_image(image_hwc),
                        "image_path": image_path,
                        "policy_action": policy_action.astype(float).tolist(),
                        "robot_action": {k: float(v) for k, v in robot_action.items()},
                        "dry_run": bool(args.dry_run),
                    },
                )

            if not args.dry_run:
                robot.send_action(robot_action)

            if step == 0 or ((step + 1) % args.log_every == 0):
                logging.info(
                    "Step %d/%d sim_action=%s robot_action=%s",
                    step + 1,
                    args.steps,
                    np.round(policy_action, 3).tolist(),
                    {k: round(v, 2) for k, v in robot_action.items()},
                )

            elapsed = time.perf_counter() - step_start
            time.sleep(max(step_duration_s - elapsed, 0.0))

    except DeviceNotConnectedError as exc:
        logging.error("Failed to connect to robot or camera: %s", exc)
        return 1
    except RuntimeError as exc:
        if "has no calibration registered" in str(exc):
            known_ids = find_existing_calibration_ids(robot.calibration_dir)
            known_ids_msg = ", ".join(known_ids) if known_ids else "(none found)"
            logging.error(
                "Robot connected, but calibration is missing for robot_id=%s at %s. "
                "Existing calibration ids in %s: %s. "
                "Run again with --calibrate or use a matching robot_id.",
                args.robot_id,
                robot.calibration_fpath,
                robot.calibration_dir,
                known_ids_msg,
            )
            return 1
        raise
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        try:
            if trace_file is not None:
                trace_file.close()
            if robot.is_connected:
                robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Robot disconnect failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
