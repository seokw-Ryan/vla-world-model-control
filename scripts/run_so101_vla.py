#!/usr/bin/env python3
"""Run a LeRobot VLA policy on a real SO-101 follower arm.

This script is intended for real hardware control, not Isaac Sim.

Examples:
    python scripts/run_so101_vla.py \
        --policy_path lerobot/smolvla_base \
        --robot_port /dev/tty.usbmodem58760431541 \
        --robot_id my_so101 \
        --camera '{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
        --rename_map '{"observation.images.front":"observation.images.camera1"}' \
        --task "pick up the red block" \
        --steps 200

    python scripts/run_so101_vla.py \
        --policy_path /path/to/checkpoint/pretrained_model \
        --robot_port /dev/tty.usbmodem58760431541 \
        --robot_id my_so101 \
        --camera '{"front":{"type":"intelrealsense","serial_number_or_name":"233522074606","width":640,"height":480,"fps":30}}' \
        --task "put the lego brick in the box" \
        --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import torch

from isaaclab_arena_vla.utils import PROJECT_ROOT, add_lerobot_import_paths_to_sys_path

add_lerobot_import_paths_to_sys_path()

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    dataset_to_policy_features,
    hw_to_dataset_features,
)
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, validate_visual_features_consistency
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LeRobot VLA checkpoint on a real SO-101 arm.")
    parser.add_argument("--policy_path", type=str, required=True, help="HF model ID or local pretrained_model dir.")
    parser.add_argument("--robot_port", type=str, required=True, help="Serial port for the SO-101 follower arm.")
    parser.add_argument("--robot_id", type=str, required=True, help="Robot calibration ID used by LeRobot.")
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help=(
            "JSON object describing cameras. Keys are camera names, values are config dicts. "
            "Supports `opencv` and `intelrealsense`."
        ),
    )
    parser.add_argument(
        "--rename_map",
        type=str,
        default="{}",
        help=(
            "Optional JSON dict to rename observation keys to the names expected by the policy, "
            'for example \'{"observation.images.front":"observation.images.camera1"}\'.'
        ),
    )
    parser.add_argument("--task", type=str, default="", help="Language instruction passed to the policy.")
    parser.add_argument(
        "--robot_type",
        type=str,
        default="so101_follower",
        help="Robot type string passed to the policy for multi-embodiment checkpoints.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override policy device, e.g. cuda, cpu, mps.")
    parser.add_argument("--steps", type=int, default=200, help="Number of control steps to run.")
    parser.add_argument("--fps", type=float, default=10.0, help="Control frequency in Hz.")
    parser.add_argument(
        "--max_relative_target",
        type=float,
        default=10.0,
        help="Safety clamp on commanded joint delta per step, in normalized robot units.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print action summaries every N steps.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run perception and policy inference but do not send actions to the arm.",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow LeRobot to prompt for calibration if required.",
    )
    return parser.parse_args()


def parse_json_dict(raw: str, arg_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object.")
    return parsed


def build_camera_config(name: str, config: dict[str, Any]):
    camera_type = config.get("type")
    if camera_type == "opencv":
        if "index_or_path" not in config:
            raise ValueError(f"Camera '{name}' is missing required key `index_or_path`.")
        return OpenCVCameraConfig(
            index_or_path=config["index_or_path"],
            width=config.get("width"),
            height=config.get("height"),
            fps=config.get("fps"),
            color_mode=config.get("color_mode", "rgb"),
            rotation=config.get("rotation", 0),
            warmup_s=config.get("warmup_s", 1),
            fourcc=config.get("fourcc"),
            backend=config.get("backend", 0),
        )

    if camera_type == "intelrealsense":
        if "serial_number_or_name" not in config:
            raise ValueError(f"Camera '{name}' is missing required key `serial_number_or_name`.")
        return RealSenseCameraConfig(
            serial_number_or_name=config["serial_number_or_name"],
            width=config.get("width"),
            height=config.get("height"),
            fps=config.get("fps"),
            color_mode=config.get("color_mode", "rgb"),
            use_depth=config.get("use_depth", False),
            rotation=config.get("rotation", 0),
            warmup_s=config.get("warmup_s", 1),
        )

    raise ValueError(
        f"Camera '{name}' has unsupported type '{camera_type}'. Use 'opencv' or 'intelrealsense'."
    )


def build_camera_configs(camera_json: dict[str, Any]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for name, config in camera_json.items():
        if not isinstance(config, dict):
            raise ValueError(f"Camera '{name}' must map to a JSON object.")
        configs[name] = build_camera_config(name, config)
    return configs


def load_policy(policy_path: str, device_override: str | None, rename_map: dict[str, str]):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if device_override is not None:
        policy_cfg.device = device_override

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    return policy_cfg, policy, preprocessor, postprocessor


def rename_dataset_features(dataset_features: dict[str, dict], rename_map: dict[str, str]) -> dict[str, dict]:
    if not rename_map:
        return dataset_features
    renamed = {}
    for key, value in dataset_features.items():
        renamed[rename_map.get(key, key)] = value
    return renamed


def connect_cameras_only(robot: SO101Follower) -> None:
    for camera in robot.cameras.values():
        camera.connect()


def disconnect_cameras_only(robot: SO101Follower) -> None:
    for camera in robot.cameras.values():
        if camera.is_connected:
            camera.disconnect()


def make_dry_run_observation(robot: SO101Follower) -> dict[str, Any]:
    observation: dict[str, Any] = {joint_name: 0.0 for joint_name in robot.action_features}
    for camera_name, camera in robot.cameras.items():
        observation[camera_name] = camera.read_latest()
    return observation


def is_permission_denied(exc: BaseException) -> bool:
    return "Permission denied" in str(exc)


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        force=True,
    )
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    camera_json = parse_json_dict(args.camera, "--camera")
    rename_map = parse_json_dict(args.rename_map, "--rename_map")

    camera_configs = build_camera_configs(camera_json)

    policy_cfg, policy, preprocessor, postprocessor = load_policy(args.policy_path, args.device, rename_map)
    device = torch.device(policy_cfg.device)

    robot_cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=camera_configs,
        max_relative_target=args.max_relative_target,
    )
    robot = SO101Follower(robot_cfg)

    dataset_features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
    )
    provided_policy_features = dataset_to_policy_features(rename_dataset_features(dataset_features, rename_map))
    validate_visual_features_consistency(policy_cfg, provided_policy_features)

    logging.info("Policy type: %s", policy_cfg.type)
    logging.info("Policy device: %s", policy_cfg.device)
    logging.info("Robot port: %s", args.robot_port)
    logging.info("Camera names: %s", ", ".join(camera_configs))
    if rename_map:
        logging.info("Rename map: %s", rename_map)
    if args.dry_run:
        logging.warning("Dry-run enabled: actions will not be sent to the robot.")
    elif not os.access(args.robot_port, os.R_OK | os.W_OK):
        logging.warning(
            "Serial port %s is not accessible to the current user. "
            "You likely need dialout access before real robot control will work.",
            args.robot_port,
        )

    step_duration_s = 1.0 / args.fps
    robot_connected = False
    cameras_connected = False

    try:
        if args.dry_run:
            connect_cameras_only(robot)
            cameras_connected = True
        else:
            robot.connect(calibrate=args.calibrate)
            robot_connected = True
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        for step in range(args.steps):
            step_start = time.perf_counter()

            if args.dry_run:
                observation = make_dry_run_observation(robot)
            else:
                observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset_features, observation, prefix=OBS_STR)

            policy_action = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_cfg.use_amp,
                task=args.task,
                robot_type=args.robot_type,
            )
            robot_action = make_robot_action(policy_action, dataset_features)

            if not args.dry_run:
                sent_action = robot.send_action(robot_action)
            else:
                sent_action = robot_action

            if step == 0 or ((step + 1) % args.log_every == 0):
                logging.info("Step %d/%d action: %s", step + 1, args.steps, sent_action)

            elapsed = time.perf_counter() - step_start
            time.sleep(max(step_duration_s - elapsed, 0.0))

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as exc:
        if is_permission_denied(exc):
            logging.error(
                "Permission denied while opening hardware device: %s. "
                "For the arm, add this user to the `dialout` group or adjust udev rules. "
                "For cameras, add this user to the `video` group if needed.",
                exc,
            )
        raise
    finally:
        try:
            if robot_connected:
                robot.disconnect()
            elif cameras_connected:
                disconnect_cameras_only(robot)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Robot disconnect failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
