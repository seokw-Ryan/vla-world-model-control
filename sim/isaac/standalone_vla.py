"""Standalone Isaac Sim script for closed-loop VLA control of a Franka Panda.

Usage:
    python sim/isaac/standalone_vla.py --headless --instruction "pick up the red cube"
    python sim/isaac/standalone_vla.py --max_steps 100

IMPORTANT: SimulationApp must be created before any torch/omni imports.
"""

from __future__ import annotations

import argparse
import sys
import os

# ─── Phase 1: Parse args (before SimulationApp) ─────────────────────────────

parser = argparse.ArgumentParser(description="Standalone VLA control in Isaac Sim")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--instruction", type=str, default="pick up the red cube",
                    help="Natural language instruction for VLA")
parser.add_argument("--max_steps", type=int, default=200, help="Max control steps")
parser.add_argument("--model_path", type=str, default=None,
                    help="Override VLA model path")
parser.add_argument("--vla_config", type=str,
                    default="configs/vla/openvla_default.yaml",
                    help="Path to VLA config YAML")
parser.add_argument("--sim_config", type=str,
                    default="configs/sim/isaac_standalone.yaml",
                    help="Path to sim config YAML")
args = parser.parse_args()

# ─── Phase 2: Create SimulationApp (must happen before any omni/torch imports)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# ─── Phase 3: Import everything else (omni, torch, project modules) ──────────

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
    interface_config_loader,
)

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vla import OpenVLAConfig, OpenVLAWrapper

# ─── Phase 4: Load configs ──────────────────────────────────────────────────


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


vla_cfg = load_yaml(args.vla_config)
sim_cfg = load_yaml(args.sim_config)

# Build OpenVLAConfig from YAML
model_cfg = vla_cfg.get("model", {})
action_cfg = vla_cfg.get("action", {})
vla_config = OpenVLAConfig(
    model_path=args.model_path or model_cfg.get("path", "openvla/openvla-7b"),
    unnorm_key=model_cfg.get("unnorm_key", "bridge_orig"),
    prompt_template=model_cfg.get("prompt_template", "In: {instruction}\nOut:"),
    image_size=model_cfg.get("image_size", 224),
    dtype=model_cfg.get("dtype", "bfloat16"),
    load_in_4bit=model_cfg.get("load_in_4bit", True),
    position_scale=action_cfg.get("position_scale", 1.0),
    rotation_scale=action_cfg.get("rotation_scale", 1.0),
    gripper_threshold=action_cfg.get("gripper_threshold", 0.5),
)

physics_cfg = sim_cfg.get("physics", {})
robot_cfg = sim_cfg.get("robot", {})
camera_cfg = sim_cfg.get("camera", {})
scene_cfg = sim_cfg.get("scene", {})

PHYSICS_DT = physics_cfg.get("dt", 1.0 / 60.0)
CONTROL_DECIMATION = physics_cfg.get("control_decimation", 12)

# ─── Phase 5: Build scene ───────────────────────────────────────────────────

world = World(physics_dt=PHYSICS_DT, stage_units_in_meters=1.0)

# Ground plane
world.scene.add_default_ground_plane()

# Franka robot
assets_root = get_assets_root_path()
robot_usd = assets_root + robot_cfg.get("usd_path", "/Isaac/Robots/Franka/franka_alt_fingers.usd")
robot_prim_path = robot_cfg.get("prim_path", "/World/Franka")
add_reference_to_stage(usd_path=robot_usd, prim_path=robot_prim_path)
franka = world.scene.add(
    Robot(prim_path=robot_prim_path, name="franka")
)

# Table
table_cfg = scene_cfg.get("table", {})
table_pos = table_cfg.get("position", [0.5, 0.0, 0.25])
table_size = table_cfg.get("size", [0.6, 0.8, 0.5])
table_color = table_cfg.get("color", [0.4, 0.3, 0.2])
world.scene.add(
    FixedCuboid(
        prim_path=table_cfg.get("prim_path", "/World/Table"),
        name="table",
        position=np.array(table_pos),
        scale=np.array(table_size),
        color=np.array(table_color),
    )
)

# Red cube (dynamic, with rigid body physics)
cube_cfg = scene_cfg.get("cube", {})
cube_pos = cube_cfg.get("position", [0.5, 0.0, 0.525])
cube_size = cube_cfg.get("size", 0.04)
cube_color = cube_cfg.get("color", [0.9, 0.1, 0.1])
cube_mass = cube_cfg.get("mass", 0.1)
world.scene.add(
    DynamicCuboid(
        prim_path=cube_cfg.get("prim_path", "/World/Cube"),
        name="cube",
        position=np.array(cube_pos),
        scale=np.array([cube_size, cube_size, cube_size]),
        color=np.array(cube_color),
        mass=cube_mass,
    )
)

# Camera
cam_res = camera_cfg.get("resolution", [256, 256])
cam_pos = camera_cfg.get("position", [0.5, 0.0, 1.2])
cam_target = camera_cfg.get("target", [0.5, 0.0, 0.5])
camera = Camera(
    prim_path=camera_cfg.get("prim_path", "/World/Camera"),
    resolution=(cam_res[0], cam_res[1]),
    position=np.array(cam_pos),
)

# ─── Phase 6: Initialize ────────────────────────────────────────────────────

world.reset()
camera.initialize()
camera.set_focal_length(1.5)

# Point camera at table (compute look-at orientation)
cam_forward = np.array(cam_target) - np.array(cam_pos)
cam_forward = cam_forward / np.linalg.norm(cam_forward)
cam_up = np.array([0.0, 0.0, 1.0])
cam_right = np.cross(cam_forward, cam_up)
cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-8)
cam_up = np.cross(cam_right, cam_forward)
rot_matrix = np.stack([cam_right, cam_up, -cam_forward], axis=1)
cam_rot = Rotation.from_matrix(rot_matrix)
cam_quat_xyzw = cam_rot.as_quat()
cam_quat_wxyz = np.array([cam_quat_xyzw[3], cam_quat_xyzw[0], cam_quat_xyzw[1], cam_quat_xyzw[2]])
camera.set_world_pose(position=np.array(cam_pos), orientation=cam_quat_wxyz)

# Set default joint positions
default_joints = robot_cfg.get(
    "default_joint_positions",
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
)
franka.set_joint_positions(np.array(default_joints, dtype=np.float32))

# Step a few times to let physics settle
for _ in range(10):
    world.step(render=True)

# ─── Phase 7: Setup IK solver ───────────────────────────────────────────────

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
    "Franka"
)
kine_solver = LulaKinematicsSolver(**kinematics_config)
art_kine_solver = ArticulationKinematicsSolver(
    robot_articulation=franka,
    kinematics_solver=kine_solver,
    end_effector_frame_name=robot_cfg.get("end_effector_frame", "panda_hand"),
)

# ─── Phase 8: Load VLA model ────────────────────────────────────────────────

vla = OpenVLAWrapper(vla_config).load()

# ─── Phase 9: Warm-up (camera may return None on first frames) ──────────────

print("[Standalone] Warming up camera...")
for i in range(20):
    world.step(render=True)
    rgba = camera.get_rgba()
    if rgba is not None:
        print(f"[Standalone] Camera ready after {i + 1} warm-up steps.")
        break
else:
    print("[Standalone] WARNING: Camera did not return data during warm-up.")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def apply_delta_rotation(
    current_quat_wxyz: np.ndarray, delta_euler: np.ndarray
) -> np.ndarray:
    """Apply euler angle delta to a quaternion.

    Args:
        current_quat_wxyz: Current orientation as (w, x, y, z) quaternion (Isaac Sim convention).
        delta_euler: Delta rotation as (rx, ry, rz) euler angles in radians.

    Returns:
        New orientation as (w, x, y, z) quaternion.
    """
    # Isaac Sim uses (w, x, y, z), scipy uses (x, y, z, w)
    current_xyzw = np.array([
        current_quat_wxyz[1], current_quat_wxyz[2],
        current_quat_wxyz[3], current_quat_wxyz[0],
    ])
    current_rot = Rotation.from_quat(current_xyzw)
    delta_rot = Rotation.from_euler("xyz", delta_euler)
    new_rot = delta_rot * current_rot
    new_xyzw = new_rot.as_quat()
    # Convert back to (w, x, y, z)
    return np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]])


# ─── Phase 10: Control loop ─────────────────────────────────────────────────

print(f"[Standalone] Starting control loop: instruction='{args.instruction}', "
      f"max_steps={args.max_steps}")

for step in range(args.max_steps):
    # 1. Capture camera image
    rgba = camera.get_rgba()
    if rgba is None:
        print(f"[Step {step}] Camera returned None, stepping physics...")
        world.step(render=True)
        continue
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8) if rgba.max() <= 1.0 else rgba[:, :, :3].astype(np.uint8)

    # 2. VLA inference
    action = vla.predict(rgb, args.instruction)

    # 3. Get current EE pose
    ee_pos, ee_quat_wxyz = art_kine_solver.get_end_effector_pose()

    # 4. Compute target EE pose (apply deltas)
    target_pos = ee_pos + action.delta_pos
    target_quat = apply_delta_rotation(ee_quat_wxyz, action.delta_rot)

    # 5. Solve IK for target pose
    joint_targets, success = art_kine_solver.compute_inverse_kinematics(
        target_position=target_pos, target_orientation=target_quat
    )

    if success:
        # 6. Set arm joint targets (first 7 joints)
        arm_targets = joint_targets[:7]
        # 7. Set gripper targets (joints 7, 8)
        gripper_val = 0.04 if action.gripper > 0.5 else 0.0
        full_targets = np.concatenate([arm_targets, [gripper_val, gripper_val]])
        franka.set_joint_positions(full_targets)
    else:
        print(f"[Step {step}] IK solve failed, skipping action.")

    # 8. Step simulation (control decimation)
    for _ in range(CONTROL_DECIMATION):
        world.step(render=True)

    # 9. Print step info
    if step % 10 == 0:
        print(f"[Step {step}] pos_delta={action.delta_pos}, "
              f"gripper={'open' if action.gripper > 0.5 else 'closed'}, "
              f"ee_pos={ee_pos}")

    if simulation_app.is_running() is False:
        print("[Standalone] SimulationApp closed by user.")
        break

# ─── Phase 11: Shutdown ─────────────────────────────────────────────────────

print("[Standalone] Shutting down...")
world.stop()
simulation_app.close()
print("[Standalone] Done.")
