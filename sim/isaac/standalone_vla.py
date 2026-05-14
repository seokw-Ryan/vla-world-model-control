"""Standalone Isaac Sim script for closed-loop VLA control of a Franka Panda.

Usage:
    <ISAAC_SIM>/python.sh sim/isaac/standalone_vla.py --headless --instruction "pick up the red cube"
    <ISAAC_SIM>/python.sh sim/isaac/standalone_vla.py --max_steps 100

IMPORTANT: Must be run with Isaac Sim's python.sh.
SimulationApp must be created before any torch/omni imports.
"""

from __future__ import annotations

import argparse

from isaaclab_arena_vla.utils import (
    add_project_root_to_sys_path,
    build_openvla_config,
    load_yaml,
    stderr_log as log,
)


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

# ─── Phase 3: Import everything else ────────────────────────────────────────

import numpy as np
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver

project_root = add_project_root_to_sys_path()
from src.models.vla import OpenVLAWrapper

# ─── Phase 4: Load configs ──────────────────────────────────────────────────

sim_cfg = load_yaml(args.sim_config)
vla_config = build_openvla_config(args.vla_config, model_path=args.model_path)

physics_cfg = sim_cfg.get("physics", {})
camera_cfg = sim_cfg.get("camera", {})
scene_cfg = sim_cfg.get("scene", {})

PHYSICS_DT = physics_cfg.get("dt", 1.0 / 60.0)
CONTROL_DECIMATION = physics_cfg.get("control_decimation", 12)

# ─── Phase 5: Build scene ───────────────────────────────────────────────────

world = World(physics_dt=PHYSICS_DT, stage_units_in_meters=1.0)

# Ground plane
world.scene.add_default_ground_plane()

# Franka robot (auto-resolves correct USD path for Isaac Sim 5.x)
franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka")
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
# Use world Y as up hint when looking straight down (forward ~parallel to Z)
world_up = np.array([0.0, 0.0, 1.0])
if abs(np.dot(cam_forward, world_up)) > 0.99:
    world_up = np.array([0.0, 1.0, 0.0])
cam_right = np.cross(cam_forward, world_up)
cam_right = cam_right / np.linalg.norm(cam_right)
cam_up = np.cross(cam_right, cam_forward)
rot_matrix = np.stack([cam_right, cam_up, -cam_forward], axis=1)
cam_rot = Rotation.from_matrix(rot_matrix)
cam_quat_xyzw = cam_rot.as_quat()
cam_quat_wxyz = np.array([cam_quat_xyzw[3], cam_quat_xyzw[0], cam_quat_xyzw[1], cam_quat_xyzw[2]])
camera.set_world_pose(position=np.array(cam_pos), orientation=cam_quat_wxyz)

# Set default joint positions
default_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
franka.set_joint_positions(np.array(default_joints, dtype=np.float32))

# Step a few times to let physics settle
for _ in range(10):
    world.step(render=True)

# ─── Phase 7: Setup IK solver ───────────────────────────────────────────────

ik_solver = KinematicsSolver(franka)
articulation_controller = franka.get_articulation_controller()

# ─── Phase 8: Load VLA model ────────────────────────────────────────────────

try:
    vla = OpenVLAWrapper(vla_config).load()
    log("[Standalone] VLA model loaded.")
except Exception as e:
    log(f"[Standalone] VLA load FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    world.stop()
    simulation_app.close()
    sys.exit(1)

# ─── Phase 9: Warm-up (camera may return None on first frames) ──────────────

log("[Standalone] Warming up camera...")
for i in range(20):
    world.step(render=True)
    rgba = camera.get_rgba()
    if rgba is not None:
        log(f"[Standalone] Camera ready after {i + 1} warm-up steps.")
        break
else:
    log("[Standalone] WARNING: Camera did not return data during warm-up.")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def apply_delta_rotation(
    current_rot_matrix: np.ndarray, delta_euler: np.ndarray
) -> np.ndarray:
    """Apply euler angle delta to a rotation matrix.

    Args:
        current_rot_matrix: Current orientation as (3, 3) rotation matrix
            (from KinematicsSolver.compute_end_effector_pose).
        delta_euler: Delta rotation as (rx, ry, rz) euler angles in radians.

    Returns:
        New orientation as (w, x, y, z) quaternion (Isaac Sim convention).
    """
    current_rot = Rotation.from_matrix(current_rot_matrix)
    delta_rot = Rotation.from_euler("xyz", delta_euler)
    new_rot = delta_rot * current_rot
    new_xyzw = new_rot.as_quat()
    return np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]])


# ─── Phase 10: Control loop ─────────────────────────────────────────────────

log(f"[Standalone] Starting control loop: instruction='{args.instruction}', "
    f"max_steps={args.max_steps}")

for step in range(args.max_steps):
    # 1. Capture camera image
    rgba = camera.get_rgba()
    if rgba is None:
        world.step(render=True)
        continue
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8) if rgba.max() <= 1.0 else rgba[:, :, :3].astype(np.uint8)

    # 2. VLA inference
    action = vla.predict(rgb, args.instruction)

    # 3. Get current EE pose (rotation is a 3x3 matrix in 5.x)
    ee_pos, ee_rot_matrix = ik_solver.compute_end_effector_pose()

    # 4. Compute target EE pose (apply deltas)
    target_pos = ee_pos + action.delta_pos
    target_quat = apply_delta_rotation(ee_rot_matrix, action.delta_rot)

    # 5. Solve IK for target pose (returns ArticulationAction with 7 arm joints)
    ik_result, success = ik_solver.compute_inverse_kinematics(
        target_position=target_pos, target_orientation=target_quat
    )

    if success:
        # 6. Build full joint targets: 7 arm joints from IK + 2 gripper joints
        arm_positions = np.array(ik_result.joint_positions, dtype=np.float32)
        gripper_val = 0.04 if action.gripper > 0.5 else 0.0
        full_positions = np.concatenate([arm_positions, [gripper_val, gripper_val]])
        franka.set_joint_positions(full_positions)
    else:
        log(f"[Step {step}] IK solve failed, skipping action.")

    # 7. Step simulation (control decimation)
    for _ in range(CONTROL_DECIMATION):
        world.step(render=True)

    # 8. Log step info
    if step % 10 == 0:
        log(f"[Step {step}] pos_delta={action.delta_pos}, "
            f"gripper={'open' if action.gripper > 0.5 else 'closed'}, "
            f"ee_pos={ee_pos}")

    if simulation_app.is_running() is False:
        log("[Standalone] SimulationApp closed by user.")
        break

# ─── Phase 11: Shutdown ─────────────────────────────────────────────────────

log("[Standalone] Shutting down...")
world.stop()
simulation_app.close()
log("[Standalone] Done.")
