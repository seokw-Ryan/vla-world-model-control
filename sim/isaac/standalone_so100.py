"""Standalone Isaac Sim script for closed-loop VLA control of an SO-100 arm.

Usage:
    <ISAAC_SIM>/python.sh sim/isaac/standalone_so100.py --headless --instruction "pick up the red cube"
    <ISAAC_SIM>/python.sh sim/isaac/standalone_so100.py --max_steps 100

IMPORTANT: Must be run with Isaac Sim's python.sh.
SimulationApp must be created before any torch/omni imports.
"""

from __future__ import annotations

import argparse
import sys
import os


def log(msg: str) -> None:
    """Write to stderr so output is visible even when Isaac Sim captures stdout."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# ─── Phase 1: Parse args (before SimulationApp) ─────────────────────────────

parser = argparse.ArgumentParser(description="Standalone SO-100 VLA control in Isaac Sim")
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
                    default="configs/sim/so100_standalone.yaml",
                    help="Path to sim config YAML")
args = parser.parse_args()

# ─── Phase 2: Create SimulationApp (must happen before any omni/torch imports)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# ─── Phase 3: Import everything else ────────────────────────────────────────

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.sensors.camera import Camera
from isaacsim.asset.importer.urdf import _urdf

import omni.kit.commands

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
camera_cfg = sim_cfg.get("camera", {})
scene_cfg = sim_cfg.get("scene", {})
robot_cfg = sim_cfg.get("robot", {})

PHYSICS_DT = physics_cfg.get("dt", 1.0 / 60.0)
CONTROL_DECIMATION = physics_cfg.get("control_decimation", 12)

# SO-100 joint info
JOINT_NAMES = robot_cfg.get("joint_names", [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
])
NUM_ARM_JOINTS = 5  # first 5 are arm, last is gripper
DEFAULT_JOINT_POS = np.array(
    robot_cfg.get("default_joint_positions", [0.0, 1.0, -1.0, -0.5, 0.0, 0.5]),
    dtype=np.float32,
)

# ─── Phase 5: Build scene ───────────────────────────────────────────────────

world = World(physics_dt=PHYSICS_DT, stage_units_in_meters=1.0)

# Ground plane
world.scene.add_default_ground_plane()

# Import SO-100 directly from URDF onto the live stage (resolves mesh refs)
urdf_path = os.path.join(project_root, robot_cfg.get("urdf_path", "assets/so100/so100.urdf"))
robot_prim_path = robot_cfg.get("prim_path", "/World/SO100")
robot_position = np.array(robot_cfg.get("position", [0.0, 0.0, 0.40]))
robot_orientation = np.array(robot_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]))

if not os.path.exists(urdf_path):
    log(f"[SO100] ERROR: URDF not found at {urdf_path}")
    simulation_app.close()
    sys.exit(1)

import_config = _urdf.ImportConfig()
import_config.set_merge_fixed_joints(False)
import_config.set_fix_base(True)
import_config.set_make_default_prim(True)
import_config.set_create_physics_scene(False)  # World already creates one
import_config.set_default_drive_type(1)  # position drive
import_config.set_default_drive_strength(1e4)
import_config.set_default_position_drive_damping(1e3)
import_config.set_self_collision(False)

# Import URDF in-memory (dest_path="" = load onto current stage, not to file)
status, robot_prim = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path="",
)
log(f"[SO100] URDF imported, prim: {robot_prim}")

so100 = world.scene.add(
    Robot(
        prim_path=robot_prim,
        name="so100",
        position=robot_position,
        orientation=robot_orientation,
    )
)

table_cfg = scene_cfg.get("table", {})
if table_cfg.get("enabled", False):
    table_pos = table_cfg.get("position", [0.0, 0.0, 0.25])
    table_size = table_cfg.get("size", [0.5, 0.5, 0.5])
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

# Red cube
cube_cfg = scene_cfg.get("cube", {})
cube_pos = cube_cfg.get("position", [0.15, 0.0, 0.415])
cube_size = cube_cfg.get("size", 0.03)
cube_color = cube_cfg.get("color", [0.9, 0.1, 0.1])
cube_mass = cube_cfg.get("mass", 0.05)
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
cam_pos = camera_cfg.get("position", [0.25, 0.25, 0.9])
cam_target = camera_cfg.get("target", [0.15, 0.0, 0.45])
camera = Camera(
    prim_path=camera_cfg.get("prim_path", "/World/Camera"),
    resolution=(cam_res[0], cam_res[1]),
    position=np.array(cam_pos),
)

# ─── Phase 6: Initialize ────────────────────────────────────────────────────

world.reset()
camera.initialize()
camera.set_focal_length(1.5)

# Point camera at workspace
cam_forward = np.array(cam_target) - np.array(cam_pos)
cam_forward = cam_forward / np.linalg.norm(cam_forward)
world_up = np.array([0.0, 0.0, 1.0])
if abs(np.dot(cam_forward, world_up)) > 0.99:
    world_up = np.array([0.0, 1.0, 0.0])
cam_right = np.cross(cam_forward, world_up)
cam_right = cam_right / np.linalg.norm(cam_right)
cam_up = np.cross(cam_right, cam_forward)
rot_matrix = np.stack([cam_right, cam_up, -cam_forward], axis=1)
cam_rot = Rotation.from_matrix(rot_matrix)
cam_quat_xyzw = cam_rot.as_quat()
cam_quat_wxyz = np.array([cam_quat_xyzw[3], cam_quat_xyzw[0],
                           cam_quat_xyzw[1], cam_quat_xyzw[2]])
camera.set_world_pose(position=np.array(cam_pos), orientation=cam_quat_wxyz)

# Initialize robot
so100.initialize()

# Get joint indices for the arm and gripper
joint_names_in_robot = so100.dof_names
log(f"[SO100] Robot DOF names: {joint_names_in_robot}")
log(f"[SO100] Num DOFs: {so100.num_dof}")

# Map our config joint names to robot DOF indices
joint_indices = []
for name in JOINT_NAMES:
    if name in joint_names_in_robot:
        joint_indices.append(joint_names_in_robot.index(name))
    else:
        log(f"[SO100] WARNING: joint '{name}' not found in robot DOFs")
joint_indices = np.array(joint_indices)

arm_joint_indices = joint_indices[:NUM_ARM_JOINTS]
gripper_joint_idx = joint_indices[NUM_ARM_JOINTS] if len(joint_indices) > NUM_ARM_JOINTS else None

# Set default joint positions
so100.set_joint_positions(DEFAULT_JOINT_POS)

# Step a few times to let physics settle
for _ in range(10):
    world.step(render=True)

# ─── Phase 7: Articulation controller ──────────────────────────────────────

articulation_controller = so100.get_articulation_controller()

# ─── Phase 8: Load VLA model ────────────────────────────────────────────────

try:
    vla = OpenVLAWrapper(vla_config).load()
    log("[SO100] VLA model loaded.")
except Exception as e:
    log(f"[SO100] VLA load FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    world.stop()
    simulation_app.close()
    sys.exit(1)

# ─── Phase 9: Warm-up ──────────────────────────────────────────────────────

log("[SO100] Warming up camera...")
for i in range(20):
    world.step(render=True)
    rgba = camera.get_rgba()
    if rgba is not None:
        log(f"[SO100] Camera ready after {i + 1} warm-up steps.")
        break
else:
    log("[SO100] WARNING: Camera did not return data during warm-up.")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def vla_action_to_joint_deltas(action) -> tuple[np.ndarray, float]:
    """Convert VLA 7D action (dx,dy,dz,drx,dry,drz,gripper) to SO-100 joint deltas.

    The SO-100 uses direct joint position control (no IK). We map VLA's
    cartesian deltas to joint-space deltas using a simple Jacobian-like mapping:

    - action[0] (dx) -> shoulder_pan (base rotation)
    - action[1] (dy) -> shoulder_lift
    - action[2] (dz) -> elbow_flex
    - action[3] (drx) -> wrist_flex
    - action[4] (dry) -> wrist_roll
    - action[5:6] ignored (extra rotation dims)
    - action[6] -> gripper

    Scale factors are applied to map VLA output range to reasonable joint deltas.
    """
    JOINT_DELTA_SCALE = 0.1  # scale VLA deltas to small joint movements

    arm_deltas = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    arm_deltas[0] = action.delta_pos[0] * JOINT_DELTA_SCALE  # dx -> shoulder_pan
    arm_deltas[1] = action.delta_pos[1] * JOINT_DELTA_SCALE  # dy -> shoulder_lift
    arm_deltas[2] = action.delta_pos[2] * JOINT_DELTA_SCALE  # dz -> elbow_flex
    arm_deltas[3] = action.delta_rot[0] * JOINT_DELTA_SCALE  # drx -> wrist_flex
    arm_deltas[4] = action.delta_rot[1] * JOINT_DELTA_SCALE  # dry -> wrist_roll

    gripper_val = action.gripper
    return arm_deltas, gripper_val


# Joint limits from URDF
JOINT_LIMITS_LOWER = np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.0, 3.5, 0.0, 1.2, 3.14158], dtype=np.float32)
GRIPPER_OPEN = 1.5   # open position (radians)
GRIPPER_CLOSED = -0.1  # closed position (radians)

# ─── Phase 10: Control loop ─────────────────────────────────────────────────

log(f"[SO100] Starting control loop: instruction='{args.instruction}', "
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

    # 3. Convert to joint deltas
    arm_deltas, gripper_cmd = vla_action_to_joint_deltas(action)

    # 4. Get current joint positions and apply deltas
    current_joints = so100.get_joint_positions()
    current_arm = current_joints[arm_joint_indices]

    target_arm = current_arm + arm_deltas
    # Clamp to joint limits
    target_arm = np.clip(target_arm, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    # Gripper
    gripper_target = GRIPPER_OPEN if gripper_cmd > 0.5 else GRIPPER_CLOSED

    # 5. Build full joint target and apply
    target_joints = current_joints.copy()
    target_joints[arm_joint_indices] = target_arm
    if gripper_joint_idx is not None:
        target_joints[gripper_joint_idx] = gripper_target

    so100.set_joint_positions(target_joints)

    # 6. Step simulation (control decimation)
    for _ in range(CONTROL_DECIMATION):
        world.step(render=True)

    # 7. Log step info
    if step % 10 == 0:
        log(f"[Step {step}] arm_deltas={arm_deltas}, "
            f"gripper={'open' if gripper_cmd > 0.5 else 'closed'}, "
            f"arm_pos={target_arm}")

    if simulation_app.is_running() is False:
        log("[SO100] SimulationApp closed by user.")
        break

# ─── Phase 11: Shutdown ─────────────────────────────────────────────────────

log("[SO100] Shutting down...")
world.stop()
simulation_app.close()
log("[SO100] Done.")
