"""Standalone Isaac Sim script for closed-loop SmolVLA control of an SO-100 arm.

Usage:
    <ISAAC_SIM>/python.sh sim/isaac/standalone_so100.py --headless --task "pick up the red cube"
    <ISAAC_SIM>/python.sh sim/isaac/standalone_so100.py --max_steps 100 --policy_path lerobot/smolvla_base

IMPORTANT: Must be run with Isaac Sim's python.sh.
SimulationApp must be created before any torch/omni imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from vla_world_model_control.shared import (
    PROJECT_ROOT,
    add_lerobot_import_paths_to_sys_path,
    add_project_root_to_sys_path,
    load_yaml,
    stderr_log as log,
)


# ─── Phase 1: Parse args (before SimulationApp) ─────────────────────────────

parser = argparse.ArgumentParser(description="Standalone SO-100 SmolVLA control in Isaac Sim")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--task", "--instruction", dest="task", type=str,
                    default="pick up the red cube",
                    help="Natural language instruction passed to SmolVLA")
parser.add_argument("--max_steps", type=int, default=200, help="Max control steps")
parser.add_argument("--policy_path", type=str, default="lerobot/smolvla_base",
                    help="SmolVLA checkpoint dir or HF repo id")
parser.add_argument("--policy_device", type=str, default=None,
                    help="Torch device for the policy (cuda / cpu). Default: policy config.")
parser.add_argument("--sim_config", type=str,
                    default="configs/sim/so100_standalone.yaml",
                    help="Path to sim config YAML")
parser.add_argument("--camera_eye", type=float, nargs=3, default=None,
                    help="Optional camera eye override as x y z in world coordinates")
parser.add_argument("--camera_target", type=float, nargs=3, default=None,
                    help="Optional camera target override as x y z in world coordinates")
parser.add_argument("--camera_orientation_wxyz", type=float, nargs=4, default=None,
                    help="Optional camera orientation override as w x y z")
parser.add_argument("--camera_focal_length", type=float, default=None,
                    help="Optional focal length override")
parser.add_argument("--log_dir", type=str, default=None,
                    help="Per-step trace directory. Default: outputs/standalone_so100/<timestamp>")
parser.add_argument("--trace_image_interval", type=int, default=1,
                    help="Save the policy input image every N steps. 0 disables image dumps.")
args = parser.parse_args()

# ─── Phase 2: Create SimulationApp (must happen before any omni/torch imports)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# ─── Phase 3: Import everything else ────────────────────────────────────────

import numpy as np
import torch
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.viewports import set_active_viewport_camera
from isaacsim.sensors.camera import Camera
from isaacsim.asset.importer.urdf import _urdf

import omni.kit.commands

project_root = add_project_root_to_sys_path()
add_lerobot_import_paths_to_sys_path()

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor

# ─── Phase 4: Load configs ──────────────────────────────────────────────────


def author_root_pose(stage, prim_path: str, position: np.ndarray, orientation_wxyz: np.ndarray) -> None:
    """Write the imported robot prim's root xform so the Stage UI matches runtime placement."""
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*[float(v) for v in position.tolist()])
    )
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(
            float(orientation_wxyz[0]),
            Gf.Vec3d(
                float(orientation_wxyz[1]),
                float(orientation_wxyz[2]),
                float(orientation_wxyz[3]),
            ),
        )
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))

sim_cfg = load_yaml(args.sim_config)

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
robot_position = np.array(robot_cfg.get("position", [0.0, 0.0, 0.42]))
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
# URDF import lands the articulation at /so_arm100 with an authored root translate of zero.
# Write the root xform on that imported prim directly so the Property panel reflects the
# actual robot placement instead of showing a misleading 0,0,0 transform.
author_root_pose(world.stage, robot_prim, robot_position, robot_orientation)

so100 = world.scene.add(
    Robot(
        prim_path=robot_prim,
        name="so100",
    )
)

table_cfg = scene_cfg.get("table", {})
if table_cfg.get("enabled", False):
    table_pos = table_cfg.get("position", [0.0, 0.0, 0.25])
    table_size = table_cfg.get("size", [0.5, 0.5, 0.5])
    table_color = table_cfg.get("color", [0.4, 0.3, 0.2])
    table_top_z = float(table_pos[2] + table_size[2] * 0.5)
    clearance = float(robot_position[2] - table_top_z)
    if clearance <= 0.0:
        log(
            "[SO100] WARNING: robot base z is at or below the tabletop top surface "
            f"(robot_z={float(robot_position[2]):.3f}, table_top_z={table_top_z:.3f}, "
            f"clearance={clearance:.3f})."
        )
    else:
        log(
            "[SO100] Table clearance check passed: "
            f"robot_z={float(robot_position[2]):.3f}, table_top_z={table_top_z:.3f}, "
            f"clearance={clearance:.3f}"
        )
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
cam_pos = args.camera_eye if args.camera_eye is not None else camera_cfg.get("position", [0.25, 0.25, 0.9])
cam_target = args.camera_target if args.camera_target is not None else camera_cfg.get("target", [0.15, 0.0, 0.45])
cam_euler_deg = camera_cfg.get("orientation_euler_xyz_deg")
if args.camera_orientation_wxyz is not None:
    cam_orientation_wxyz = np.array(args.camera_orientation_wxyz, dtype=np.float32)
elif camera_cfg.get("orientation_wxyz") is not None:
    cam_orientation_wxyz = np.array(camera_cfg["orientation_wxyz"], dtype=np.float32)
else:
    cam_orientation_wxyz = None
cam_focal_length = float(args.camera_focal_length or camera_cfg.get("focal_length", 1.5))
camera = Camera(
    prim_path=camera_cfg.get("prim_path", "/World/Camera"),
    resolution=(cam_res[0], cam_res[1]),
    position=np.array(cam_pos),
)

# ─── Phase 6: Initialize ────────────────────────────────────────────────────

world.reset()
camera.initialize()
camera.set_focal_length(cam_focal_length)


def compute_lookat_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute wxyz quaternion in Isaac's world camera axes (+X forward, +Z up)."""
    from isaacsim.core.utils.rotations import gf_quat_to_np_array, lookat_to_quatf

    eye_gf = Gf.Vec3f(float(eye[0]), float(eye[1]), float(eye[2]))
    target_gf = Gf.Vec3f(float(target[0]), float(target[1]), float(target[2]))
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    up = Gf.Vec3f(0.0, 0.0, 1.0)
    if abs(float(np.dot(forward, np.array([0.0, 0.0, 1.0])))) > 0.99:
        up = Gf.Vec3f(0.0, 1.0, 0.0)
    return gf_quat_to_np_array(lookat_to_quatf(target_gf, eye_gf, up)).astype(np.float32)


def euler_xyz_deg_to_wxyz(euler_deg) -> np.ndarray:
    """USD XYZ Euler degrees → wxyz quaternion (intrinsic XYZ, matches the Property panel)."""
    rot = Rotation.from_euler("xyz", np.asarray(euler_deg, dtype=np.float64), degrees=True)
    xyzw = rot.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


if cam_orientation_wxyz is not None:
    camera_orientation = cam_orientation_wxyz
    camera_axes = "world"
elif cam_euler_deg is not None:
    camera_orientation = euler_xyz_deg_to_wxyz(cam_euler_deg)
    camera_axes = "usd"
else:
    camera_orientation = compute_lookat_quat(
        np.array(cam_pos, dtype=np.float32),
        np.array(cam_target, dtype=np.float32),
    )
    camera_axes = "world"
camera.set_world_pose(
    position=np.array(cam_pos, dtype=np.float32),
    orientation=camera_orientation,
    camera_axes=camera_axes,
)
try:
    set_active_viewport_camera(camera_cfg.get("prim_path", "/World/Camera"))
    log(f"[SO100] Active viewport camera set to {camera_cfg.get('prim_path', '/World/Camera')}")
except Exception as exc:
    log(f"[SO100] Failed to switch active viewport camera: {exc}")

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

# ─── Phase 8: Load SmolVLA policy ──────────────────────────────────────────

try:
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    device = torch.device(args.policy_device or policy.config.device)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
    tokenizer = processor.tokenizer
    log(f"[SO100] SmolVLA loaded from {args.policy_path} on {device}.")
except Exception as e:
    log(f"[SO100] SmolVLA load FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    world.stop()
    simulation_app.close()
    sys.exit(1)

task_text = args.task if args.task.endswith("\n") else f"{args.task}\n"
tokenized = tokenizer(
    [task_text],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=48,
)
task_tokens = tokenized["input_ids"].to(device)
task_mask = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
policy.reset()

# Discover which image observation keys this checkpoint expects (varies between
# the base checkpoint, which uses camera1/2/3, and fine-tuned ones using "front").
policy_image_keys = [
    key for key in policy.config.input_features.keys()
    if key.startswith("observation.images.")
]
if not policy_image_keys:
    log("[SO100] WARNING: policy has no image input features; falling back to observation.images.front")
    policy_image_keys = ["observation.images.front"]
log(f"[SO100] Policy image keys: {policy_image_keys}")

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

# SmolVLA outputs normalized actions in [-1, 1]; scale to small joint deltas.
SIM_ARM_DELTA_SCALE = 0.1
GRIPPER_OPEN = 1.5   # open position (radians)
GRIPPER_CLOSED = -0.1  # closed position (radians)

# Joint limits from URDF
JOINT_LIMITS_LOWER = np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.0, 3.5, 0.0, 1.2, 3.14158], dtype=np.float32)


def rgba_to_rgb_uint8(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    return rgb


def image_hash(rgb_uint8: np.ndarray) -> str:
    return hashlib.md5(rgb_uint8.tobytes()).hexdigest()[:8]


def build_policy_batch(rgb_uint8: np.ndarray, current_state: np.ndarray) -> dict[str, torch.Tensor]:
    image_chw = np.transpose(rgb_uint8, (2, 0, 1))
    image_tensor = torch.from_numpy(image_chw).float().unsqueeze(0).to(device) / 255.0
    state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(device)
    batch: dict[str, torch.Tensor] = {
        "observation.state": state_tensor,
        "observation.language.tokens": task_tokens,
        "observation.language.attention_mask": task_mask,
    }
    # The single physical camera is fed into every image slot the policy expects.
    for key in policy_image_keys:
        batch[key] = image_tensor
    return batch


# ─── Phase 10: Set up trace logging ────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "outputs" / "standalone_so100" / timestamp
log_dir.mkdir(parents=True, exist_ok=True)
trace_path = log_dir / "step_trace.jsonl"
image_dir = log_dir / "step_images" if args.trace_image_interval > 0 else None
if image_dir is not None:
    image_dir.mkdir(parents=True, exist_ok=True)
trace_file = trace_path.open("w", encoding="utf-8")
log(f"[SO100] Trace log: {trace_path}")
if image_dir is not None:
    log(f"[SO100] Step images: {image_dir} (every {args.trace_image_interval} step(s))")


def save_image(rgb_uint8: np.ndarray, step_idx: int) -> str | None:
    if image_dir is None:
        return None
    try:
        from PIL import Image
        path = image_dir / f"step{step_idx:05d}.png"
        Image.fromarray(rgb_uint8, "RGB").save(path)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        log(f"[SO100] Image save failed at step {step_idx}: {exc}")
        return None


# ─── Phase 11: Control loop ────────────────────────────────────────────────

log(
    f"[SO100] Camera pose: eye={tuple(float(v) for v in cam_pos)}, "
    f"orientation_wxyz={tuple(float(v) for v in camera_orientation)}, "
    f"target={tuple(float(v) for v in cam_target)}, focal_length={cam_focal_length}"
)
log(f"[SO100] Starting control loop: task='{args.task}', max_steps={args.max_steps}")

for step in range(args.max_steps):
    # 1. Capture camera image
    rgba = camera.get_rgba()
    if rgba is None:
        world.step(render=True)
        continue
    rgb = rgba_to_rgb_uint8(rgba)
    img_hash = image_hash(rgb)

    # 2. Build state: 5 arm joints (radians) + 1 gripper (radians, sim range)
    current_joints = so100.get_joint_positions()
    current_arm = np.array(current_joints[arm_joint_indices], dtype=np.float32)
    current_gripper = float(current_joints[gripper_joint_idx]) if gripper_joint_idx is not None else 0.0
    current_state = np.concatenate([current_arm, np.array([current_gripper], dtype=np.float32)])

    # 3. SmolVLA inference
    batch = build_policy_batch(rgb, current_state)
    with torch.inference_mode():
        action_tensor = policy.select_action(batch)
    raw_action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    clipped_action = np.clip(raw_action, -1.0, 1.0)

    # 4. Map to joint deltas
    arm_deltas = clipped_action[:NUM_ARM_JOINTS] * SIM_ARM_DELTA_SCALE
    gripper_cmd = float(clipped_action[NUM_ARM_JOINTS]) if clipped_action.size > NUM_ARM_JOINTS else 0.0
    gripper_target = GRIPPER_OPEN if gripper_cmd > 0.0 else GRIPPER_CLOSED

    target_arm = np.clip(current_arm + arm_deltas, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    target_joints = np.array(current_joints, dtype=np.float32, copy=True)
    target_joints[arm_joint_indices] = target_arm
    if gripper_joint_idx is not None:
        target_joints[gripper_joint_idx] = gripper_target
    so100.set_joint_positions(target_joints)

    # 5. Step simulation (control decimation)
    for _ in range(CONTROL_DECIMATION):
        world.step(render=True)

    # 6. Per-step trace logging
    image_path = None
    if image_dir is not None and (step % args.trace_image_interval == 0):
        image_path = save_image(rgb, step)
    trace_file.write(json.dumps({
        "step": step,
        "task": args.task,
        "image_hash": img_hash,
        "image_path": image_path,
        "raw_action": raw_action.astype(float).tolist(),
        "clipped_action": clipped_action.astype(float).tolist(),
        "arm_deltas": arm_deltas.astype(float).tolist(),
        "gripper_cmd": gripper_cmd,
        "gripper_target": gripper_target,
        "current_arm": current_arm.astype(float).tolist(),
        "current_gripper": current_gripper,
        "target_arm": target_arm.astype(float).tolist(),
    }) + "\n")
    trace_file.flush()

    # 7. Stderr summary every 10 steps + step 0
    if step == 0 or step % 10 == 0:
        log(
            f"[Step {step}] img={img_hash} raw_action={np.round(raw_action, 3).tolist()} "
            f"arm_deltas={np.round(arm_deltas, 4).tolist()} "
            f"gripper={'open' if gripper_cmd > 0.0 else 'closed'} "
            f"target_arm={np.round(target_arm, 3).tolist()}"
        )

    if simulation_app.is_running() is False:
        log("[SO100] SimulationApp closed by user.")
        break

# ─── Phase 12: Shutdown ─────────────────────────────────────────────────────

log("[SO100] Shutting down...")
trace_file.close()
world.stop()
log("[SO100] Done.")
# Isaac Sim 5.1 segfaults inside libomni.graph.core when SimulationApp.close()
# tears down. Trace + images are already flushed, so skip the noisy shutdown.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
