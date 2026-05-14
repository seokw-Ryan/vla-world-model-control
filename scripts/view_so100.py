"""Open Isaac Sim GUI to view the SO-100 arm on a table.

Usage:
    <ISAAC_SIM>/python.sh scripts/view_so100.py

No VLA model loaded — just the robot, table, and cube for visual inspection.
Close the Isaac Sim window to exit.
"""

from __future__ import annotations

import os
import sys

import yaml


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.viewports import set_active_viewport_camera
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from pxr import Gf, UsdGeom, UsdLux

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
urdf_path = os.path.join(project_root, "assets", "so100", "so100.urdf")
config_path = os.path.join(project_root, "configs", "sim", "so100_standalone.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    sim_cfg = yaml.safe_load(f)

robot_cfg = sim_cfg.get("robot", {})
scene_cfg = sim_cfg.get("scene", {})
robot_position = robot_cfg.get("position", [0.0, 0.0, 0.42])
robot_orientation = robot_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])
table_cfg = scene_cfg.get("table", {})
cube_cfg = scene_cfg.get("cube", {})
camera_cfg = sim_cfg.get("camera", {})

enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera

world = World(physics_dt=1.0 / 60.0, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()


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

# Add lighting
stage = world.stage
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1000)

# Import SO-100 from URDF
import_config = _urdf.ImportConfig()
import_config.set_merge_fixed_joints(False)
import_config.set_fix_base(True)
import_config.set_make_default_prim(True)
import_config.set_create_physics_scene(False)
import_config.set_default_drive_type(1)
import_config.set_default_drive_strength(1e4)
import_config.set_default_position_drive_damping(1e3)
import_config.set_self_collision(False)

status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path="",
)
log(f"[view] Robot imported at: {robot_prim_path}")

# Place robot using the same pose convention as the training env.
prim = stage.GetPrimAtPath(robot_prim_path)
xform = UsdGeom.Xformable(prim)
xform.ClearXformOpOrder()
xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Vec3d(*[float(v) for v in robot_position])
)
xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Quatd(
        float(robot_orientation[0]),
        Gf.Vec3d(
            float(robot_orientation[1]),
            float(robot_orientation[2]),
            float(robot_orientation[3]),
        ),
    )
)
xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))

so100 = world.scene.add(
    Robot(prim_path=robot_prim_path, name="so100")
)

table_position = np.array(table_cfg.get("position", [0.12, 0.0, 0.38]))
table_size = np.array(table_cfg.get("size", [0.70, 0.60, 0.04]))
table_top_z = float(table_position[2] + table_size[2] * 0.5)
log(
    "[view] Table clearance check: "
    f"robot_z={float(robot_position[2]):.3f}, table_top_z={table_top_z:.3f}, "
    f"clearance={float(robot_position[2]) - table_top_z:.3f}"
)
world.scene.add(
    FixedCuboid(
        prim_path=table_cfg.get("prim_path", "/World/Table"),
        name="table",
        position=table_position,
        scale=table_size,
        color=np.array(table_cfg.get("color", [0.4, 0.3, 0.2])),
    )
)

# Red cube
world.scene.add(
    DynamicCuboid(
        prim_path=cube_cfg.get("prim_path", "/World/Cube"),
        name="cube",
        position=np.array(cube_cfg.get("position", [0.15, 0.0, 0.415])),
        scale=np.array([cube_cfg.get("size", 0.03)] * 3),
        color=np.array(cube_cfg.get("color", [0.9, 0.1, 0.1])),
        mass=float(cube_cfg.get("mass", 0.05)),
    )
)

cam_res = camera_cfg.get("resolution", [640, 480])
cam_pos = np.array(camera_cfg.get("position", [0.12, 0.0, 0.95]), dtype=np.float32)
cam_target = np.array(camera_cfg.get("target", [0.12, 0.0, 0.40]), dtype=np.float32)
camera = Camera(
    prim_path=camera_cfg.get("prim_path", "/World/Camera"),
    resolution=(int(cam_res[0]), int(cam_res[1])),
    position=cam_pos,
)

world.reset()
so100.initialize()
camera.initialize()
camera.set_focal_length(float(camera_cfg.get("focal_length", 1.5)))
camera.set_world_pose(
    position=cam_pos,
    orientation=compute_lookat_quat(cam_pos, cam_target),
    camera_axes="world",
)
set_active_viewport_camera(camera_cfg.get("prim_path", "/World/Camera"))

log(f"[view] DOFs: {so100.dof_names}")
log(f"[view] Robot pose: position={robot_position}, orientation_wxyz={robot_orientation}")
log(f"[view] Camera pose: eye={cam_pos.tolist()}, target={cam_target.tolist()}")
log(f"[view] Scene ready — use the viewport to navigate around the robot.")
log(f"[view] Close the Isaac Sim window to exit.")

while simulation_app.is_running():
    world.step(render=True)

world.stop()
simulation_app.close()
