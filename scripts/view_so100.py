"""Open Isaac Sim GUI to view the SO-100 arm on a table.

Usage:
    <ISAAC_SIM>/python.sh scripts/view_so100.py

No VLA model loaded — just the robot, table, and cube for visual inspection.
Close the Isaac Sim window to exit.
"""

from __future__ import annotations

import os
import sys


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
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from pxr import UsdGeom, Gf, UsdLux

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
urdf_path = os.path.join(project_root, "assets", "so100", "so100.urdf")

world = World(physics_dt=1.0 / 60.0, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

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

# Place robot on the tabletop
prim = stage.GetPrimAtPath(robot_prim_path)
xform = UsdGeom.Xformable(prim)
xform.ClearXformOpOrder()
xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.40))

so100 = world.scene.add(
    Robot(prim_path=robot_prim_path, name="so100")
)

# Thin tabletop
world.scene.add(
    FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.12, 0.0, 0.38]),
        scale=np.array([0.70, 0.60, 0.04]),
        color=np.array([0.4, 0.3, 0.2]),
    )
)

# Red cube
world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0.15, 0.0, 0.415]),
        scale=np.array([0.03, 0.03, 0.03]),
        color=np.array([0.9, 0.1, 0.1]),
        mass=0.05,
    )
)

world.reset()
so100.initialize()
log(f"[view] DOFs: {so100.dof_names}")
log(f"[view] Scene ready — use the viewport to navigate around the robot.")
log(f"[view] Close the Isaac Sim window to exit.")

while simulation_app.is_running():
    world.step(render=True)

world.stop()
simulation_app.close()
