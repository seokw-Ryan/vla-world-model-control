"""Convert SO-100 URDF to USD for Isaac Sim.

Usage:
    <ISAAC_SIM>/python.sh scripts/convert_so100_urdf.py

Output:
    assets/so100/so100_base.usd   # raw URDF conversion
    assets/so100/so100.usd        # thin wrapper with the authored root translate
"""

from __future__ import annotations

import os
import sys

# Must create SimulationApp before any omni imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "so100", "so100.urdf")
USD_BASE_PATH = os.path.join(PROJECT_ROOT, "assets", "so100", "so100_base.usd")
USD_WRAPPER_PATH = os.path.join(PROJECT_ROOT, "assets", "so100", "so100.usd")


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def main() -> None:
    log(f"[convert] URDF: {URDF_PATH}")
    log(f"[convert] USD base output: {USD_BASE_PATH}")
    log(f"[convert] USD wrapper output: {USD_WRAPPER_PATH}")

    if not os.path.exists(URDF_PATH):
        log(f"[convert] ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    # Build ImportConfig object with proper setter methods
    import_config = _urdf.ImportConfig()
    import_config.set_merge_fixed_joints(False)
    import_config.set_fix_base(True)
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(True)
    import_config.set_default_drive_type(1)  # 1 = position drive
    import_config.set_default_drive_strength(1e4)
    import_config.set_default_position_drive_damping(1e3)
    import_config.set_self_collision(False)
    import_config.set_parse_mimic(True)

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        dest_path=USD_BASE_PATH,
        import_config=import_config,
    )

    if prim_path:
        log(f"[convert] Success! Robot prim: {prim_path}")
        wrapper_text = """#usda 1.0
(
    defaultPrim = "so_arm100"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "so_arm100" (
    prepend references = @./so100_base.usd@
)
{
    # Keep an authored root translate on the robot prim itself so the Stage UI does
    # not show a misleading zero transform for the spawned SO100 asset in Isaac Lab.
    quatd xformOp:orient = (1, 0, 0, 0)
    double3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0.02)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}
"""
        with open(USD_WRAPPER_PATH, "w", encoding="utf-8") as f:
            f.write(wrapper_text)
        log(f"[convert] USD base saved to: {USD_BASE_PATH}")
        log(f"[convert] USD wrapper saved to: {USD_WRAPPER_PATH}")
    else:
        log("[convert] ERROR: URDF import failed.")
        sys.exit(1)

    simulation_app.close()


if __name__ == "__main__":
    main()
