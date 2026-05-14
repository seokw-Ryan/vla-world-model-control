"""SO-100 pick-and-place task for IsaacLab Arena.

Simplified pick-and-place task for the SO-100 tabletop arm:
- Scene: table + red cube
- Success: cube lifted above a height threshold
- Termination: success, cube dropped below table, or timeout
"""

from __future__ import annotations

from typing import Any

import torch

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase


# ─── Custom termination: cube above height ───────────────────────────────────


def cube_above_height(
    env,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.6,
) -> torch.Tensor:
    """Returns True for envs where the cube is above the height threshold."""
    asset = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w[:, 2]
    return pos_w > minimum_height


# ─── Scene config ────────────────────────────────────────────────────────────


@configclass
class SO100TaskSceneCfg:
    """Scene objects added by the pick-and-place task."""

    # Table (static collider)
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.3, 0.2),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Red cube (dynamic rigid body)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, 0.0, 0.525),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
            ),
        ),
    )


# ─── Terminations ────────────────────────────────────────────────────────────


@configclass
class SO100TerminationsCfg:
    """Termination conditions for SO-100 pick-and-place."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = TerminationTermCfg(
        func=cube_above_height,
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "minimum_height": 0.6,
        },
    )

    cube_dropped: TerminationTermCfg = TerminationTermCfg(
        func=mdp_isaac_lab.root_height_below_minimum,
        params={
            "minimum_height": 0.0,
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


# ─── Events ──────────────────────────────────────────────────────────────────


@configclass
class SO100TaskEventsCfg:
    """Reset events for the task (cube randomization)."""

    reset_cube_pose = EventTermCfg(
        func=mdp_isaac_lab.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.07, 0.07),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


# ─── Task ────────────────────────────────────────────────────────────────────


class SO100PickAndPlaceTask(TaskBase):
    """Pick-and-place task for the SO-100 arm.

    The robot must pick up a small red cube from the table and lift it
    above a height threshold (0.6m). This is the canonical SO-100
    benchmark task for Arena.
    """

    def __init__(self, episode_length_s: float = 40.0):
        super().__init__(episode_length_s=episode_length_s)
        self.scene_config = SO100TaskSceneCfg()
        self.termination_cfg = SO100TerminationsCfg()
        self.events_cfg = SO100TaskEventsCfg()

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_prompt(self) -> str:
        return "Pick up the red cube from the table."

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(
            eye=(0.5, 0.5, 1.0),
            lookat=(0.15, 0.0, 0.55),
        )
