"""SO-100 embodiment for IsaacLab Arena.

The SO-100 is a 6-DOF tabletop arm (5 arm joints + 1 gripper) popular in
the LeRobot community. This is the first Arena registration of this robot.

Joint space:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_camera_frame_orientation_convention, create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose
import torch

# ─── Constants ────────────────────────────────────────────────────────────────

SO100_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

SO100_ARM_JOINT_NAMES = SO100_JOINT_NAMES[:5]

SO100_DEFAULT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 1.0,
    "elbow_flex": -1.0,
    "wrist_flex": -0.5,
    "wrist_roll": 0.0,
    "gripper": 0.5,
}

# Path to SO-100 USD asset
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SO100_USD_PATH = os.path.join(_PROJECT_ROOT, "assets", "so100", "so100.usd")

_DEFAULT_SO100_CAMERA_EYE = (0.25, 0.25, 0.9)
_DEFAULT_SO100_CAMERA_TARGET = (0.15, 0.0, 0.45)
_DEFAULT_SO100_CAMERA_RESOLUTION = (640, 480)


def _camera_pose_from_eye_target(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> Pose:
    """Create a world-frame camera pose from an eye/target pair."""
    eyes = torch.tensor([eye], dtype=torch.float32)
    targets = torch.tensor([target], dtype=torch.float32)
    rot_opengl = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device="cpu"))
    rot_world = convert_camera_frame_orientation_convention(rot_opengl, origin="opengl", target="world")[0]
    return Pose(position_xyz=eye, rotation_wxyz=tuple(float(v) for v in rot_world.tolist()))


# ─── Articulation Config ─────────────────────────────────────────────────────

SO100_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path=_SO100_USD_PATH,
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The task table top sits at z=0.50 (table center z=0.25 with size z=0.50).
        # Keep the robot base slightly above that surface so the arm does not spawn
        # intersecting the tabletop when the Arena env resets.
        pos=(0.0, 0.0, 0.52),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=SO100_DEFAULT_JOINT_POS,
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=SO100_ARM_JOINT_NAMES,
            stiffness=1e4,
            damping=1e3,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            stiffness=1e4,
            damping=1e3,
        ),
    },
)


# ─── Scene Configuration ─────────────────────────────────────────────────────


@configclass
class SO100SceneCfg:
    """Scene config: just the robot (table is part of the task scene)."""

    robot: ArticulationCfg = SO100_CFG


# ─── Camera Configuration ─────────────────────────────────────────────────────


@configclass
class SO100CameraCfg:
    """External workspace RGB camera for SO-100 policy observations."""

    front_camera: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled_camera = getattr(self, "_is_tiled_camera", True)
        camera_pose = getattr(
            self,
            "_camera_pose",
            _camera_pose_from_eye_target(_DEFAULT_SO100_CAMERA_EYE, _DEFAULT_SO100_CAMERA_TARGET),
        )
        camera_resolution = getattr(self, "_camera_resolution", _DEFAULT_SO100_CAMERA_RESOLUTION)
        focal_length = getattr(self, "_camera_focal_length", 1.5)
        horizontal_aperture = getattr(self, "_camera_horizontal_aperture", 20.955)
        vertical_aperture = getattr(self, "_camera_vertical_aperture", None)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        spawn_kwargs = {
            "focal_length": focal_length,
            "horizontal_aperture": horizontal_aperture,
            "clipping_range": (0.01, 100.0),
        }
        if vertical_aperture is not None:
            spawn_kwargs["vertical_aperture"] = vertical_aperture

        self.front_camera = CameraClass(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.0,
            height=int(camera_resolution[1]),
            width=int(camera_resolution[0]),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(**spawn_kwargs),
            offset=OffsetClass(
                pos=camera_pose.position_xyz,
                rot=camera_pose.rotation_wxyz,
                convention="world",
            ),
        )


# ─── Action Configuration ────────────────────────────────────────────────────


@configclass
class SO100ActionsCfg:
    """Action specification: direct joint position control for all 6 joints."""

    joint_pos: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=SO100_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )


# ─── Observation Configuration ───────────────────────────────────────────────


@configclass
class SO100ObservationsCfg:
    """Observation specification for the SO-100."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: joint state."""

        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ─── Event Configuration ─────────────────────────────────────────────────────


@configclass
class SO100EventCfg:
    """Reset events for the SO-100."""

    init_arm_pose = EventTerm(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ─── Embodiment ──────────────────────────────────────────────────────────────


@register_asset
class SO100Embodiment(EmbodimentBase):
    """SO-100 tabletop arm embodiment for IsaacLab Arena.

    6-DOF robot: 5 arm joints (shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll) + 1 gripper joint.

    Uses direct joint position control (no IK).
    """

    name = "so100"
    tags = ["embodiment", "so100", "tabletop"]

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_eye: tuple[float, float, float] = _DEFAULT_SO100_CAMERA_EYE,
        camera_target: tuple[float, float, float] = _DEFAULT_SO100_CAMERA_TARGET,
        camera_resolution: tuple[int, int] = _DEFAULT_SO100_CAMERA_RESOLUTION,
        use_tiled_camera: bool = True,
        camera_focal_length: float = 1.5,
        camera_horizontal_aperture: float = 20.955,
        camera_vertical_aperture: float | None = None,
    ):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = SO100SceneCfg()
        self.camera_eye = camera_eye
        self.camera_target = camera_target
        self.camera_resolution = camera_resolution
        self.camera_config = SO100CameraCfg()
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_pose = _camera_pose_from_eye_target(camera_eye, camera_target)
        self.camera_config._camera_resolution = camera_resolution
        self.camera_config._camera_focal_length = camera_focal_length
        self.camera_config._camera_horizontal_aperture = camera_horizontal_aperture
        self.camera_config._camera_vertical_aperture = camera_vertical_aperture
        self.camera_config.__post_init__()
        self.action_config = SO100ActionsCfg()
        self.observation_config = SO100ObservationsCfg()
        self.event_config = SO100EventCfg()

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        """Align the default viewport with the policy camera for env_0."""
        if self.enable_cameras and env_cfg.viewer is not None:
            env_cfg.viewer.eye = self.camera_eye
            env_cfg.viewer.lookat = self.camera_target
            env_cfg.viewer.origin_type = "env"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.cam_prim_path = "/World/envs/env_0/FrontCamera"
            env_cfg.viewer.resolution = self.camera_resolution
        return env_cfg
