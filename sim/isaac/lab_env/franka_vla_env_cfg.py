"""Isaac Lab ManagerBasedRLEnv configuration for Franka VLA evaluation.

Defines scene, actions, observations, rewards, terminations, and events
for a tabletop pick-and-place task driven by a VLA policy.
"""

from __future__ import annotations

import math

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import (
    FrameTransformerCfg,
    OffsetCfg,
    TiledCameraCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG

# Import custom MDP terms
from sim.isaac.lab_env.mdp import rewards as custom_rewards

# ─── Scene ───────────────────────────────────────────────────────────────────


@configclass
class FrankaVLASceneCfg(InteractiveSceneCfg):
    """Scene configuration: robot, table, cube, camera, lights."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )

    # Franka Panda robot
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.3, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.25)),
    )

    # Cube (manipulation target)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.525)),
    )

    # Overhead camera (TiledCamera for batched rendering)
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 1.2),
            rot=(0.0, 0.0, 0.0, 1.0),  # will be adjusted by convention
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.5,
            horizontal_aperture=20.955,
        ),
        width=256,
        height=256,
    )

    # End-effector frame tracker
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
    )


# ─── Actions ─────────────────────────────────────────────────────────────────


@configclass
class ActionsCfg:
    """Action configuration: differential IK for arm + binary gripper."""

    arm_action: ActionTermCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=mdp.DifferentialInverseKinematicsActionCfg.InverseKinematicsCfg(
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        scale=0.5,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1034),
        ),
    )

    gripper_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


# ─── Observations ────────────────────────────────────────────────────────────


@configclass
class ObservationsCfg:
    """Observation configuration: RGB image + proprioception."""

    @configclass
    class ImageObsCfg(ObservationGroupCfg):
        concatenate_terms = False

        image: ObservationTermCfg = ObservationTermCfg(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "normalize": False},
        )

    @configclass
    class ProprioceptionCfg(ObservationGroupCfg):
        concatenate_terms = True

        joint_pos: ObservationTermCfg = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    image: ImageObsCfg = ImageObsCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()


# ─── Rewards ─────────────────────────────────────────────────────────────────


@configclass
class RewardsCfg:
    """Reward: object above height as success metric."""

    object_lifted: RewardTermCfg = RewardTermCfg(
        func=custom_rewards.object_above_height,
        params={"object_cfg": "cube", "min_height": 0.6},
        weight=1.0,
    )


# ─── Terminations ────────────────────────────────────────────────────────────


@configclass
class TerminationsCfg:
    """Termination: episode timeout."""

    time_out: TerminationTermCfg = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )


# ─── Events ──────────────────────────────────────────────────────────────────


@configclass
class EventsCfg:
    """Event: randomize cube position on reset."""

    reset_object_position: EventTermCfg = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

    reset_robot: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )


# ─── Main Env Config ────────────────────────────────────────────────────────


@configclass
class FrankaVLAEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment configuration for VLA evaluation."""

    # Scene
    scene: FrankaVLASceneCfg = FrankaVLASceneCfg(num_envs=1, env_spacing=2.5)

    # Managers
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    decimation: int = 12  # 12 physics steps per env step = 5 Hz control
    episode_length_s: float = 40.0
