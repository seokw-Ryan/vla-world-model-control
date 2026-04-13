"""Gymnasium wrapper around the Isaac Sim SO-100 environment.

Provides a standard gymnasium.Env interface compatible with LeRobot's
training and evaluation pipelines.

Usage:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from sim.isaac.so100_gym_wrapper import IsaacSO100Env
    env = IsaacSO100Env(simulation_app=simulation_app)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

IMPORTANT: Must be run with Isaac Sim's python.sh.
SimulationApp must be created BEFORE importing this module.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# SO-100 joint info
SO100_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
SO100_NUM_ARM_JOINTS = 5
SO100_JOINT_LIMITS_LOWER = np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158], dtype=np.float32)
SO100_JOINT_LIMITS_UPPER = np.array([2.0, 3.5, 0.0, 1.2, 3.14158], dtype=np.float32)
SO100_GRIPPER_OPEN = 1.5
SO100_GRIPPER_CLOSED = -0.1


class IsaacSO100Env(gym.Env):
    """Gymnasium environment wrapping Isaac Sim SO-100 tabletop scene.

    Observation space:
        - observation.state: (6,) float32 — all joint positions (5 arm + 1 gripper)
        - observation.images.front: (3, H, W) uint8 — RGB camera image

    Action space:
        - (6,) float32 — [shoulder_pan, shoulder_lift, elbow_flex,
                          wrist_flex, wrist_roll, gripper]
          Joint position deltas for the 5 arm joints + gripper command.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        simulation_app,
        sim_config_path: str = "configs/sim/so100_standalone.yaml",
        image_height: int = 256,
        image_width: int = 256,
        max_episode_steps: int = 200,
        control_decimation: int = 12,
        physics_dt: float = 1.0 / 60.0,
        render_mode: str = "rgb_array",
        cube_randomize: bool = True,
        cube_pos_range: Optional[dict] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self._sim_app = simulation_app
        self._image_height = image_height
        self._image_width = image_width
        self._max_episode_steps = max_episode_steps
        self._control_decimation = control_decimation
        self._physics_dt = physics_dt
        self._cube_randomize = cube_randomize
        self._cube_pos_range = cube_pos_range
        self._cube_spawn_cfg = {
            "radius": (0.10, 0.22),
            "angle_deg": (-90.0, 90.0),
            "z": 0.415,
        }

        self._step_count = 0
        self._world = None
        self._robot = None
        self._cube = None
        self._camera = None
        self._robot_origin_xy = np.zeros(2, dtype=np.float32)

        # Joint index mapping (populated during scene build)
        self._joint_indices = None
        self._arm_joint_indices = None
        self._gripper_joint_idx = None

        # Default joint positions (5 arm + 1 gripper)
        self._default_joints = np.array(
            [0.0, 1.0, -1.0, -0.5, 0.0, 0.5], dtype=np.float32
        )
        self._default_cube_pos = np.array([0.15, 0.0, 0.415], dtype=np.float32)

        # Lifted threshold for reward
        self._lift_height = 0.48

        # Define spaces — 6D: 5 arm joint deltas + 1 gripper
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32
                ),
                "observation.images.front": spaces.Box(
                    low=0, high=255,
                    shape=(3, image_height, image_width),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._build_scene(sim_config_path)

    # ─── Scene construction ────────────────────────────────────────────

    def _build_scene(self, config_path: str) -> None:
        """Build the Isaac Sim scene with SO-100."""
        import yaml

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.api.robots import Robot
        from isaacsim.sensors.camera import Camera
        from isaacsim.asset.importer.urdf import _urdf
        import omni.kit.commands

        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        scene_cfg = cfg.get("scene", {})
        camera_cfg = cfg.get("camera", {})
        robot_cfg = cfg.get("robot", {})

        # Create world
        self._world = World(
            physics_dt=self._physics_dt, stage_units_in_meters=1.0
        )
        self._world.scene.add_default_ground_plane()

        # Import SO-100 directly from URDF onto live stage
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        urdf_path = os.path.join(
            project_root,
            robot_cfg.get("urdf_path", "assets/so100/so100.urdf"),
        )
        robot_position = np.array(robot_cfg.get("position", [0.0, 0.0, 0.40]))
        self._robot_origin_xy = robot_position[:2].astype(np.float32)
        robot_orientation = np.array(
            robot_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])
        )

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"SO-100 URDF not found: {urdf_path}"
            )

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
        log(f"[SO100Env] URDF imported, prim: {robot_prim_path}")

        self._robot = self._world.scene.add(
            Robot(
                prim_path=robot_prim_path,
                name="so100",
                position=robot_position,
                orientation=robot_orientation,
            )
        )

        table_cfg = scene_cfg.get("table", {})
        if table_cfg.get("enabled", False):
            self._world.scene.add(
                FixedCuboid(
                    prim_path=table_cfg.get("prim_path", "/World/Table"),
                    name="table",
                    position=np.array(table_cfg.get("position", [0.0, 0.0, 0.25])),
                    scale=np.array(table_cfg.get("size", [0.5, 0.5, 0.5])),
                    color=np.array(table_cfg.get("color", [0.4, 0.3, 0.2])),
                )
            )

        # Cube
        cube_cfg = scene_cfg.get("cube", {})
        cube_size = cube_cfg.get("size", 0.03)
        self._default_cube_pos = np.array(
            cube_cfg.get("position", [0.15, 0.0, 0.415]), dtype=np.float32
        )
        self._lift_height = float(cube_cfg.get("lift_height", float(self._default_cube_pos[2]) + 0.065))
        spawn_cfg = cube_cfg.get("spawn", {})
        self._cube_spawn_cfg = {
            "radius": tuple(spawn_cfg.get("radius", [0.10, 0.22])),
            "angle_deg": tuple(spawn_cfg.get("angle_deg", [-90.0, 90.0])),
            "z": float(spawn_cfg.get("z", self._default_cube_pos[2])),
        }
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=cube_cfg.get("prim_path", "/World/Cube"),
                name="cube",
                position=self._default_cube_pos.copy(),
                scale=np.array([cube_size, cube_size, cube_size]),
                color=np.array(cube_cfg.get("color", [0.9, 0.1, 0.1])),
                mass=cube_cfg.get("mass", 0.05),
            )
        )

        # Camera
        cam_res = camera_cfg.get("resolution", [256, 256])
        cam_pos = camera_cfg.get("position", [0.25, 0.25, 0.9])
        cam_target = camera_cfg.get("target", [0.15, 0.0, 0.45])
        self._camera = Camera(
            prim_path=camera_cfg.get("prim_path", "/World/Camera"),
            resolution=(cam_res[0], cam_res[1]),
            position=np.array(cam_pos),
        )

        # Initialize
        self._world.reset()
        self._camera.initialize()
        self._camera.set_focal_length(1.5)

        # Point camera at workspace
        cam_quat = self._compute_lookat_quat(
            np.array(cam_pos), np.array(cam_target)
        )
        self._camera.set_world_pose(
            position=np.array(cam_pos), orientation=cam_quat
        )

        # Initialize robot and discover joint mapping
        self._robot.initialize()
        dof_names = self._robot.dof_names
        log(f"[SO100Env] Robot DOF names: {dof_names}")

        self._joint_indices = []
        for name in SO100_JOINT_NAMES:
            if name in dof_names:
                self._joint_indices.append(dof_names.index(name))
            else:
                log(f"[SO100Env] WARNING: joint '{name}' not found")
        self._joint_indices = np.array(self._joint_indices)
        self._arm_joint_indices = self._joint_indices[:SO100_NUM_ARM_JOINTS]
        if len(self._joint_indices) > SO100_NUM_ARM_JOINTS:
            self._gripper_joint_idx = self._joint_indices[SO100_NUM_ARM_JOINTS]

        # Warm up camera
        for i in range(20):
            self._world.step(render=True)
            if self._camera.get_rgba() is not None:
                break

        log("[SO100Env] Scene built and ready.")

    @property
    def cube_rest_height(self) -> float:
        return float(self._default_cube_pos[2])

    @property
    def lift_height(self) -> float:
        return float(self._lift_height)

    @staticmethod
    def _compute_lookat_quat(
        eye: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Compute wxyz quaternion for camera looking from eye to target."""
        cam_forward = target - eye
        cam_forward = cam_forward / np.linalg.norm(cam_forward)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(cam_forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        cam_right = np.cross(cam_forward, world_up)
        cam_right = cam_right / np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_forward)
        rot_matrix = np.stack([cam_right, cam_up, -cam_forward], axis=1)
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # ─── Gym interface ─────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed, options=options)
        self._step_count = 0

        # Reset robot to default pose
        all_joints = self._robot.get_joint_positions()
        all_joints[self._joint_indices] = self._default_joints
        self._robot.set_joint_positions(all_joints)

        # Reset cube position (optionally randomized)
        if self._cube_randomize:
            cube_pos = self._sample_cube_position()
        else:
            cube_pos = self._default_cube_pos.copy()

        self._cube.set_world_pose(position=cube_pos)
        self._cube.set_linear_velocity(np.zeros(3))
        self._cube.set_angular_velocity(np.zeros(3))

        # Step to let physics settle
        for _ in range(10):
            self._world.step(render=True)

        obs = self._get_obs()
        info = {"cube_pos": cube_pos.copy()}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Action: [5 arm joint deltas, 1 gripper command]
        JOINT_DELTA_SCALE = 0.1
        arm_deltas = action[:SO100_NUM_ARM_JOINTS] * JOINT_DELTA_SCALE
        gripper_cmd = float(action[SO100_NUM_ARM_JOINTS])

        # Get current joint positions and apply deltas
        current_joints = self._robot.get_joint_positions()
        current_arm = current_joints[self._arm_joint_indices]

        target_arm = current_arm + arm_deltas
        target_arm = np.clip(
            target_arm, SO100_JOINT_LIMITS_LOWER, SO100_JOINT_LIMITS_UPPER
        )

        # Gripper
        gripper_target = SO100_GRIPPER_OPEN if gripper_cmd > 0.0 else SO100_GRIPPER_CLOSED

        # Apply
        target_joints = current_joints.copy()
        target_joints[self._arm_joint_indices] = target_arm
        if self._gripper_joint_idx is not None:
            target_joints[self._gripper_joint_idx] = gripper_target
        self._robot.set_joint_positions(target_joints)

        # Step simulation
        for _ in range(self._control_decimation):
            self._world.step(render=True)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self._step_count >= self._max_episode_steps
        info = {
            "is_success": terminated,
            "cube_pos": self._cube.get_world_pose()[0].copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._get_camera_rgb()
        return None

    def close(self) -> None:
        if self._world is not None:
            self._world.stop()

    # ─── Internal helpers ──────────────────────────────────────────────

    def _get_obs(self) -> dict:
        """Build observation dict matching LeRobot convention."""
        current_joints = self._robot.get_joint_positions()
        # 6 joints: 5 arm + 1 gripper
        so100_joints = current_joints[self._joint_indices].astype(np.float32)

        rgb = self._get_camera_rgb()

        return {
            "observation.state": so100_joints,
            "observation.images.front": rgb,
        }

    def _get_camera_rgb(self) -> np.ndarray:
        """Get camera RGB as (3, H, W) uint8 array (CHW for LeRobot)."""
        rgba = self._camera.get_rgba()
        if rgba is None:
            return np.zeros(
                (3, self._image_height, self._image_width), dtype=np.uint8
            )
        if rgba.max() <= 1.0:
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        else:
            rgb = rgba[:, :, :3].astype(np.uint8)
        # HWC -> CHW
        return np.transpose(rgb, (2, 0, 1))

    def _compute_reward(self) -> float:
        """Reward: +1 if cube is lifted above threshold, else 0."""
        cube_pos, _ = self._cube.get_world_pose()
        return 1.0 if cube_pos[2] > self._lift_height else 0.0

    def _check_success(self) -> bool:
        """Episode succeeds if cube is above lift height."""
        cube_pos, _ = self._cube.get_world_pose()
        return bool(cube_pos[2] > self._lift_height)

    def _sample_cube_position(self) -> np.ndarray:
        rng = self.np_random
        if self._cube_pos_range is not None:
            cx = rng.uniform(*self._cube_pos_range["x"])
            cy = rng.uniform(*self._cube_pos_range["y"])
            cz = rng.uniform(*self._cube_pos_range["z"])
            return np.array([cx, cy, cz], dtype=np.float32)

        radius = rng.uniform(*self._cube_spawn_cfg["radius"])
        angle_deg = rng.uniform(*self._cube_spawn_cfg["angle_deg"])
        angle = np.deg2rad(angle_deg)

        x = self._robot_origin_xy[0] + radius * np.cos(angle)
        y = self._robot_origin_xy[1] + radius * np.sin(angle)
        z = self._cube_spawn_cfg["z"]
        return np.array([x, y, z], dtype=np.float32)
