"""Gymnasium wrapper around the Isaac Sim standalone Franka environment.

Provides a standard gymnasium.Env interface compatible with LeRobot's
training and evaluation pipelines.

Usage:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from sim.isaac.gym_wrapper import IsaacFrankaEnv
    env = IsaacFrankaEnv(simulation_app=simulation_app)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

IMPORTANT: Must be run with Isaac Sim's python.sh.
SimulationApp must be created BEFORE importing this module.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


class IsaacFrankaEnv(gym.Env):
    """Gymnasium environment wrapping Isaac Sim standalone Franka tabletop scene.

    Observation space:
        - observation.state: (7,) float32 — arm joint positions (excludes gripper)
        - observation.images.front: (3, H, W) uint8 — RGB camera image

    Action space:
        - (7,) float32 — [dx, dy, dz, drx, dry, drz, gripper]
          Position/rotation deltas applied via IK, gripper > 0 = open.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        simulation_app,
        sim_config_path: str = "configs/sim/isaac_standalone.yaml",
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
        self._cube_pos_range = cube_pos_range or {
            "x": (0.4, 0.6),
            "y": (-0.15, 0.15),
            "z": (0.525, 0.525),
        }

        self._step_count = 0
        self._world = None
        self._franka = None
        self._cube = None
        self._camera = None
        self._ik_solver = None
        self._articulation_controller = None

        # Default joint positions (7 arm + 2 gripper)
        self._default_joints = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            dtype=np.float32,
        )
        self._default_cube_pos = np.array([0.5, 0.0, 0.525])

        # Lifted threshold for reward
        self._lift_height = 0.6

        # Define spaces
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=-2 * np.pi, high=2 * np.pi, shape=(7,), dtype=np.float32
                ),
                "observation.images.front": spaces.Box(
                    low=0, high=255,
                    shape=(3, image_height, image_width),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self._build_scene(sim_config_path)

    # ─── Scene construction ────────────────────────────────────────────

    def _build_scene(self, config_path: str) -> None:
        """Build the Isaac Sim scene (SimulationApp must already exist)."""
        import yaml

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
        from isaacsim.sensors.camera import Camera

        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        scene_cfg = cfg.get("scene", {})
        camera_cfg = cfg.get("camera", {})

        # Create world
        self._world = World(
            physics_dt=self._physics_dt, stage_units_in_meters=1.0
        )
        self._world.scene.add_default_ground_plane()

        # Franka
        self._franka = self._world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )

        # Table
        table_cfg = scene_cfg.get("table", {})
        self._world.scene.add(
            FixedCuboid(
                prim_path=table_cfg.get("prim_path", "/World/Table"),
                name="table",
                position=np.array(table_cfg.get("position", [0.5, 0.0, 0.25])),
                scale=np.array(table_cfg.get("size", [0.6, 0.8, 0.5])),
                color=np.array(table_cfg.get("color", [0.4, 0.3, 0.2])),
            )
        )

        # Cube
        cube_cfg = scene_cfg.get("cube", {})
        cube_size = cube_cfg.get("size", 0.04)
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=cube_cfg.get("prim_path", "/World/Cube"),
                name="cube",
                position=np.array(cube_cfg.get("position", [0.5, 0.0, 0.525])),
                scale=np.array([cube_size, cube_size, cube_size]),
                color=np.array(cube_cfg.get("color", [0.9, 0.1, 0.1])),
                mass=cube_cfg.get("mass", 0.1),
            )
        )

        # Camera
        cam_res = camera_cfg.get("resolution", [256, 256])
        cam_pos = camera_cfg.get("position", [0.5, 0.0, 1.2])
        cam_target = camera_cfg.get("target", [0.5, 0.0, 0.5])
        self._camera = Camera(
            prim_path=camera_cfg.get("prim_path", "/World/Camera"),
            resolution=(cam_res[0], cam_res[1]),
            position=np.array(cam_pos),
        )

        # Initialize
        self._world.reset()
        self._camera.initialize()
        self._camera.set_focal_length(1.5)

        # Point camera at table
        cam_quat = self._compute_lookat_quat(
            np.array(cam_pos), np.array(cam_target)
        )
        self._camera.set_world_pose(
            position=np.array(cam_pos), orientation=cam_quat
        )

        # IK solver
        self._ik_solver = KinematicsSolver(self._franka)
        self._articulation_controller = self._franka.get_articulation_controller()

        # Warm up camera
        for i in range(20):
            self._world.step(render=True)
            if self._camera.get_rgba() is not None:
                break

        log("[GymWrapper] Scene built and ready.")

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
        self._franka.set_joint_positions(self._default_joints)

        # Reset cube position (optionally randomized)
        if self._cube_randomize:
            rng = self.np_random
            cx = rng.uniform(*self._cube_pos_range["x"])
            cy = rng.uniform(*self._cube_pos_range["y"])
            cz = rng.uniform(*self._cube_pos_range["z"])
            cube_pos = np.array([cx, cy, cz])
        else:
            cube_pos = self._default_cube_pos.copy()

        self._cube.set_world_pose(position=cube_pos)
        # Zero out cube velocity
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

        delta_pos = action[:3].astype(np.float64)
        delta_rot = action[3:6].astype(np.float64)
        gripper_cmd = float(action[6])

        # Get current EE pose
        ee_pos, ee_rot_matrix = self._ik_solver.compute_end_effector_pose()

        # Compute target
        target_pos = ee_pos + delta_pos
        target_quat = self._apply_delta_rotation(ee_rot_matrix, delta_rot)

        # Solve IK
        ik_result, success = self._ik_solver.compute_inverse_kinematics(
            target_position=target_pos, target_orientation=target_quat
        )

        if success:
            arm_positions = np.array(ik_result.joint_positions, dtype=np.float32)
            gripper_val = 0.04 if gripper_cmd > 0.0 else 0.0
            full_positions = np.concatenate(
                [arm_positions, [gripper_val, gripper_val]]
            )
            self._franka.set_joint_positions(full_positions)

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
            "ik_success": bool(success),
            "cube_pos": self._cube.get_world_pose()[0].copy(),
            "ee_pos": ee_pos.copy(),
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
        # Joint positions (7 arm joints, exclude 2 gripper fingers)
        joint_positions = self._franka.get_joint_positions()
        arm_joints = joint_positions[:7].astype(np.float32)

        # Camera image: (H, W, 4) RGBA float -> (3, H, W) uint8 RGB
        rgb = self._get_camera_rgb()

        return {
            "observation.state": arm_joints,
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

    @staticmethod
    def _apply_delta_rotation(
        current_rot_matrix: np.ndarray, delta_euler: np.ndarray
    ) -> np.ndarray:
        """Apply euler delta to rotation matrix, return wxyz quaternion."""
        current_rot = Rotation.from_matrix(current_rot_matrix)
        delta_rot = Rotation.from_euler("xyz", delta_euler)
        new_rot = delta_rot * current_rot
        xyzw = new_rot.as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
