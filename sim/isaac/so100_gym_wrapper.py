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
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def _author_root_pose(prim, position: np.ndarray, orientation_wxyz: np.ndarray) -> None:
    """Write the imported robot prim's root xform so the Stage UI matches runtime placement."""
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
        image_height: int | None = None,
        image_width: int | None = None,
        max_episode_steps: int = 200,
        control_decimation: int = 12,
        physics_dt: float = 1.0 / 60.0,
        render_mode: str = "rgb_array",
        cube_randomize: bool = True,
        cube_pos_range: Optional[dict] = None,
        camera_position: Optional[tuple[float, float, float]] = None,
        camera_target: Optional[tuple[float, float, float]] = None,
        camera_focal_length: Optional[float] = None,
        default_joint_positions: Optional[list[float]] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self._sim_app = simulation_app
        self._requested_image_height = image_height
        self._requested_image_width = image_width
        self._max_episode_steps = max_episode_steps
        self._control_decimation = control_decimation
        self._physics_dt = physics_dt
        self._cube_randomize = cube_randomize
        self._cube_pos_range = cube_pos_range
        self._camera_position_override = camera_position
        self._camera_target_override = camera_target
        self._camera_focal_length_override = camera_focal_length
        self._default_joint_positions_override = default_joint_positions
        self._cube_spawn_cfg = {
            "mode": "tabletop_rect",
            "edge_margin": 0.05,
            "table_local_x": (-0.33, -0.23),
            "table_local_y": (0.06, 0.22),
            "z": 0.415,
        }

        self._step_count = 0
        self._world = None
        self._robot = None
        self._cube = None
        self._camera = None
        self._robot_origin_xy = np.zeros(2, dtype=np.float32)
        self._robot_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._table_spawn_rect_xy = None

        # Joint index mapping (populated during scene build)
        self._joint_indices = None
        self._arm_joint_indices = None
        self._gripper_joint_idx = None

        # Default joint positions (5 arm + 1 gripper)
        self._default_joints = np.zeros(6, dtype=np.float32)
        self._default_cube_pos = np.array([0.15, 0.0, 0.415], dtype=np.float32)
        self._default_cube_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Lifted threshold for reward
        self._lift_height = 0.48

        self._build_scene(sim_config_path)
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32
                ),
                "observation.images.front": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self._image_height, self._image_width),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    # ─── Scene construction ────────────────────────────────────────────

    def _build_scene(self, config_path: str) -> None:
        """Build the Isaac Sim scene with SO-100."""
        import yaml

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.viewports import set_active_viewport_camera
        import omni.kit.commands

        # Isaac Lab's AppLauncher path does not always preload these extensions.
        enable_extension("isaacsim.sensors.camera")
        enable_extension("isaacsim.asset.importer.urdf")
        from isaacsim.asset.importer.urdf import _urdf
        from isaacsim.sensors.camera import Camera

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
        robot_position = np.array(robot_cfg.get("position", [0.0, 0.0, 0.42]))
        self._robot_origin_xy = robot_position[:2].astype(np.float32)
        robot_orientation = np.array(
            robot_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])
        )
        self._robot_orientation_wxyz = robot_orientation.astype(np.float32)

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
        # URDF import lands the articulation at /so_arm100 with an authored root translate of zero.
        # Write the root xform on that imported prim directly so the Property panel reflects the
        # actual robot placement used by training, instead of showing a misleading 0,0,0 transform.
        robot_prim = self._world.stage.GetPrimAtPath(robot_prim_path)
        _author_root_pose(robot_prim, robot_position, robot_orientation)

        self._robot = self._world.scene.add(
            Robot(
                prim_path=robot_prim_path,
                name="so100",
            )
        )

        table_cfg = scene_cfg.get("table", {})
        if table_cfg.get("enabled", False):
            table_position = np.array(table_cfg.get("position", [0.0, 0.0, 0.25]), dtype=np.float32)
            table_size = np.array(table_cfg.get("size", [0.5, 0.5, 0.5]), dtype=np.float32)
            # FixedCuboid size is the full box extent. The tabletop top surface is center_z + size_z / 2.
            # Keep the robot base slightly above that surface; otherwise the arm gets spawned intersecting
            # the table and visually ends up underneath / inside it.
            table_top_z = float(table_position[2] + table_size[2] * 0.5)
            clearance = float(robot_position[2] - table_top_z)
            if clearance <= 0.0:
                log(
                    "[SO100Env] WARNING: robot base z is at or below the tabletop top surface "
                    f"(robot_z={float(robot_position[2]):.3f}, table_top_z={table_top_z:.3f}, "
                    f"clearance={clearance:.3f}). "
                    "Increase robot.position[2] in configs/sim/so100_standalone.yaml."
                )
            else:
                log(
                    "[SO100Env] Table clearance check passed: "
                    f"robot_z={float(robot_position[2]):.3f}, table_top_z={table_top_z:.3f}, "
                    f"clearance={clearance:.3f}"
                )
            self._world.scene.add(
                FixedCuboid(
                    prim_path=table_cfg.get("prim_path", "/World/Table"),
                    name="table",
                    position=table_position,
                    scale=table_size,
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
            "mode": str(spawn_cfg.get("mode", "tabletop_rect")),
            "edge_margin": float(spawn_cfg.get("edge_margin", 0.05)),
            "table_local_x": tuple(spawn_cfg.get("table_local_x", [-0.33, -0.23])),
            "table_local_y": tuple(spawn_cfg.get("table_local_y", [0.06, 0.22])),
            "z": float(spawn_cfg.get("z", self._default_cube_pos[2])),
        }
        if table_cfg.get("enabled", False):
            self._table_spawn_rect_xy = self._compute_tabletop_spawn_rect_xy(
                table_position=table_position,
                table_size=table_size,
                cube_cfg=cube_cfg,
            )
        self._default_cube_orientation = np.array(
            cube_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32
        )
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=cube_cfg.get("prim_path", "/World/Cube"),
                name="cube",
                position=self._default_cube_pos.copy(),
                orientation=self._default_cube_orientation.copy(),
                scale=np.array([cube_size, cube_size, cube_size]),
                color=np.array(cube_cfg.get("color", [0.9, 0.1, 0.1])),
                mass=cube_cfg.get("mass", 0.05),
            )
        )
        if self._cube_spawn_cfg["mode"] == "tabletop_rect" and self._table_spawn_rect_xy is not None:
            log(
                "[SO100Env] Cube spawn rectangle: "
                f"x={self._table_spawn_rect_xy['x']} y={self._table_spawn_rect_xy['y']} "
                f"z={self._cube_spawn_cfg['z']:.3f}"
            )

        default_joint_positions = self._default_joint_positions_override
        if default_joint_positions is None:
            default_joint_positions = robot_cfg.get("default_joint_positions", [0.0, 1.0, -1.0, -0.5, 0.0, 0.5])
        self._default_joints = np.asarray(default_joint_positions, dtype=np.float32)
        if self._default_joints.shape != (6,):
            raise ValueError(
                f"default_joint_positions must have shape (6,), got {self._default_joints.shape}"
            )

        # Camera
        cam_res = camera_cfg.get("resolution", [640, 480])
        self._image_width = int(self._requested_image_width or cam_res[0])
        self._image_height = int(self._requested_image_height or cam_res[1])
        cam_pos = self._camera_position_override or tuple(camera_cfg.get("position", [0.25, 0.25, 0.9]))
        cam_target = self._camera_target_override or tuple(camera_cfg.get("target", [0.15, 0.0, 0.45]))
        focal_length = float(self._camera_focal_length_override or camera_cfg.get("focal_length", 1.5))
        cam_euler_deg = camera_cfg.get("orientation_euler_xyz_deg")
        cam_orient_wxyz_cfg = camera_cfg.get("orientation_wxyz")
        self._camera = Camera(
            prim_path=camera_cfg.get("prim_path", "/World/Camera"),
            resolution=(self._image_width, self._image_height),
            position=np.array(cam_pos),
        )

        # Initialize
        self._world.reset()
        self._camera.initialize()
        self._camera.set_focal_length(focal_length)

        # Explicit orientation (Euler XYZ degrees or wxyz quaternion) takes precedence
        # over look-at, matching the USD "Orient" widget the user authored in the viewport.
        if cam_euler_deg is not None:
            cam_quat = self._euler_xyz_deg_to_wxyz(cam_euler_deg)
            cam_axes = "usd"
        elif cam_orient_wxyz_cfg is not None:
            cam_quat = np.array(cam_orient_wxyz_cfg, dtype=np.float32)
            cam_axes = "world"
        else:
            cam_quat = self._compute_lookat_quat(np.array(cam_pos), np.array(cam_target))
            cam_axes = "world"
        self._camera.set_world_pose(
            position=np.array(cam_pos), orientation=cam_quat, camera_axes=cam_axes
        )
        camera_prim_path = camera_cfg.get("prim_path", "/World/Camera")
        try:
            # In GUI mode, force the active viewport to show the training camera so you
            # immediately see the exact camera feeding the policy instead of the default
            # perspective viewport.
            set_active_viewport_camera(camera_prim_path)
            log(f"[SO100Env] Active viewport camera set to {camera_prim_path}.")
        except Exception as exc:
            log(
                "[SO100Env] Camera prim is "
                f"{camera_prim_path} "
                f"(viewport camera menu -> Cameras -> {camera_prim_path}). "
                f"Automatic viewport switch failed: {exc}"
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
    def _euler_xyz_deg_to_wxyz(euler_deg) -> np.ndarray:
        """Convert USD-style XYZ Euler degrees to a wxyz quaternion.

        The USD Property panel's "Orient" widget applies rotations in the local
        frame in X→Y→Z order (intrinsic XYZ), which matches scipy's lowercase
        'xyz' convention.
        """
        rot = Rotation.from_euler("xyz", np.asarray(euler_deg, dtype=np.float64), degrees=True)
        xyzw = rot.as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)

    @staticmethod
    def _compute_lookat_quat(
        eye: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Compute wxyz quaternion in Isaac's world camera axes (+X forward, +Z up)."""
        from isaacsim.core.utils.rotations import gf_quat_to_np_array, lookat_to_quatf

        eye_gf = Gf.Vec3f(float(eye[0]), float(eye[1]), float(eye[2]))
        target_gf = Gf.Vec3f(float(target[0]), float(target[1]), float(target[2]))
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        # Straight top-down views are parallel to +Z/-Z, so use +Y as the fallback up vector
        # to avoid a degenerate look-at basis while still pointing the camera downward.
        up = Gf.Vec3f(0.0, 0.0, 1.0)
        if abs(float(np.dot(forward, np.array([0.0, 0.0, 1.0])))) > 0.99:
            up = Gf.Vec3f(0.0, 1.0, 0.0)
        # Isaac's camera utility calls lookat_to_quatf(target, camera, up) for camera placement.
        return gf_quat_to_np_array(lookat_to_quatf(target_gf, eye_gf, up)).astype(np.float32)

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

        # Reset the cube quaternion every episode so it starts axis-aligned in front of the arm.
        self._cube.set_world_pose(
            position=cube_pos,
            orientation=self._default_cube_orientation.copy(),
        )
        self._cube.set_linear_velocity(np.zeros(3))
        self._cube.set_angular_velocity(np.zeros(3))
        log(
            "[SO100Env] Reset workspace: "
            f"robot_xy=({self._robot_origin_xy[0]:.3f}, {self._robot_origin_xy[1]:.3f}) "
            f"cube_xyz=({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}) "
            f"cube_quat_wxyz={self._default_cube_orientation.tolist()}"
        )

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

        if self._cube_spawn_cfg["mode"] != "tabletop_rect":
            raise ValueError(f"Unsupported cube spawn mode: {self._cube_spawn_cfg['mode']}")
        if self._table_spawn_rect_xy is None:
            raise RuntimeError("Cube spawn mode 'tabletop_rect' requires an enabled table in the scene config.")

        x = rng.uniform(*self._table_spawn_rect_xy["x"])
        y = rng.uniform(*self._table_spawn_rect_xy["y"])
        z = self._cube_spawn_cfg["z"]
        return np.array([x, y, z], dtype=np.float32)

    def _compute_tabletop_spawn_rect_xy(
        self,
        table_position: np.ndarray,
        table_size: np.ndarray,
        cube_cfg: dict,
    ) -> dict[str, tuple[float, float]]:
        """Compute a spawn rectangle inside the table using table-local coordinates only."""
        cube_size = float(cube_cfg.get("size", 0.03))
        inset = float(self._cube_spawn_cfg["edge_margin"]) + cube_size * 0.5

        table_x = (
            float(table_position[0] - table_size[0] * 0.5 + inset),
            float(table_position[0] + table_size[0] * 0.5 - inset),
        )
        table_y = (
            float(table_position[1] - table_size[1] * 0.5 + inset),
            float(table_position[1] + table_size[1] * 0.5 - inset),
        )

        table_local_x = self._cube_spawn_cfg["table_local_x"]
        table_local_y = self._cube_spawn_cfg["table_local_y"]
        requested_x = (
            float(table_position[0] + table_local_x[0]),
            float(table_position[0] + table_local_x[1]),
        )
        requested_y = (
            float(table_position[1] + table_local_y[0]),
            float(table_position[1] + table_local_y[1]),
        )
        spawn_x = (max(table_x[0], requested_x[0]), min(table_x[1], requested_x[1]))
        spawn_y = (max(table_y[0], requested_y[0]), min(table_y[1], requested_y[1]))
        if spawn_x[0] >= spawn_x[1] or spawn_y[0] >= spawn_y[1]:
            raise ValueError(
                "Computed cube spawn rectangle is empty. "
                f"table_x={table_x} table_y={table_y} requested_x={requested_x} requested_y={requested_y}"
            )
        return {"x": spawn_x, "y": spawn_y}
