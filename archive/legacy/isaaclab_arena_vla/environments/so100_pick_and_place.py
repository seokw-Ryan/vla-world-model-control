"""SO-100 pick-and-place Arena environment.

Composes SO100Embodiment + table/cube scene + PickAndPlaceTask into a
complete IsaacLab Arena environment.
"""

from __future__ import annotations

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class SO100PickAndPlaceEnvironment(ExampleEnvironmentBase):
    """SO-100 pick-and-place environment for IsaacLab Arena.

    This is the first SO-100 environment registered in Arena, bringing
    the most popular LeRobot community arm into standardized benchmarking.
    """

    name: str = "so100_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        # Import our custom embodiment and task (triggers @register_asset)
        from isaaclab_arena_vla.embodiments.so100 import SO100Embodiment
        from isaaclab_arena_vla.tasks.pick_and_place import SO100PickAndPlaceTask

        enable_cameras = getattr(args_cli, "enable_cameras", False)
        embodiment = SO100Embodiment(
            enable_cameras=enable_cameras,
            camera_eye=tuple(getattr(args_cli, "camera_eye", (0.25, 0.25, 0.9))),
            camera_target=tuple(getattr(args_cli, "camera_target", (0.15, 0.0, 0.45))),
            camera_resolution=(
                int(getattr(args_cli, "camera_width", 640)),
                int(getattr(args_cli, "camera_height", 480)),
            ),
            use_tiled_camera=bool(getattr(args_cli, "num_envs", 1) > 1),
            camera_focal_length=float(getattr(args_cli, "camera_focal_length", 1.5)),
            camera_horizontal_aperture=float(getattr(args_cli, "camera_horizontal_aperture", 20.955)),
            camera_vertical_aperture=getattr(args_cli, "camera_vertical_aperture", None),
        )

        task = SO100PickAndPlaceTask(
            episode_length_s=getattr(args_cli, "episode_length", 40.0),
        )

        # The task scene includes the table and cube
        scene = Scene()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--enable_cameras", action="store_true",
            help="Enable RGB camera observations",
        )
        parser.add_argument(
            "--episode_length", type=float, default=40.0,
            help="Episode length in seconds",
        )
        parser.add_argument(
            "--camera_eye", type=float, nargs=3, default=(0.25, 0.25, 0.9),
            help="SO-100 Arena camera eye position (x y z) in meters",
        )
        parser.add_argument(
            "--camera_target", type=float, nargs=3, default=(0.15, 0.0, 0.45),
            help="SO-100 Arena camera look-at target (x y z) in meters",
        )
        parser.add_argument(
            "--camera_width", type=int, default=640,
            help="SO-100 Arena camera width in pixels",
        )
        parser.add_argument(
            "--camera_height", type=int, default=480,
            help="SO-100 Arena camera height in pixels",
        )
        parser.add_argument(
            "--camera_focal_length", type=float, default=1.5,
            help="Isaac Lab pinhole camera focal length in cm",
        )
        parser.add_argument(
            "--camera_horizontal_aperture", type=float, default=20.955,
            help="Isaac Lab pinhole camera horizontal aperture in cm",
        )
        parser.add_argument(
            "--camera_vertical_aperture", type=float, default=None,
            help="Optional pinhole camera vertical aperture in cm; default preserves square pixels",
        )
