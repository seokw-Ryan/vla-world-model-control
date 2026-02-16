"""Custom termination terms for VLA evaluation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_dropped_off_table(
    env: ManagerBasedRLEnv,
    object_cfg: str = "cube",
    x_bounds: tuple[float, float] = (-0.5, 1.5),
    y_bounds: tuple[float, float] = (-1.0, 1.0),
    z_min: float = 0.0,
) -> torch.Tensor:
    """Terminate if the object falls off the table.

    Args:
        env: The environment instance.
        object_cfg: Name of the object in the scene.
        x_bounds: Valid x range for the object.
        y_bounds: Valid y range for the object.
        z_min: Minimum z height (below = dropped).

    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    obj = env.scene[object_cfg]
    pos = obj.data.root_pos_w  # (num_envs, 3)

    out_of_bounds = (
        (pos[:, 0] < x_bounds[0]) | (pos[:, 0] > x_bounds[1]) |
        (pos[:, 1] < y_bounds[0]) | (pos[:, 1] > y_bounds[1]) |
        (pos[:, 2] < z_min)
    )
    return out_of_bounds
