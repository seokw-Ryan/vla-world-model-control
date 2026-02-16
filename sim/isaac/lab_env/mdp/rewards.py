"""Custom reward terms for VLA evaluation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_above_height(
    env: ManagerBasedRLEnv,
    object_cfg: str = "cube",
    min_height: float = 0.6,
) -> torch.Tensor:
    """Reward of 1.0 if the object is above the specified height.

    Used as a success metric for pick-and-lift tasks.

    Args:
        env: The environment instance.
        object_cfg: Name of the object in the scene.
        min_height: Minimum z-height for success.

    Returns:
        Tensor of shape (num_envs,) with 1.0 for success, 0.0 otherwise.
    """
    obj = env.scene[object_cfg]
    obj_pos = obj.data.root_pos_w  # (num_envs, 3)
    return (obj_pos[:, 2] > min_height).float()
