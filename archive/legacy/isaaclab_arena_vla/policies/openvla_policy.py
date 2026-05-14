"""OpenVLA policy wrapper for IsaacLab Arena.

Wraps the existing OpenVLAWrapper (src/models/vla/openvla_wrapper.py) as
an Arena PolicyBase, mapping the 7D VLA output to SO-100 6D joint actions.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_vla.utils import add_project_root_to_sys_path, build_openvla_config

add_project_root_to_sys_path()


class OpenVLAPolicy(PolicyBase):
    """Arena policy that runs OpenVLA inference.

    Maps VLA 7D output (dx, dy, dz, drx, dry, drz, gripper) to
    SO-100 6D joint positions using the same mapping from standalone_so100.py:
        dx  -> shoulder_pan delta
        dy  -> shoulder_lift delta
        dz  -> elbow_flex delta
        drx -> wrist_flex delta
        dry -> wrist_roll delta
        gripper -> gripper position
    """

    # SO-100 joint limits
    JOINT_LIMITS_LOWER = np.array([-2.0, 0.0, -3.14158, -2.5, -3.14158], dtype=np.float32)
    JOINT_LIMITS_UPPER = np.array([2.0, 3.5, 0.0, 1.2, 3.14158], dtype=np.float32)
    GRIPPER_OPEN = 1.5
    GRIPPER_CLOSED = -0.1
    JOINT_DELTA_SCALE = 0.1

    def __init__(
        self,
        instruction: str = "Pick up the red cube from the table.",
        model_path: Optional[str] = None,
        vla_config_path: Optional[str] = None,
    ):
        super().__init__()
        self.instruction = instruction
        self._vla = None
        self._model_path = model_path
        self._vla_config_path = vla_config_path

    def _ensure_loaded(self):
        """Lazy-load the VLA model on first use."""
        if self._vla is not None:
            return

        from src.models.vla.openvla_wrapper import OpenVLAWrapper

        config = build_openvla_config(
            self._vla_config_path or "configs/vla/openvla_default.yaml",
            model_path=self._model_path,
        )
        self._vla = OpenVLAWrapper(config).load()

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Run VLA inference and return joint position targets.

        Expects observation to contain camera images (when cameras enabled)
        or generates a dummy image. Returns (num_envs, 6) action tensor.
        """
        self._ensure_loaded()

        device = torch.device(env.unwrapped.device)
        num_envs = env.unwrapped.num_envs

        # Get camera image from observations if available
        image = self._extract_image(observation)

        # Run VLA inference (single image — no batching in OpenVLA)
        vla_action = self._vla.predict(image, self.instruction)

        # Map 7D VLA action to 6D SO-100 joint targets
        arm_deltas = np.zeros(5, dtype=np.float32)
        arm_deltas[0] = vla_action.delta_pos[0] * self.JOINT_DELTA_SCALE
        arm_deltas[1] = vla_action.delta_pos[1] * self.JOINT_DELTA_SCALE
        arm_deltas[2] = vla_action.delta_pos[2] * self.JOINT_DELTA_SCALE
        arm_deltas[3] = vla_action.delta_rot[0] * self.JOINT_DELTA_SCALE
        arm_deltas[4] = vla_action.delta_rot[1] * self.JOINT_DELTA_SCALE

        # Get current joint state from observation
        joint_pos = observation.get("policy", {}).get("joint_pos", None)
        if joint_pos is not None:
            if isinstance(joint_pos, torch.Tensor):
                current_arm = joint_pos[0, :5].cpu().numpy()
            else:
                current_arm = np.array(joint_pos)[:5]
        else:
            current_arm = np.array([0.0, 1.0, -1.0, -0.5, 0.0], dtype=np.float32)

        # Apply deltas and clamp
        target_arm = np.clip(
            current_arm + arm_deltas,
            self.JOINT_LIMITS_LOWER,
            self.JOINT_LIMITS_UPPER,
        )

        # Gripper
        gripper_target = self.GRIPPER_OPEN if vla_action.gripper > 0.5 else self.GRIPPER_CLOSED

        # Build 6D action: [5 arm joints, 1 gripper]
        action_np = np.concatenate([target_arm, [gripper_target]])
        action = torch.tensor(action_np, dtype=torch.float32, device=device)

        # Broadcast to all envs
        return action.unsqueeze(0).expand(num_envs, -1)

    def _extract_image(self, observation: GymSpacesDict) -> np.ndarray:
        """Extract RGB image from observation dict."""
        # Try camera observations
        policy_obs = observation.get("policy", {})
        for key in ["camera_rgb", "front_camera", "wrist_camera"]:
            if key in policy_obs:
                img = policy_obs[key]
                if isinstance(img, torch.Tensor):
                    img = img[0].cpu().numpy()  # Take first env
                # Handle CHW -> HWC
                if img.ndim == 3 and img.shape[0] in (3, 4):
                    img = np.transpose(img, (1, 2, 0))
                return img

        # Fallback: dummy image (VLA will still produce actions, just poor ones)
        return np.zeros((256, 256, 3), dtype=np.uint8)
