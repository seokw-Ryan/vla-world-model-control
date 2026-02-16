"""Isaac Lab environment for VLA evaluation with Franka Panda."""

import gymnasium as gym

from .franka_vla_env_cfg import FrankaVLAEnvCfg

gym.register(
    id="FrankaVLA-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": FrankaVLAEnvCfg},
    disable_env_checker=True,
)
