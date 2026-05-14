"""Shared helpers and wrappers used by both sim and robot code."""

from .openvla import build_openvla_config, load_yaml
from .paths import (
    LEROBOT_SRC,
    PROJECT_ROOT,
    add_lerobot_import_paths_to_sys_path,
    add_project_root_to_sys_path,
    extend_pythonpath,
    project_path,
    resolve_project_path,
    stderr_log,
)

try:  # Keep pure helper imports available in minimal Python envs.
    from .openvla_wrapper import OpenVLAConfig, OpenVLAWrapper, VLAAction
except ImportError:  # pragma: no cover
    OpenVLAConfig = None
    OpenVLAWrapper = None
    VLAAction = None

__all__ = [
    "LEROBOT_SRC",
    "PROJECT_ROOT",
    "OpenVLAConfig",
    "OpenVLAWrapper",
    "VLAAction",
    "add_lerobot_import_paths_to_sys_path",
    "add_project_root_to_sys_path",
    "build_openvla_config",
    "extend_pythonpath",
    "load_yaml",
    "project_path",
    "resolve_project_path",
    "stderr_log",
]
