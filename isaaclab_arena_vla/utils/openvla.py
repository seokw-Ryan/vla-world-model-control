"""Pure-Python helpers for loading OpenVLA configuration."""

from __future__ import annotations

from typing import Any

from .paths import resolve_project_path


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file from an absolute or repo-relative path."""
    import yaml

    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_openvla_config(
    config_path: str = "configs/vla/openvla_default.yaml",
    *,
    model_path: str | None = None,
):
    """Build an ``OpenVLAConfig`` instance from the shared YAML format."""
    from src.models.vla import OpenVLAConfig

    vla_yaml = load_yaml(config_path)
    model_cfg = vla_yaml.get("model", {})
    action_cfg = vla_yaml.get("action", {})
    return OpenVLAConfig(
        model_path=model_path or model_cfg.get("path", "openvla/openvla-7b"),
        unnorm_key=model_cfg.get("unnorm_key", "bridge_orig"),
        prompt_template=model_cfg.get("prompt_template", "In: {instruction}\nOut:"),
        image_size=model_cfg.get("image_size", 224),
        dtype=model_cfg.get("dtype", "bfloat16"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        position_scale=action_cfg.get("position_scale", 1.0),
        rotation_scale=action_cfg.get("rotation_scale", 1.0),
        gripper_threshold=action_cfg.get("gripper_threshold", 0.5),
        gripper_open_value=action_cfg.get("gripper_open_value", 1.0),
        gripper_close_value=action_cfg.get("gripper_close_value", -1.0),
    )
