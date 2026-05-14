from __future__ import annotations

import unittest

from vla_world_model_control.shared.openvla import build_openvla_config, load_yaml
from vla_world_model_control.shared.paths import PROJECT_ROOT

try:
    import yaml  # noqa: F401
except ImportError:  # pragma: no cover
    yaml = None

try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover
    numpy = None


@unittest.skipIf(yaml is None or numpy is None, "PyYAML or numpy is not installed in the active interpreter")
class OpenVLAUtilsTest(unittest.TestCase):
    def test_load_yaml_resolves_repo_relative_path(self) -> None:
        config = load_yaml("configs/vla/openvla_default.yaml")
        self.assertEqual(config["model"]["path"], "openvla/openvla-7b")

    def test_build_openvla_config_uses_yaml_defaults(self) -> None:
        config = build_openvla_config()
        self.assertEqual(config.model_path, "openvla/openvla-7b")
        self.assertEqual(config.image_size, 224)
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.gripper_open_value, 1.0)

    def test_build_openvla_config_applies_model_override(self) -> None:
        config = build_openvla_config(model_path="local/model")
        self.assertEqual(config.model_path, "local/model")

    def test_project_root_is_repo_root(self) -> None:
        self.assertTrue((PROJECT_ROOT / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
