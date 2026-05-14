from __future__ import annotations

import os
import sys
import unittest

from isaaclab_arena_vla.utils.paths import (
    LEROBOT_SRC,
    PROJECT_ROOT,
    add_lerobot_import_paths_to_sys_path,
    add_project_root_to_sys_path,
    extend_pythonpath,
    project_path,
    resolve_project_path,
)


class PathsTest(unittest.TestCase):
    def test_project_path_builds_from_repo_root(self) -> None:
        self.assertEqual(project_path("configs", "vla"), PROJECT_ROOT / "configs" / "vla")

    def test_resolve_project_path_keeps_absolute_paths(self) -> None:
        absolute = PROJECT_ROOT / "README.md"
        self.assertEqual(resolve_project_path(str(absolute)), absolute)

    def test_resolve_project_path_expands_relative_paths(self) -> None:
        self.assertEqual(resolve_project_path("README.md"), PROJECT_ROOT / "README.md")

    def test_add_project_root_to_sys_path_is_idempotent(self) -> None:
        root_str = str(PROJECT_ROOT)
        while root_str in sys.path:
            sys.path.remove(root_str)

        add_project_root_to_sys_path()
        add_project_root_to_sys_path()

        self.assertEqual(sys.path.count(root_str), 1)

    def test_add_lerobot_import_paths_to_sys_path_is_idempotent(self) -> None:
        root_str = str(PROJECT_ROOT)
        lerobot_str = str(LEROBOT_SRC)
        for path_str in (root_str, lerobot_str):
            while path_str in sys.path:
                sys.path.remove(path_str)

        add_lerobot_import_paths_to_sys_path()
        add_lerobot_import_paths_to_sys_path()

        self.assertEqual(sys.path.count(root_str), 1)
        self.assertEqual(sys.path.count(lerobot_str), 1)

    def test_extend_pythonpath_prepends_paths(self) -> None:
        env = extend_pythonpath({"PYTHONPATH": "existing"}, PROJECT_ROOT, LEROBOT_SRC)
        self.assertEqual(
            env["PYTHONPATH"],
            os.pathsep.join([str(PROJECT_ROOT), str(LEROBOT_SRC), "existing"]),
        )


if __name__ == "__main__":
    unittest.main()
