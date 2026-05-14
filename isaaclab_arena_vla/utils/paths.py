"""Path and logging helpers shared by scripts and modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"


def _ensure_sys_path(path: Path) -> Path:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path


def add_project_root_to_sys_path() -> Path:
    """Ensure the repo root is importable."""
    return _ensure_sys_path(PROJECT_ROOT)


def add_lerobot_import_paths_to_sys_path() -> tuple[Path, Path]:
    """Ensure both the repo root and vendored LeRobot source tree are importable."""
    return add_project_root_to_sys_path(), _ensure_sys_path(LEROBOT_SRC)


def project_path(*parts: str) -> Path:
    """Build an absolute path rooted at the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(pathlike: str | os.PathLike[str]) -> Path:
    """Resolve an absolute path, or a repo-relative path when given a relative one."""
    path = Path(pathlike)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def extend_pythonpath(
    env: Mapping[str, str] | None = None,
    *paths: str | os.PathLike[str],
) -> dict[str, str]:
    """Return an env dict with the given paths prepended to PYTHONPATH."""
    base_env = dict(os.environ if env is None else env)
    extra_entries = [str(Path(path)) for path in paths]
    existing = base_env.get("PYTHONPATH")
    base_env["PYTHONPATH"] = (
        os.pathsep.join([*extra_entries, existing]) if existing else os.pathsep.join(extra_entries)
    )
    return base_env


def stderr_log(msg: str) -> None:
    """Write a line to stderr."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()
