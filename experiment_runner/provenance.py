from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_PACKAGES = ("newton", "warp-lang", "mujoco", "mujoco-warp", "usd-core")


def require_git_commit(value: str) -> str:
    normalized = value.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(normalized):
        raise ValueError("Git commit must be a complete 40-character hexadecimal commit ID")
    return normalized


def resolve_git_commit(value: str | None = None, *, source_root: Path | None = None) -> str:
    """Return an explicit commit or discover the commit containing this source tree."""

    if value is not None:
        return require_git_commit(value)
    root = source_root or Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            "Cannot determine the deployed Git commit; pass --git-commit with the exact deployed commit"
        ) from exc
    if completed.returncode != 0:
        raise ValueError(
            "Cannot determine the deployed Git commit; pass --git-commit with the exact deployed commit"
        )
    return require_git_commit(completed.stdout.strip())


def collect_runtime_environment(profile: dict[str, Any]) -> dict[str, Any]:
    software: dict[str, str] = {"python": platform.python_version()}
    for package in _RUNTIME_PACKAGES:
        try:
            software[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            software[package] = "not-installed"

    cpu_profile = bool(profile.get("use_mujoco_cpu"))
    return {
        "software": software,
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "execution": {
            "physics_device": "cpu" if cpu_profile else "gpu-authorization-required",
            "warp_device": "cpu" if cpu_profile else None,
            "rendering_device": None,
            "opengl_renderer": None,
            "cuda_driver": None,
            "gpu": None,
            "cuda_used": False,
        },
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
    }
