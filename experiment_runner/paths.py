"""Shared path boundaries; importing this module never loads the physics stack."""
from pathlib import Path


def source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def require_separate(first: Path, second: Path) -> None:
    first, second = first.expanduser().resolve(), second.expanduser().resolve()
    if first == second or first.is_relative_to(second) or second.is_relative_to(first):
        raise ValueError("Viewer、试验数据目录和源码必须相互独立 (separate roots required)")


def require_viewer_source(path: Path, data_root: Path | None = None) -> None:
    require_separate(path, source_root())
    if data_root is not None:
        require_separate(path, data_root)
