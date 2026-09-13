from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DATA_ROOT_ENV = "NEWTON_DATA_ROOT"

LAYOUT = {
    "inbox": Path("inbox"),
    "assets": Path("assets"),
    "asset_trash": Path("asset-trash"),
    "import_reports": Path("import-reports"),
    "batches": Path("runtime") / "batches",
    "runs": Path("runtime") / "runs",
    "trash": Path("trash"),
    "web": Path("web"),
}

_ALLOWED_TOP_LEVEL = {
    "inbox",
    "assets",
    "asset-trash",
    "import-reports",
    "runtime",
    "trash",
    "web",
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def atomic_write_bytes(path: Path, content: bytes, *, mode: int | None = None) -> None:
    """Replace a file atomically without exposing partially-written content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> None:
    atomic_write_bytes(path, content.encode(encoding), mode=mode)


def atomic_write_json(path: Path, content: Any, *, mode: int | None = None) -> None:
    serialized = json.dumps(
        content,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(path, f"{serialized}\n", mode=mode)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class DataRoot:
    """The single configured root for assets, results, trash, and the static UI."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _resolved(self.path))

    @classmethod
    def from_environment(cls) -> DataRoot:
        value = os.environ.get(DATA_ROOT_ENV)
        if not value:
            raise ValueError(f"{DATA_ROOT_ENV} is not configured")
        return cls(Path(value))

    def location(self, name: str) -> Path:
        try:
            relative = LAYOUT[name]
        except KeyError as exc:
            choices = ", ".join(sorted(LAYOUT))
            raise ValueError(f"Unknown data-root location {name!r}; expected one of: {choices}") from exc
        return self.path / relative

    def validate_location(
        self,
        *,
        source_root: Path | None = None,
        separate_from: Iterable[Path] = (),
    ) -> None:
        """Revalidate location boundaries before every use of a managed root."""

        root = self.path
        anchor = Path(root.anchor).resolve(strict=False)
        if root == anchor:
            raise ValueError("The filesystem root cannot be used as NEWTON_DATA_ROOT")
        if root == Path.home().resolve(strict=False):
            raise ValueError("The user home directory itself cannot be used as NEWTON_DATA_ROOT")

        boundaries = list(separate_from)
        from .paths import source_root as project_source_root
        boundaries.append(project_source_root())
        if source_root is not None:
            boundaries.append(source_root)
        for boundary in boundaries:
            other = _resolved(boundary)
            if root == other or _is_relative_to(root, other) or _is_relative_to(other, root):
                raise ValueError("The source/viewer roots and NEWTON_DATA_ROOT must be separate")

        if root.exists() and root.is_symlink():
            raise ValueError("NEWTON_DATA_ROOT cannot be a symbolic link")
        if root.exists() and not root.is_dir():
            raise ValueError("NEWTON_DATA_ROOT must be a directory")
        if root.exists():
            unrelated = sorted(
                child.name for child in root.iterdir() if child.name not in _ALLOWED_TOP_LEVEL
            )
            if unrelated:
                names = ", ".join(unrelated[:5])
                suffix = "…" if len(unrelated) > 5 else ""
                raise ValueError(
                    f"NEWTON_DATA_ROOT contains unrelated existing entries: {names}{suffix}"
                )
        for relative in LAYOUT.values():
            target = root
            for part in relative.parts:
                target = target / part
                if target.is_symlink() or target.is_junction():
                    raise ValueError(f"Managed data directory cannot be a symbolic link: {relative}")
                if not target.resolve().is_relative_to(root):
                    raise ValueError(f"Managed data directory escapes its root: {relative}")

    def require_existing_owned_directory(self) -> None:
        """Require an administrator-prepared target before server configuration writes."""

        if not self.path.exists():
            raise ValueError("The administrator-approved server data root must already exist")
        if self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("The administrator-approved server data root must be a real directory")
        if hasattr(os, "geteuid") and self.path.stat().st_uid != os.geteuid():
            raise ValueError("The server data root must belong to the current personal account")
        if not os.access(self.path, os.W_OK | os.X_OK):
            raise ValueError("The server data root is not writable by the current personal account")

    def initialize(
        self,
        *,
        source_root: Path | None = None,
        separate_from: Iterable[Path] = (),
    ) -> None:
        """Create the fixed layout after rejecting dangerous or unrelated targets."""

        self.validate_location(source_root=source_root, separate_from=separate_from)
        root = self.path

        root.mkdir(parents=True, exist_ok=True)
        for relative in LAYOUT.values():
            target = root / relative
            if target.exists() and target.is_symlink():
                raise ValueError(f"Managed data directory cannot be a symbolic link: {relative}")
            target.mkdir(parents=True, exist_ok=True)

    def require_initialized(
        self,
        *,
        source_root: Path | None = None,
        separate_from: Iterable[Path] = (),
    ) -> None:
        self.validate_location(source_root=source_root, separate_from=separate_from)
        missing = [name for name, relative in LAYOUT.items() if not (self.path / relative).is_dir()]
        if missing:
            raise ValueError(
                "NEWTON_DATA_ROOT is not initialized; missing: " + ", ".join(sorted(missing))
            )

    def resolve_managed(self, name: str, *relative_parts: str) -> Path:
        """Resolve a child path and prove that it stays within its managed directory."""

        base = self.location(name).resolve(strict=False)
        if not base.is_relative_to(self.path):
            raise ValueError("Managed directory escapes NEWTON_DATA_ROOT")
        candidate = base.joinpath(*relative_parts).resolve(strict=False)
        if not _is_relative_to(candidate, base):
            raise ValueError(f"Path escapes managed data directory {name!r}")
        return candidate
