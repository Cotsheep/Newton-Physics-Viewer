"""Export committed source for an offline trial; never deploy or submit a job."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .paths import require_separate, source_root
from .provenance import require_git_commit

MANIFEST = ".newton-source.json"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        encoding="utf-8", errors="strict", timeout=15,
    ).stdout.strip()


def source_files(root: Path) -> dict[str, str]:
    """Fingerprint executable source, static UI and dependency declarations."""
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths.extend(root.glob("*.py"))
    for name in ("experiment_runner", "asset_viewer"):
        if not (root / name).is_dir():
            raise ValueError(f"Missing source package: {name}")
        paths.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    records = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source contains an escaped or symbolic-link file")
        records[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def source_state(root: Path | None = None) -> dict:
    root = root or source_root()
    try:
        commit = require_git_commit(git(root, "rev-parse", "HEAD"))
        dirty = bool(git(root, "status", "--porcelain", "--untracked-files=all"))
    except (OSError, ValueError, subprocess.SubprocessError):
        commit, dirty = None, None
    try:
        files = source_files(root)
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError):
        digest = None
    return {"git_commit": commit, "worktree_dirty": dirty, "source_sha256": digest}


def require_gpu_source(commit: str, root: Path | None = None) -> dict:
    commit = require_git_commit(commit)
    root = root or source_root()
    state = source_state(root)
    if state["git_commit"] is not None:
        if state["worktree_dirty"] is not False or state["git_commit"] != commit:
            raise ValueError("GPU execution requires a clean checkout at the exact deployed commit")
        if state["source_sha256"] is None:
            raise ValueError("Cannot fingerprint the deployed source")
        return {**state, "verification": "clean_git_checkout"}
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["git_commit"] != commit:
            raise ValueError("Source export commit does not match the requested commit")
        files = source_files(root)
        if not files or files != manifest["files"]:
            raise ValueError("Source export differs from its committed release manifest")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("GPU execution requires a verified export made by experiment_runner.release") from exc
    return {**state, "git_commit": commit, "worktree_dirty": False, "verification": "verified_source_export"}


def export_source(output: Path, root: Path | None = None) -> Path:
    root = (root or source_root()).resolve()
    output = output.expanduser().resolve()
    require_separate(root, output)
    commit = require_git_commit(git(root, "rev-parse", "HEAD"))
    require_gpu_source(commit, root)
    if output.exists():
        raise ValueError("Release output must be a new directory; exports are never overwritten")
    # Export committed blobs rather than a working tree that could change mid-copy.
    blobs = []
    for entry in filter(None, git(root, "ls-tree", "-r", "-z", commit).split("\0")):
        meta, name = entry.split("\t", 1)
        mode, kind, _sha = meta.split()
        target = output / name
        if mode not in {"100644", "100755"} or kind != "blob" or not target.resolve().is_relative_to(output):
            raise ValueError("Release must contain only ordinary committed files")
        content = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{name}"],
            check=True, capture_output=True, timeout=15,
        ).stdout
        blobs.append((target, content))
    output.mkdir(parents=True, exist_ok=False)
    for target, content in blobs:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {"schema_version": 1, "git_commit": commit, "files": source_files(output)}
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_gpu_source(commit, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New source export directory outside the checkout")
    args = parser.parse_args()
    print(export_source(args.output))


if __name__ == "__main__":
    main()
