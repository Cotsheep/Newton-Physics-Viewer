"""Register USD inputs in place; all writes belong to the managed data root."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .assets import (
    ASSET_ENTRYPOINT, CHECKER_VERSION, AssetValidationError, FileDigest,
    _VERSION_PATTERN, _check_identity_case_collision, _new_attempt_id,
    _path_has_symlink, _require_package_path, collect_usd_dependencies,
    compute_asset_version, digest_dependencies, inspect_template_readiness,
    normalize_asset_identity, utc_now,
)
from .provenance import resolve_git_commit
from .storage import DataRoot, atomic_write_json, read_json


EXTERNAL_MODE = "external_readonly"


def _source_root(data_root: DataRoot, value: Path) -> Path:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        raise AssetValidationError("invalid_external_root", "External source root must be absolute")
    if _path_has_symlink(candidate, Path(candidate.anchor)):
        raise AssetValidationError("external_source_symlink", "External source root must not contain links or junctions")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise AssetValidationError("external_source_unavailable", "External source root does not exist or cannot be read") from exc
    if not root.is_dir() or root == Path(root.anchor) or root == Path.home().resolve():
        raise AssetValidationError("invalid_external_root", "Select the explicit directory containing the USD dependencies")
    if root.is_relative_to(data_root.path) or data_root.path.is_relative_to(root):
        raise AssetValidationError("external_source_output_overlap", "External source and managed output directories must be separate")
    return root


def _entrypoint(value: str) -> str:
    entrypoint = normalize_asset_identity(value)
    if Path(entrypoint).suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise AssetValidationError("invalid_asset_entrypoint", "External entrypoint must be a USD, USDA or USDC file")
    return entrypoint


def external_record_path(data_root: DataRoot, identity: str, version: str) -> Path:
    # resolve_managed alone would normalize away a link into another managed
    # subtree. Check the original metadata path before resolving it as well.
    candidate = data_root.location("asset_references").joinpath(*identity.split("/"), version, "registration.json")
    if _path_has_symlink(candidate, data_root.path):
        raise AssetValidationError("external_registry_symlink", "External registration metadata must not contain links")
    return data_root.resolve_managed("asset_references", *identity.split("/"), version, "registration.json")


def _read_record(path: Path) -> dict[str, Any]:
    try:
        record = read_json(path)
        identity = normalize_asset_identity(record["identity"])
        version = record["version"]
        if (record["schema_version"] != 1 or record["storage_mode"] != EXTERNAL_MODE
                or not _VERSION_PATTERN.fullmatch(version)
                or path.parent.name != version
                or not isinstance(record["source_root"], str)
                or not isinstance(record["source_name"], str)
                or not isinstance(record["registered_at"], str)):
            raise ValueError("Invalid registration fields")
        _entrypoint(record["entrypoint"])
        if path.parent.parent.parts[-len(identity.split("/")):] != tuple(identity.split("/")):
            raise ValueError("Registration identity differs from its directory")
        return record
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AssetValidationError("external_registration_invalid", "External registration metadata is missing or invalid") from exc


def list_external_asset_versions(data_root: DataRoot) -> list[dict[str, str]]:
    root = data_root.location("asset_references")
    if not root.exists():
        return []
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise AssetValidationError("external_registry_symlink", "External registry must be a real directory")
    rows = []
    for path in sorted(root.rglob("registration.json")):
        if _path_has_symlink(path, data_root.path):
            raise AssetValidationError("external_registry_symlink", "External registration metadata must not contain links")
        record = _read_record(path)
        expected = external_record_path(data_root, record["identity"], record["version"])
        if expected != path:
            raise AssetValidationError("external_registration_invalid", "Registration is outside its expected location")
        rows.append({"identity": record["identity"], "version": record["version"],
                     "storage_mode": EXTERNAL_MODE, "source_name": record["source_name"]})
    return rows


def _states(root: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in files:
        path = _require_package_path(root, root / record["path"])
        info = path.stat()
        result.append({"path": record["path"], "size": info.st_size,
                       "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
                       "inode": info.st_ino, "device": info.st_dev})
    return result


def _external_version(entrypoint: str, records: tuple[FileDigest, ...]) -> str:
    # Two different USD entrypoints can share exactly the same dependency set.
    # Bind entrypoint selection as well as file content to the selected version.
    content = f"external-readonly-v1\0{entrypoint}\0{compute_asset_version(records)}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _scan(root: Path, entrypoint: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    _require_package_path(root, root / entrypoint)
    dependencies = collect_usd_dependencies(root, entrypoint, refresh_layers=True)
    before = _states(root, [{"path": path.relative_to(root).as_posix()} for path in dependencies])
    records = digest_dependencies(root, dependencies)
    files = [asdict(item) for item in records]
    after = _states(root, files)
    if before != after:
        raise AssetValidationError("external_asset_changed", "External files changed during inspection; retry after the owner finishes updating")
    return _external_version(entrypoint, records), files, after


def _refresh_verified_layers(root: Path, files: list[dict[str, Any]]) -> None:
    from pxr import Sdf

    # Refresh after hashing as well: a source might change while the first USD
    # dependency scan was reading it, before its file states were captured.
    for item in files:
        cached = Sdf.Layer.Find(str(root / item["path"]))
        if cached is not None:
            cached.Reload(force=True)


def register_external_asset(
    data_root: DataRoot, identity_value: str, *, source_root: Path,
    entrypoint: str = ASSET_ENTRYPOINT, source_name: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Store metadata only, including readiness failures; never alter the source."""
    data_root.require_initialized()
    # Reject output overlap before writing even a failure report.
    root = _source_root(data_root, source_root)
    report: dict[str, Any] = {
        "schema_version": 1, "attempt_id": _new_attempt_id(), "created_at": utc_now(),
        "checker_version": CHECKER_VERSION, "git_commit": resolve_git_commit(git_commit),
        "storage_mode": EXTERNAL_MODE, "status": "failed",
    }
    try:
        identity = normalize_asset_identity(identity_value)
        entrypoint = _entrypoint(entrypoint)
        if not source_name.strip() or any(ord(char) < 32 for char in source_name):
            raise AssetValidationError("invalid_source_name", "Provide a readable source name")
        report["asset_identity"] = identity
        _check_identity_case_collision(data_root, identity)
        from .assets import list_asset_versions

        if any(row["identity"] == identity and row.get("storage_mode") != EXTERNAL_MODE
               for row in list_asset_versions(data_root)):
            raise AssetValidationError("asset_storage_mode_conflict", "This identity is already managed; choose a different identity")
        for row in list_external_asset_versions(data_root):
            if row["identity"] == identity:
                old = _read_record(external_record_path(data_root, identity, row["version"]))
                if (old["source_root"], old["entrypoint"], old["source_name"]) != (str(root), entrypoint, source_name.strip()):
                    raise AssetValidationError("external_source_binding_conflict", "This identity already refers to a different source; choose a different identity")

        version, files, states = _scan(root, entrypoint)
        _refresh_verified_layers(root, files)
        readiness = inspect_template_readiness(root / entrypoint)
        from .asset_parameters import snapshot_physics_parameters

        parameters = snapshot_physics_parameters(root / entrypoint)
        if _states(root, files) != states:
            raise AssetValidationError("external_asset_changed", "External files changed during inspection")
        record = {
            "schema_version": 1, "identity": identity, "version": version,
            "entrypoint": entrypoint, "source_root": str(root),
            "source_name": source_name.strip(), "storage_mode": EXTERNAL_MODE,
            "registered_at": utc_now(), "checker_version": CHECKER_VERSION,
            "git_commit": report["git_commit"], "dependencies": files,
            "readiness": readiness, "physics_parameters": parameters,
        }
        report.update({"asset_version": version, "source_name": source_name.strip(),
                       "entrypoint": entrypoint, "dependencies": files,
                       "dependency_count": len(files), "total_bytes": sum(item["size"] for item in files),
                       "template_readiness": readiness,
                       "ready_templates": [name for name, value in readiness.items() if value["status"] == "ready"]})
        path = external_record_path(data_root, identity, version)
        if path.exists():
            report["status"] = "duplicate"
        else:
            atomic_write_json(path, record)
            report["status"] = "registered"
        report["message"] = "Only metadata was registered; source files remain in place. Registration does not imply experiment readiness."
    except AssetValidationError as exc:
        report["error"] = {"code": exc.code, "message": str(exc)}
    except OSError:
        report["error"] = {"code": "external_asset_read_failed", "message": "External files could not be read; check the source path and read permission"}
    except Exception:
        report["error"] = {"code": "external_asset_validation_failed", "message": "USD inspection failed; source files were left unchanged"}
    atomic_write_json(data_root.resolve_managed("import_reports", report["attempt_id"], "report.json"), report)
    return report


def snapshot_external_asset(data_root: DataRoot, identity: str, version: str) -> dict[str, Any]:
    record = _read_record(external_record_path(data_root, identity, version))
    root = _source_root(data_root, Path(record["source_root"]))
    actual, files, states = _scan(root, record["entrypoint"])
    if actual != version:
        raise AssetValidationError("external_asset_version_mismatch", "External source no longer matches the selected version; register its new content explicitly")
    _refresh_verified_layers(root, files)
    from .asset_parameters import snapshot_physics_parameters

    snapshot = {
        "identity": identity, "version": version, "entrypoint": record["entrypoint"],
        "dependencies": files, "readiness": inspect_template_readiness(root / record["entrypoint"]),
        "physics_parameters": snapshot_physics_parameters(root / record["entrypoint"]),
        "package_root": root, "storage_mode": EXTERNAL_MODE,
        "source_name": record["source_name"],
        "external_source": {"root": str(root), "registered_at": record["registered_at"],
                            "file_states": states, "verification": "before_load_and_after_run"},
    }
    if _states(root, files) != states:
        raise AssetValidationError("external_asset_changed", "External files changed while preparing the run")
    return snapshot


def verify_external_asset_snapshot(data_root: DataRoot, snapshot: dict[str, Any]) -> None:
    """Fail a run if shared inputs changed, even when changed bytes were restored."""
    if snapshot.get("storage_mode") != EXTERNAL_MODE:
        return
    try:
        root = _source_root(data_root, snapshot["package_root"])
        files = snapshot["dependencies"]
        if _states(root, files) != snapshot["external_source"]["file_states"]:
            raise AssetValidationError("external_asset_changed", "External source changed during this run; results cannot be assigned to the selected version")
        records = digest_dependencies(root, [root / item["path"] for item in files])
        if (_external_version(snapshot["entrypoint"], records) != snapshot["version"]
                or _states(root, files) != snapshot["external_source"]["file_states"]):
            raise AssetValidationError("external_asset_changed", "External source changed during this run; results cannot be assigned to the selected version")
    except (OSError, AssetValidationError) as exc:
        if isinstance(exc, AssetValidationError) and exc.code == "external_asset_changed":
            raise
        raise AssetValidationError("external_source_unavailable", "External source disappeared, became unreadable or violated its path boundary during this run") from exc
