from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .provenance import resolve_git_commit
from .storage import DataRoot, atomic_write_json


ASSET_ENTRYPOINT = "newton-mujoco.usda"
CHECKER_VERSION = "asset-readiness-v2"
_VERSION_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class AssetValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FileDigest:
    path: str
    size: int
    sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_asset_identity(value: str) -> str:
    """Normalize a user-facing inbox-relative package directory."""

    if not isinstance(value, str):
        raise AssetValidationError("invalid_asset_identity", "Asset identity must be text")
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/"))
    if not normalized or normalized.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(normalized):
        raise AssetValidationError(
            "invalid_asset_identity",
            "Asset identity must be a non-empty relative directory",
        )
    raw_parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise AssetValidationError(
            "invalid_asset_identity",
            "Asset identity contains an empty, current, or parent path segment",
        )
    if any(
        unicodedata.category(character).startswith("C")
        for part in raw_parts
        for character in part
    ):
        raise AssetValidationError(
            "invalid_asset_identity",
            "Asset identity contains a control character",
        )
    return PurePosixPath(*raw_parts).as_posix()


def _path_has_symlink(path: Path, stop: Path) -> bool:
    candidate = path
    while candidate != stop:
        if candidate.is_symlink():
            return True
        if candidate.parent == candidate:
            return True
        candidate = candidate.parent
    return stop.is_symlink()


def _require_package_path(package_root: Path, candidate: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AssetValidationError(
            "missing_asset_dependency",
            "A referenced asset dependency does not exist",
        ) from exc
    root = package_root.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AssetValidationError(
            "asset_dependency_escape",
            "A referenced asset dependency escapes the uploaded package",
        ) from exc
    if not resolved.is_file():
        raise AssetValidationError(
            "invalid_asset_dependency",
            "A referenced asset dependency is not a regular file",
        )
    if _path_has_symlink(candidate, package_root):
        raise AssetValidationError(
            "asset_dependency_symlink",
            "Symbolic links are not accepted inside an asset package",
        )
    return resolved


def collect_usd_dependencies(package_root: Path) -> tuple[Path, ...]:
    """Resolve the composed USD dependency graph and keep it inside the package."""

    package_root = package_root.resolve(strict=True)
    entrypoint = package_root / ASSET_ENTRYPOINT
    if not entrypoint.is_file():
        raise AssetValidationError(
            "missing_asset_entrypoint",
            f"Asset package must directly contain {ASSET_ENTRYPOINT}",
        )
    try:
        from pxr import Sdf, Usd, UsdUtils
    except ImportError as exc:
        raise AssetValidationError(
            "usd_runtime_unavailable",
            "OpenUSD Python bindings are required for asset acceptance",
        ) from exc

    try:
        stage = Usd.Stage.Open(str(entrypoint))
    except Exception as exc:
        raise AssetValidationError(
            "usd_stage_open_failed",
            f"{ASSET_ENTRYPOINT} could not be opened",
        ) from exc
    if stage is None:
        raise AssetValidationError(
            "usd_stage_open_failed",
            f"{ASSET_ENTRYPOINT} could not be opened",
        )

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(str(entrypoint))
        )
    except Exception as exc:
        raise AssetValidationError(
            "usd_dependency_scan_failed",
            "The USD dependency graph could not be resolved",
        ) from exc
    if unresolved:
        raise AssetValidationError(
            "unresolved_asset_dependency",
            "The USD stage contains an unresolved dependency",
        )

    candidates = [entrypoint]
    for layer in layers:
        real_path = getattr(layer, "realPath", "")
        if not real_path:
            raise AssetValidationError(
                "non_file_asset_dependency",
                "The USD stage contains a dependency without a local file",
            )
        candidates.append(Path(real_path))
    for asset in assets:
        resolved_path = getattr(asset, "resolvedPath", "")
        if not resolved_path:
            raise AssetValidationError(
                "unresolved_asset_dependency",
                "The USD stage contains an unresolved dependency",
            )
        candidates.append(Path(resolved_path))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        resolved = _require_package_path(package_root, candidate)
        relative = resolved.relative_to(package_root).as_posix()
        unique[relative] = resolved
    return tuple(unique[key] for key in sorted(unique))


def digest_dependencies(package_root: Path, dependencies: Iterable[Path]) -> tuple[FileDigest, ...]:
    root = package_root.resolve(strict=True)
    records: list[FileDigest] = []
    for dependency in dependencies:
        path = _require_package_path(root, dependency)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        records.append(
            FileDigest(
                path=path.relative_to(root).as_posix(),
                size=size,
                sha256=digest.hexdigest(),
            )
        )
    return tuple(sorted(records, key=lambda record: record.path))


def compute_asset_version(files: Iterable[FileDigest]) -> str:
    digest = hashlib.sha256()
    records = tuple(files)
    if not records:
        raise AssetValidationError(
            "empty_asset_dependency_set",
            "The asset dependency set is empty",
        )
    for record in sorted(records, key=lambda item: item.path):
        digest.update(record.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _authored_values(prims: Iterable[Any], name: str) -> list[Any]:
    values: list[Any] = []
    for prim in prims:
        attribute = prim.GetAttribute(name)
        if attribute and attribute.HasAuthoredValueOpinion():
            value = attribute.Get()
            if value is not None:
                values.append(value)
    return values


def _numbers(value: Any) -> tuple[float, ...]:
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    if isinstance(value, (int, float)):
        return (float(value),)
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _all_finite(values: tuple[float, ...]) -> bool:
    return bool(values) and all(math.isfinite(value) for value in values)


def _valid_mass(value: Any) -> bool:
    numbers = _numbers(value)
    return len(numbers) == 1 and _all_finite(numbers) and numbers[0] > 0.0


def _valid_center(value: Any) -> bool:
    numbers = _numbers(value)
    return len(numbers) == 3 and _all_finite(numbers)


def _valid_inertia(value: Any) -> bool:
    numbers = _numbers(value)
    if len(numbers) != 3 or not _all_finite(numbers) or any(item <= 0.0 for item in numbers):
        return False
    return all(numbers[index] <= sum(numbers) - numbers[index] + 1.0e-9 for index in range(3))


def _valid_quaternion(value: Any) -> bool:
    numbers = _numbers(value)
    return (
        len(numbers) == 4
        and _all_finite(numbers)
        and sum(item * item for item in numbers) > 1.0e-12
    )


def _valid_nonnegative_scalar(value: Any) -> bool:
    numbers = _numbers(value)
    return len(numbers) == 1 and _all_finite(numbers) and numbers[0] >= 0.0


def _valid_condim(value: Any) -> bool:
    numbers = _numbers(value)
    return (
        len(numbers) == 1
        and _all_finite(numbers)
        and numbers[0].is_integer()
        and int(numbers[0]) in {1, 3, 4, 6}
    )


def _valid_solref(value: Any) -> bool:
    numbers = _numbers(value)
    if len(numbers) != 2 or not _all_finite(numbers):
        return False
    first, second = numbers
    return (first > 0.0 and second >= 0.0) or (first < 0.0 and second < 0.0)


def _valid_solimp(value: Any) -> bool:
    numbers = _numbers(value)
    if len(numbers) != 5 or not _all_finite(numbers):
        return False
    dmin, dmax, width, midpoint, power = numbers
    return (
        0.0 <= dmin <= dmax <= 1.0
        and width > 0.0
        and 0.0 < midpoint < 1.0
        and power >= 1.0
    )


def _result(reason_codes: list[str], checks: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready" if not reason_codes else "not_ready",
        "reason_codes": sorted(set(reason_codes)),
        "checks": checks,
    }


def inspect_template_readiness(entrypoint: Path) -> dict[str, dict[str, Any]]:
    """Apply the first conservative per-template physics-field checker."""

    try:
        from pxr import Usd, UsdPhysics, UsdShade
    except ImportError as exc:
        raise AssetValidationError(
            "usd_runtime_unavailable",
            "OpenUSD Python bindings are required for readiness checks",
        ) from exc
    try:
        stage = Usd.Stage.Open(str(entrypoint))
    except Exception as exc:
        raise AssetValidationError(
            "usd_stage_open_failed",
            f"{ASSET_ENTRYPOINT} could not be opened",
        ) from exc
    if stage is None:
        raise AssetValidationError(
            "usd_stage_open_failed",
            f"{ASSET_ENTRYPOINT} could not be opened",
        )

    prims = list(stage.Traverse())
    rigid_bodies = [
        prim
        for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and prim.GetAttribute("physics:rigidBodyEnabled").Get() is not False
    ]
    collision_shapes = [
        prim
        for prim in prims
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.GetAttribute("physics:collisionEnabled").Get() is not False
    ]
    bound_physics_materials: dict[str, Any] = {}
    unbound_collision_shapes = 0
    for shape in collision_shapes:
        try:
            material, _relationship = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial(
                "physics"
            )
        except Exception:
            material = None
        material_prim = material.GetPrim() if material else None
        if (
            material_prim is None
            or not material_prim.IsValid()
            or not material_prim.HasAPI(UsdPhysics.MaterialAPI)
        ):
            unbound_collision_shapes += 1
            continue
        bound_physics_materials[str(material_prim.GetPath())] = material_prim

    common_reasons: list[str] = []
    common_checks: dict[str, Any] = {
        "rigid_body_count": len(rigid_bodies),
        "collision_shape_count": len(collision_shapes),
    }
    if not rigid_bodies:
        common_reasons.append("missing_rigid_body")
    if not collision_shapes:
        common_reasons.append("missing_collision_shape")

    required_body_attributes = {
        "physics:mass": _valid_mass,
        "physics:centerOfMass": _valid_center,
        "physics:diagonalInertia": _valid_inertia,
        "physics:principalAxes": _valid_quaternion,
    }
    for name, validator in required_body_attributes.items():
        values = _authored_values(rigid_bodies, name)
        common_checks[name] = {"authored": len(values), "required": len(rigid_bodies)}
        if len(values) != len(rigid_bodies):
            common_reasons.append("missing_required_physics_attribute")
        elif not all(validator(value) for value in values):
            common_reasons.append("invalid_required_physics_attribute")

    drop_reasons = list(common_reasons)
    drop_checks = dict(common_checks)
    for name, validator in {
        "mjc:solref": _valid_solref,
        "mjc:solimp": _valid_solimp,
    }.items():
        values = _authored_values(collision_shapes, name)
        drop_checks[name] = {"authored": len(values), "required": len(collision_shapes)}
        if len(values) != len(collision_shapes):
            drop_reasons.append("missing_required_physics_attribute")
        elif not all(validator(value) for value in values):
            drop_reasons.append("invalid_required_physics_attribute")

    slope_reasons = list(common_reasons)
    slope_checks = dict(common_checks)
    slope_checks["physics_material_bindings"] = {
        "bound_materials": len(bound_physics_materials),
        "unbound_collision_shapes": unbound_collision_shapes,
    }
    if unbound_collision_shapes:
        slope_reasons.append("missing_physics_material_binding")
    physics_materials = list(bound_physics_materials.values())
    friction_values = _authored_values(physics_materials, "physics:dynamicFriction")
    friction_numbers = [_numbers(value) for value in friction_values]
    unique_frictions = {
        round(numbers[0], 12)
        for numbers in friction_numbers
        if len(numbers) == 1 and _all_finite(numbers)
    }
    slope_checks["physics:dynamicFriction"] = {
        "authored": len(friction_values),
        "unique_values": len(unique_frictions),
    }
    if len(friction_values) != len(physics_materials) or not friction_values:
        slope_reasons.append("missing_required_physics_attribute")
    elif (
        not all(_valid_nonnegative_scalar(value) for value in friction_values)
        or len(unique_frictions) != 1
    ):
        slope_reasons.append("invalid_required_physics_attribute")

    for name, validator in {
        "mjc:rollingfriction": _valid_nonnegative_scalar,
    }.items():
        values = _authored_values(physics_materials, name)
        slope_checks[name] = {"authored": len(values)}
        if len(values) != len(physics_materials) or not values:
            slope_reasons.append("missing_required_physics_attribute")
        elif not all(validator(value) for value in values):
            slope_reasons.append("invalid_required_physics_attribute")

    condim_values = _authored_values(collision_shapes, "mjc:condim")
    slope_checks["mjc:condim"] = {
        "authored": len(condim_values),
        "required": len(collision_shapes),
    }
    if len(condim_values) != len(collision_shapes):
        slope_reasons.append("missing_required_physics_attribute")
    elif not all(_valid_condim(value) for value in condim_values):
        slope_reasons.append("invalid_required_physics_attribute")

    return {
        "drop": _result(drop_reasons, drop_checks),
        "slope_friction": _result(slope_reasons, slope_checks),
    }


def list_asset_versions(data_root: DataRoot) -> list[dict[str, str]]:
    assets_root = data_root.location("assets")
    versions: list[dict[str, str]] = []
    if not assets_root.is_dir():
        return versions
    for entrypoint in assets_root.rglob(ASSET_ENTRYPOINT):
        version_directory = entrypoint.parent
        if not _VERSION_PATTERN.fullmatch(version_directory.name):
            continue
        identity_path = version_directory.parent.relative_to(assets_root)
        if not identity_path.parts:
            continue
        versions.append(
            {
                "identity": identity_path.as_posix(),
                "version": version_directory.name,
            }
        )
    return sorted(versions, key=lambda item: (item["identity"], item["version"]))


def snapshot_asset_version(
    data_root: DataRoot,
    identity_value: str,
    version: str,
) -> dict[str, Any]:
    """Revalidate one immutable asset version before creating a run."""

    identity = normalize_asset_identity(identity_value)
    if not _VERSION_PATTERN.fullmatch(version):
        raise AssetValidationError(
            "invalid_asset_version",
            "Asset version must be a complete SHA-256 value",
        )
    package_root = data_root.resolve_managed(
        "assets",
        *identity.split("/"),
        version,
    )
    if not package_root.is_dir():
        raise AssetValidationError(
            "asset_version_not_found",
            "The selected immutable asset version does not exist",
        )
    dependencies = collect_usd_dependencies(package_root)
    file_digests = digest_dependencies(package_root, dependencies)
    actual_version = compute_asset_version(file_digests)
    if actual_version != version:
        raise AssetValidationError(
            "asset_version_hash_mismatch",
            "The immutable asset content no longer matches its version",
        )
    return {
        "identity": identity,
        "version": version,
        "entrypoint": ASSET_ENTRYPOINT,
        "dependencies": [asdict(record) for record in file_digests],
        "readiness": inspect_template_readiness(package_root / ASSET_ENTRYPOINT),
        "package_root": package_root,
    }


def _check_identity_case_collision(data_root: DataRoot, identity: str) -> None:
    folded = identity.casefold()
    for existing in list_asset_versions(data_root):
        if existing["identity"].casefold() == folded and existing["identity"] != identity:
            raise AssetValidationError(
                "asset_identity_case_collision",
                "Asset identity differs from an existing identity only by letter case",
            )


def _make_tree_readonly(root: Path) -> None:
    if os.name != "posix":
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        current = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(current & ~0o222)
        else:
            path.chmod(current & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def _new_attempt_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"import-{stamp}-{uuid.uuid4().hex[:8]}"


def accept_asset(
    data_root: DataRoot,
    identity_value: str,
    *,
    git_commit: str | None = None,
    enforce_readonly: bool = True,
) -> dict[str, Any]:
    """Validate and atomically move one inbox package into the immutable store."""

    data_root.require_initialized()
    git_commit = resolve_git_commit(git_commit)
    attempt_id = _new_attempt_id()
    report_path = data_root.location("import_reports") / attempt_id / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "created_at": utc_now(),
        "checker_version": CHECKER_VERSION,
        "git_commit": git_commit,
        "status": "failed",
    }

    try:
        identity = normalize_asset_identity(identity_value)
        report["asset_identity"] = identity
        _check_identity_case_collision(data_root, identity)
        uploaded_relative_path = identity_value.replace("\\", "/")
        inbox_root = data_root.location("inbox")
        package_root = inbox_root.joinpath(*uploaded_relative_path.split("/"))
        if not package_root.is_dir():
            raise AssetValidationError(
                "asset_package_not_found",
                "The selected asset package does not exist in the inbox",
            )
        if _path_has_symlink(package_root, inbox_root) or any(
            path.is_symlink() for path in package_root.rglob("*")
        ):
            raise AssetValidationError(
                "asset_package_symlink",
                "Symbolic links are not accepted in an asset package",
            )
        try:
            package_root.resolve(strict=True).relative_to(inbox_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as exc:
            raise AssetValidationError(
                "asset_package_path_escape",
                "The selected asset package escapes the inbox",
            ) from exc

        dependencies = collect_usd_dependencies(package_root)
        file_digests = digest_dependencies(package_root, dependencies)
        version = compute_asset_version(file_digests)
        readiness = inspect_template_readiness(package_root / ASSET_ENTRYPOINT)
        ready_templates = [
            name for name, result in readiness.items() if result["status"] == "ready"
        ]

        report.update(
            {
                "asset_version": version,
                "dependency_count": len(file_digests),
                "total_bytes": sum(record.size for record in file_digests),
                "dependencies": [asdict(record) for record in file_digests],
                "template_readiness": readiness,
            }
        )
        if not ready_templates:
            raise AssetValidationError(
                "no_ready_experiment_template",
                "The asset is not ready for any approved experiment template",
            )

        destination = data_root.resolve_managed(
            "assets",
            *identity.split("/"),
            version,
        )
        if destination.exists():
            report["status"] = "duplicate"
            report["ready_templates"] = ready_templates
            report["message"] = "This exact asset version already exists; the inbox copy was left unchanged"
            atomic_write_json(report_path, report)
            return report

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(package_root, destination)
        except OSError as exc:
            raise AssetValidationError(
                "asset_atomic_move_failed",
                "The asset package could not be moved atomically into the immutable store",
            ) from exc
        if enforce_readonly:
            _make_tree_readonly(destination)

        report["status"] = "accepted" if len(ready_templates) == len(readiness) else "partially_ready"
        report["ready_templates"] = ready_templates
        report["accepted_at"] = utc_now()
        report["message"] = "Asset version accepted into the immutable store"
    except AssetValidationError as exc:
        report["error"] = {
            "code": exc.code,
            "message": str(exc),
        }
    except Exception:
        report["error"] = {
            "code": "unexpected_asset_validation_error",
            "message": "An unexpected error interrupted asset validation",
        }

    atomic_write_json(report_path, report)
    return report
