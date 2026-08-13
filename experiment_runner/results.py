from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .profiles import get_profile
from .provenance import collect_runtime_environment, require_git_commit
from .storage import DataRoot, atomic_write_json, atomic_write_text, read_json


RESULT_SCHEMA_VERSION = 1
RUN_STATUSES = {
    "created",
    "running",
    "succeeded",
    "partially_succeeded",
    "failed",
    "cancelled",
    "interrupted",
}
TEMPLATES = {"drop", "slope_friction"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_identifier(prefix: str) -> str:
    if not _SAFE_ID.fullmatch(prefix):
        raise ValueError(f"Unsafe identifier prefix: {prefix!r}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _require_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a safe identifier")
    return value


def create_batch_scaffold(
    data_root: DataRoot,
    *,
    asset_identity: str,
    asset_version: str,
    templates: Iterable[str],
    profile_name: str,
    request_summary: dict[str, Any],
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Create the immutable identity of a batch and one run per template."""

    data_root.require_initialized()
    ordered_templates = list(templates)
    if not ordered_templates or len(set(ordered_templates)) != len(ordered_templates):
        raise ValueError("A batch needs one or more unique experiment templates")
    unknown = [name for name in ordered_templates if name not in TEMPLATES]
    if unknown:
        raise ValueError(f"Unknown experiment template: {unknown[0]}")
    profile = get_profile(profile_name)
    batch_id = _require_id(batch_id or new_identifier("batch"), "batch_id")
    run_records = [
        {
            "run_id": new_identifier(template.replace("_", "-")),
            "template": template,
            "order": index,
        }
        for index, template in enumerate(ordered_templates, start=1)
    ]
    document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": utc_now(),
        "asset": {
            "identity": asset_identity,
            "version": asset_version,
        },
        "profile": profile.expanded(),
        "request_summary": request_summary,
        "runs": run_records,
    }
    batch_directory = data_root.resolve_managed("batches", batch_id)
    batch_directory.mkdir(parents=False, exist_ok=False)
    atomic_write_json(batch_directory / "batch.json", document)
    atomic_write_json(
        batch_directory / "status.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "batch_id": batch_id,
            "status": "created",
            "current_run_id": None,
            "completed_runs": 0,
            "total_runs": len(run_records),
            "runs": [
                {
                    "run_id": record["run_id"],
                    "template": record["template"],
                    "status": "created",
                }
                for record in run_records
            ],
            "updated_at": utc_now(),
        },
    )
    return document


def create_run_scaffold(
    data_root: DataRoot,
    *,
    run_id: str,
    batch_id: str,
    template: str,
    asset: dict[str, Any],
    profile: dict[str, Any],
    git_commit: str,
) -> Path:
    run_id = _require_id(run_id, "run_id")
    _require_id(batch_id, "batch_id")
    if template not in TEMPLATES:
        raise ValueError(f"Unknown experiment template: {template}")
    git_commit = require_git_commit(git_commit)
    run_directory = data_root.resolve_managed("runs", run_id)
    run_directory.mkdir(parents=False, exist_ok=False)
    (run_directory / "cases").mkdir()
    created_at = utc_now()
    atomic_write_json(
        run_directory / "manifest.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "batch_id": batch_id,
            "template": template,
            "asset": asset,
            "profile": profile,
            "git_commit": git_commit,
            "environment": collect_runtime_environment(profile),
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "result_files": [],
            "exit_code": None,
            "failure_summary": None,
        },
    )
    atomic_write_json(
        run_directory / "status.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "created",
            "phase": "created",
            "progress": "Waiting to start",
            "current_case": None,
            "completed_cases": 0,
            "total_cases": 0,
            "preview_updated_at": None,
            "result_files": [],
            "failure_summary": None,
            "updated_at": created_at,
        },
    )
    return run_directory


def update_run_status(
    run_directory: Path,
    *,
    status: str,
    phase: str,
    progress: str,
    current_case: str | None,
    completed_cases: int,
    total_cases: int,
    preview_updated_at: str | None = None,
    result_files: list[str] | None = None,
    failure_summary: str | None = None,
) -> dict[str, Any]:
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {status}")
    manifest = read_json(run_directory / "manifest.json")
    document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": status,
        "phase": phase,
        "progress": progress,
        "current_case": current_case,
        "completed_cases": completed_cases,
        "total_cases": total_cases,
        "preview_updated_at": preview_updated_at,
        "result_files": result_files or [],
        "failure_summary": failure_summary,
        "updated_at": utc_now(),
    }
    atomic_write_json(run_directory / "status.json", document)
    return document


def _safe_document(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _public_case(run_id: str, case_directory: Path) -> dict[str, Any] | None:
    document = _safe_document(case_directory / "case.json")
    if document is None:
        return None
    case_id = document.get("case_id", case_directory.name)
    if not isinstance(case_id, str):
        return None
    public: dict[str, Any] = {
        "case_id": case_id,
        "directory": case_directory.name,
        "label": document.get("label", case_id),
        "status": document.get("status", "unknown"),
        "condition": document.get("condition", {}),
        "duration_seconds": document.get("duration_seconds"),
        "started_at": document.get("started_at"),
        "finished_at": document.get("finished_at"),
        "authoritative": document.get("authoritative"),
    }
    if isinstance(document.get("development_outcome"), str):
        public["development_outcome"] = document["development_outcome"]
    for key in (
        "slope_angle_degrees",
        "initial_position",
        "final_position",
        "displacement_along_slope",
        "final_linear_velocity",
        "measurement_units",
        "finite",
        "physics_steps",
    ):
        if key in document:
            public[key] = document[key]
    base_url = f"/runs/{run_id}/cases/{case_directory.name}"
    for key, filename in {
        "video_url": "video.mp4",
        "poster_url": "poster.jpg",
        "final_url": "final.jpg",
    }.items():
        if (case_directory / filename).is_file():
            public[key] = f"{base_url}/{filename}"
    return public


def _public_run(run_directory: Path) -> dict[str, Any] | None:
    run_id = run_directory.name
    if not _SAFE_ID.fullmatch(run_id):
        return None
    manifest = _safe_document(run_directory / "manifest.json")
    status = _safe_document(run_directory / "status.json")
    if manifest is None or status is None or manifest.get("run_id") != run_id:
        return None
    template = manifest.get("template")
    asset = manifest.get("asset")
    if template not in TEMPLATES or not isinstance(asset, dict):
        return None
    identity = asset.get("identity")
    version = asset.get("version")
    if not isinstance(identity, str) or not isinstance(version, str):
        return None

    case_root = run_directory / "cases"
    cases = []
    if case_root.is_dir():
        for case_directory in sorted(path for path in case_root.iterdir() if path.is_dir()):
            public_case = _public_case(run_id, case_directory)
            if public_case is not None:
                cases.append(public_case)

    profile = manifest.get("profile")
    authoritative = profile.get("authoritative") if isinstance(profile, dict) else None
    public = {
        "run_id": run_id,
        "batch_id": manifest.get("batch_id"),
        "template": template,
        "asset_identity": identity,
        "asset_version": version,
        "profile_name": profile.get("name") if isinstance(profile, dict) else None,
        "authoritative": authoritative,
        "created_at": manifest.get("created_at"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "status": status.get("status", "unknown"),
        "progress": status.get("progress"),
        "failure_summary": status.get("failure_summary"),
        "cases": cases,
    }
    if (run_directory / "preview.jpg").is_file():
        public["preview_url"] = f"/runs/{run_id}/preview.jpg"
    if (run_directory / "checksums.sha256").is_file():
        public["checksums_url"] = f"/runs/{run_id}/checksums.sha256"
    if (run_directory / "asset-cover.jpg").is_file():
        public["asset_cover_url"] = f"/runs/{run_id}/asset-cover.jpg"
    return public


def build_result_index(data_root: DataRoot) -> dict[str, Any]:
    """Rebuild the browser-facing asset-first index from canonical run files."""

    data_root.require_initialized()
    runs: list[dict[str, Any]] = []
    ignored_runs: list[str] = []
    for run_directory in sorted(
        (path for path in data_root.location("runs").iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        public = _public_run(run_directory)
        if public is None:
            ignored_runs.append(run_directory.name)
        else:
            runs.append(public)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["asset_identity"], []).append(run)

    assets: list[dict[str, Any]] = []
    for identity, asset_runs in grouped.items():
        asset_runs.sort(
            key=lambda item: (item.get("created_at") or "", item["run_id"]),
            reverse=True,
        )
        latest = asset_runs[0]
        cover_url = next(
            (
                run.get("asset_cover_url")
                for run in asset_runs
                if run.get("asset_cover_url")
            ),
            None,
        )
        assets.append(
            {
                "identity": identity,
                "display_name": identity.rsplit("/", 1)[-1],
                "cover_url": cover_url,
                "templates": sorted({run["template"] for run in asset_runs}),
                "latest_status": latest["status"],
                "latest_run_at": latest.get("created_at"),
                "runs": asset_runs,
            }
        )

    assets.sort(
        key=lambda item: (item.get("latest_run_at") or "", item["identity"]),
        reverse=True,
    )
    index = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "assets": assets,
        "ignored_runs": ignored_runs,
    }
    atomic_write_json(data_root.location("web") / "index.json", index)
    return index


def write_run_checksums(run_directory: Path) -> Path:
    """Hash a finished result package, excluding the checksum list itself."""

    checksum_path = run_directory / "checksums.sha256"
    lines: list[str] = []
    for path in sorted(item for item in run_directory.rglob("*") if item.is_file()):
        if path == checksum_path or path.is_symlink():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(run_directory).as_posix()}")
    atomic_write_text(checksum_path, "\n".join(lines) + ("\n" if lines else ""))
    return checksum_path
