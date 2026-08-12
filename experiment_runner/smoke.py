from __future__ import annotations

from pathlib import Path
from typing import Any

from .assets import snapshot_asset_version, utc_now
from .cpu_safety import prepare_cpu_smoke_environment
from .profiles import get_profile
from .provenance import resolve_git_commit
from .results import (
    build_result_index,
    create_batch_scaffold,
    create_run_scaffold,
    update_run_status,
    write_run_checksums,
)
from .storage import (
    DataRoot,
    atomic_write_json,
    atomic_write_text,
    read_json,
)
from .web import install_static_site


SMOKE_PROFILE = "mujoco-cpu-wsl-smoke-v1"


def _update_manifest(run_directory: Path, **values: Any) -> dict[str, Any]:
    manifest_path = run_directory / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(values)
    atomic_write_json(manifest_path, manifest)
    return manifest


def _update_batch_status(
    data_root: DataRoot,
    *,
    batch_id: str,
    run_id: str,
    status: str,
) -> None:
    status_path = data_root.location("batches") / batch_id / "status.json"
    document = read_json(status_path)
    document.update(
        {
            "status": status,
            "current_run_id": None,
            "completed_runs": 1,
            "updated_at": utc_now(),
        }
    )
    document["runs"][0]["status"] = status
    atomic_write_json(status_path, document)


def _append_run_log(run_directory: Path, message: str) -> None:
    with (run_directory / "run.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{utc_now()} {message}\n")


def run_cpu_smoke_drop(
    data_root: DataRoot,
    *,
    asset_identity: str,
    asset_version: str,
    git_commit: str | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one non-authoritative medium-height drop through native MuJoCo contacts."""

    prepare_cpu_smoke_environment()
    from .experiments.drop import (
        create_drop_scene,
        measure_drop_geometry,
        record_drop_case,
        render_asset_cover,
    )

    git_commit = resolve_git_commit(git_commit)
    data_root.require_initialized()
    snapshot = snapshot_asset_version(data_root, asset_identity, asset_version)
    readiness = snapshot["readiness"]["drop"]
    if readiness["status"] != "ready":
        reasons = ", ".join(readiness["reason_codes"]) or "unknown"
        raise ValueError(f"Asset version is not ready for drop: {reasons}")

    profile = get_profile(SMOKE_PROFILE)
    duration = duration_seconds or profile.case_duration_seconds
    if duration is None:
        raise ValueError("The smoke profile does not define a case duration")
    geometry = measure_drop_geometry(
        snapshot["package_root"] / snapshot["entrypoint"],
        profile=profile,
        clearance_scale=1.0,
    )
    batch = create_batch_scaffold(
        data_root,
        asset_identity=snapshot["identity"],
        asset_version=snapshot["version"],
        templates=["drop"],
        profile_name=profile.name,
        request_summary={
            "purpose": "local_cpu_smoke",
            "authoritative": False,
            "drop_clearance_scale": 1.0,
            "case_duration_seconds": duration,
        },
    )
    batch_id = batch["batch_id"]
    run_id = batch["runs"][0]["run_id"]
    public_asset = {
        key: value
        for key, value in snapshot.items()
        if key != "package_root"
    }
    run_directory = create_run_scaffold(
        data_root,
        run_id=run_id,
        batch_id=batch_id,
        template="drop",
        asset=public_asset,
        profile=profile.expanded(),
        git_commit=git_commit,
    )
    case_id = "drop-medium"
    case_directory = run_directory / "cases" / "001-medium"
    case_directory.mkdir()
    atomic_write_text(
        run_directory / "run.log",
        f"{utc_now()} CPU smoke run created; Warp CPU and software-rendering policy enabled\n",
    )
    started_at = utc_now()
    case_document: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case_id,
        "label": "中等高度（CPU 冒烟）",
        "status": "running",
        "condition": {
            "clearance_scale": 1.0,
            "clearance_m": geometry.clearance,
            "characteristic_length_m": geometry.characteristic_length,
            "effective_length_m": geometry.effective_length,
            "initial_velocity_mps": [0.0, 0.0, 0.0],
            "initial_angular_velocity_rps": [0.0, 0.0, 0.0],
        },
        "authoritative": False,
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": duration,
    }
    atomic_write_json(case_directory / "case.json", case_document)
    _update_manifest(run_directory, started_at=started_at)
    update_run_status(
        run_directory,
        status="running",
        phase="recording",
        progress="Recording the only CPU smoke case",
        current_case=case_id,
        completed_cases=0,
        total_cases=1,
    )

    try:
        cover_rendering = render_asset_cover(
            snapshot["package_root"] / snapshot["entrypoint"],
            profile=profile,
            output_path=run_directory / "asset-cover.jpg",
        )
        _append_run_log(run_directory, "Starting native-contact MuJoCo CPU drop case")
        scene = create_drop_scene(
            snapshot["package_root"] / snapshot["entrypoint"],
            profile=profile,
            clearance=geometry.clearance,
            measured_bounds=geometry.initial_bounds,
        )
        result = record_drop_case(
            scene,
            profile=profile,
            output_directory=case_directory,
            duration_seconds=duration,
        )
        finished_at = utc_now()
        case_document.update(result)
        case_document["status"] = "succeeded"
        case_document["finished_at"] = finished_at
        atomic_write_json(case_directory / "case.json", case_document)
        _append_run_log(run_directory, "CPU smoke drop case completed successfully")
        manifest = read_json(run_directory / "manifest.json")
        environment = manifest["environment"]
        environment["execution"].update(
            {
                "warp_device": "cpu",
                "rendering_device": result["rendering"]["device"],
                "opengl_renderer": result["rendering"]["renderer"],
                "opengl_vendor": result["rendering"]["vendor"],
                "asset_cover_rendering_device": cover_rendering["device"],
                "cuda_used": False,
            }
        )
        result_files = [
            "asset-cover.jpg",
            "preview.jpg",
            "run.log",
            "cases/001-medium/case.json",
            "cases/001-medium/video.mp4",
            "cases/001-medium/poster.jpg",
            "cases/001-medium/final.jpg",
            "checksums.sha256",
        ]
        _update_manifest(
            run_directory,
            finished_at=finished_at,
            result_files=result_files,
            exit_code=0,
            failure_summary=None,
            environment=environment,
        )
        update_run_status(
            run_directory,
            status="succeeded",
            phase="finished",
            progress="1/1 CPU smoke case complete",
            current_case=None,
            completed_cases=1,
            total_cases=1,
            preview_updated_at=finished_at,
            result_files=result_files,
        )
        write_run_checksums(run_directory)
        _update_batch_status(
            data_root,
            batch_id=batch_id,
            run_id=run_id,
            status="succeeded",
        )
        install_static_site(data_root)
        build_result_index(data_root)
        return {
            "batch_id": batch_id,
            "run_id": run_id,
            "case_id": case_id,
            "status": "succeeded",
            "authoritative": False,
            "profile": profile.name,
        }
    except Exception as exc:
        failed_at = utc_now()
        safe_failure = f"CPU smoke drop failed ({type(exc).__name__})"
        _append_run_log(run_directory, safe_failure)
        case_document["status"] = "failed"
        case_document["finished_at"] = failed_at
        case_document["failure"] = {
            "code": "cpu_smoke_drop_failed",
            "message": safe_failure,
        }
        atomic_write_json(case_directory / "case.json", case_document)
        _update_manifest(
            run_directory,
            finished_at=failed_at,
            result_files=[
                "run.log",
                "cases/001-medium/case.json",
                "checksums.sha256",
            ],
            exit_code=1,
            failure_summary=safe_failure,
        )
        update_run_status(
            run_directory,
            status="failed",
            phase="failed",
            progress="The CPU smoke case failed",
            current_case=None,
            completed_cases=0,
            total_cases=1,
            result_files=[
                "run.log",
                "cases/001-medium/case.json",
                "checksums.sha256",
            ],
            failure_summary=safe_failure,
        )
        write_run_checksums(run_directory)
        _update_batch_status(
            data_root,
            batch_id=batch_id,
            run_id=run_id,
            status="failed",
        )
        install_static_site(data_root)
        build_result_index(data_root)
        raise
