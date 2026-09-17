from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from .assets import normalize_asset_identity, snapshot_asset_version, utc_now
from .cpu_safety import prepare_cpu_smoke_environment
from .profiles import get_profile
from .provenance import resolve_git_commit
from .slope_policy import single_body_slope_smoke_error
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
GPU_SMOKE_PROFILE = "mujoco-warp-cuda-dt1ms-integration-smoke-v1"
GPU_VIDEO_SMOKE_PROFILE = "mujoco-warp-cuda-dt1ms-video-smoke-v1"
SLOPE_SMOKE_ANGLE_DEGREES = 25.0


def _require_single_body_slope_smoke(readiness: dict[str, Any]) -> None:
    """Apply the current local-smoke policy without changing generic readiness."""

    checks = readiness.get("checks")
    rigid_body_count = checks.get("rigid_body_count") if isinstance(checks, dict) else None
    if type(rigid_body_count) is not int or rigid_body_count != 1:
        raise ValueError(single_body_slope_smoke_error(rigid_body_count))


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


def _rendering_log_message(rendering: dict[str, Any]) -> str:
    return (
        "Viewer rendering initialized; "
        f"device={rendering['device']}; renderer={rendering['renderer']}; "
        f"vendor={rendering['vendor']}"
    )


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
        f"{utc_now()} CPU smoke run created; MuJoCo/Warp physics locked to CPU; "
        "CUDA hidden; rendering device pending Viewer initialization\n",
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
        _append_run_log(run_directory, _rendering_log_message(result["rendering"]))
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
    except (Exception, KeyboardInterrupt) as exc:
        failed_at = utc_now()
        interrupted = isinstance(exc, KeyboardInterrupt)
        terminal_status = "interrupted" if interrupted else "failed"
        safe_failure = (
            "CPU smoke drop interrupted (KeyboardInterrupt)"
            if interrupted
            else f"CPU smoke drop failed ({type(exc).__name__})"
        )
        _append_run_log(run_directory, safe_failure)
        case_document["status"] = terminal_status
        case_document["finished_at"] = failed_at
        case_document["failure"] = {
            "code": (
                "cpu_smoke_drop_interrupted"
                if interrupted
                else "cpu_smoke_drop_failed"
            ),
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
            exit_code=130 if interrupted else 1,
            failure_summary=safe_failure,
        )
        update_run_status(
            run_directory,
            status=terminal_status,
            phase=terminal_status,
            progress=(
                "The CPU smoke case was interrupted"
                if interrupted
                else "The CPU smoke case failed"
            ),
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
            status=terminal_status,
        )
        install_static_site(data_root)
        build_result_index(data_root)
        raise


def run_gpu_smoke_drop(
    data_root: DataRoot,
    *,
    asset_identity: str,
    asset_version: str,
    git_commit: str | None = None,
    gpu_runtime: Any | None = None,
    record_video: bool = False,
) -> dict[str, Any]:
    """Run one bounded, non-authoritative MJWarp CUDA integration drop."""

    from .gpu_safety import (
        initialize_logical_cuda_gpu,
        issue_gpu_execution_permit,
        require_scene_on_logical_gpu,
        require_single_warp_visible_gpu,
    )

    # Validation is deliberately ordered so neither Warp discovery nor CUDA context
    # creation can happen before all non-GPU prerequisites and the trial gate pass.
    profile = get_profile(GPU_VIDEO_SMOKE_PROFILE if record_video else GPU_SMOKE_PROFILE)
    if profile.authoritative or profile.use_mujoco_cpu:
        raise RuntimeError("The GPU integration smoke profile safety boundary is invalid")
    duration = profile.case_duration_seconds
    if duration is None:
        raise RuntimeError("The GPU integration smoke profile has no fixed case duration")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"GPU integration smoke requires Python 3.12, got "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    normalized_identity = normalize_asset_identity(asset_identity)
    normalized_version = asset_version.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized_version):
        raise ValueError("GPU smoke asset version must be a complete SHA-256 digest")
    git_commit = resolve_git_commit(git_commit)
    data_root.require_initialized()
    snapshot = snapshot_asset_version(
        data_root, normalized_identity, normalized_version
    )
    readiness = snapshot["readiness"]["drop"]
    if readiness["status"] != "ready":
        reasons = ", ".join(readiness["reason_codes"]) or "unknown"
        raise ValueError(f"Asset version is not ready for drop: {reasons}")
    gpu_permit = issue_gpu_execution_permit()
    allocation = gpu_permit.allocation
    from .release import require_gpu_source
    source_evidence = require_gpu_source(git_commit)
    from .readiness import require_locked_runtime
    locked_runtime = require_locked_runtime()

    batch = create_batch_scaffold(
        data_root,
        asset_identity=snapshot["identity"],
        asset_version=snapshot["version"],
        templates=["drop"],
        profile_name=profile.name,
        request_summary={
            "purpose": "development_gpu_integration_smoke",
            "authoritative": False,
            "single_case": True,
            "drop_clearance_scale": 1.0,
            "case_duration_seconds": duration,
            "wall_time_limit_seconds": profile.wall_time_limit_seconds,
            "device_selection": "determined_trial_gate_then_container_logical_cuda_0",
            "scheduler_gate_is_authentication": False,
            "recording_mode": profile.recording_mode,
        },
    )
    batch_id = batch["batch_id"]
    run_id = batch["runs"][0]["run_id"]
    public_asset = {key: value for key, value in snapshot.items() if key != "package_root"}
    run_directory = create_run_scaffold(
        data_root,
        run_id=run_id,
        batch_id=batch_id,
        template="drop",
        asset=public_asset,
        profile=profile.expanded(),
        git_commit=git_commit,
    )
    manifest = read_json(run_directory / "manifest.json")
    environment = manifest["environment"]
    environment["source"] = source_evidence
    environment["locked_runtime"] = locked_runtime
    environment["execution"].update(
        {
            "physics_device": "unknown",
            "warp_device": "unknown",
            "actual_compute_device": "unknown",
            "process_visible_gpu_count": "unknown",
            "gpu_count": "unknown",
            "gpu": None,
            "nvidia_driver_version": "unknown",
            "cuda_driver_api_version": "unknown",
            "cuda_toolkit_version": "unknown",
            "warp_version": "unknown",
            "cuda_used": False,
            "cpu_fallback": False,
            "cpu_fallback_allowed": False,
            "scheduler_allocation_metadata_verified": True,
            "scheduler_gate_is_authentication": False,
            "determined_trial_allocation": allocation.expanded(),
            "selected_logical_device": "unknown",
            "cuda_context_initialized": False,
            "model_device_verified": False,
            "model_compute_device": "unknown",
            "gpu_physics_started": False,
            "gpu_physics_completed": False,
            "gpu_physics_verified": False,
            "completed_physics_steps": 0,
            "solver_step_started": False,
            "solver_step_completed": False,
            "attempted_physics_step": 0,
            "uuid_verification_status": "not_checked",
            "allocated_gpu_uuid": allocation.nvidia_visible_device_uuid,
            "runtime_gpu_uuid": None,
            "container_visibility": {
                "NVIDIA_VISIBLE_DEVICES": allocation.nvidia_visible_device_uuid,
                "CUDA_VISIBLE_DEVICES": "unknown",
            },
        }
    )
    _update_manifest(run_directory, environment=environment)

    case_id = "drop-medium-gpu-integration-smoke"
    case_relative = "cases/001-medium-gpu-integration-smoke"
    case_directory = run_directory / case_relative
    case_directory.mkdir()
    atomic_write_text(
        run_directory / "run.log",
        f"{utc_now()} GPU integration smoke created after the Determined trial "
        "metadata and NVIDIA visibility misuse gate passed; this environment gate "
        "is not authentication and real isolation remains the responsibility of "
        "Determined and the NVIDIA container runtime; CUDA is not initialized yet; "
        "CPU fallback forbidden; "
        + ("headless recording required\n" if record_video else "headless recording not attempted\n"),
    )
    started_at = utc_now()
    case_document: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case_id,
        "label": "中等高度（GPU 集成冒烟，非正式）",
        "git_commit": git_commit,
        "source": source_evidence,
        "status": "running",
        "condition": {
            "clearance_scale": 1.0,
            "case_scope": "single_medium_height_drop_only",
            "initial_velocity_mps": [0.0, 0.0, 0.0],
            "initial_angular_velocity_rps": [0.0, 0.0, 0.0],
        },
        "execution": {
            "solver": profile.expanded()["solver"],
            "compute_backend": profile.expanded()["compute_backend"],
            "determined_trial_allocation": allocation.expanded(),
            "scheduler_allocation_metadata_verified": True,
            "scheduler_gate_is_authentication": False,
            "visible_gpu_count": "unknown",
            "process_visible_gpu_count": "unknown",
            "selected_logical_device": "unknown",
            "cuda_context_initialized": False,
            "model_device_verified": False,
            "model_compute_device": "unknown",
            "gpu_physics_started": False,
            "gpu_physics_completed": False,
            "gpu_physics_verified": False,
            "completed_physics_steps": 0,
            "solver_step_started": False,
            "solver_step_completed": False,
            "attempted_physics_step": 0,
            "uuid_verification_status": "not_checked",
            "allocated_gpu_uuid": allocation.nvidia_visible_device_uuid,
            "runtime_gpu_uuid": None,
            "actual_compute_device": "unknown",
            "cuda_used": False,
            "cpu_fallback": False,
            "cpu_fallback_allowed": False,
        },
        "authoritative": False,
        "interpretation": (
            "Development integration smoke only; it validates a bounded CUDA execution "
            "path and does not produce an authoritative physics conclusion."
        ),
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": duration,
        "wall_time_limit_seconds": profile.wall_time_limit_seconds,
    }
    atomic_write_json(case_directory / "case.json", case_document)
    _update_manifest(run_directory, started_at=started_at)
    update_run_status(
        run_directory,
        status="running",
        phase="simulating",
        progress="Running the only bounded GPU integration smoke case",
        current_case=case_id,
        completed_cases=0,
        total_cases=1,
    )

    def persist_execution_state(**changes: Any) -> None:
        environment["execution"].update(changes)
        case_document["execution"].update(changes)
        _update_manifest(run_directory, environment=environment)
        atomic_write_json(case_directory / "case.json", case_document)

    scene: Any | None = None
    gpu_audit: Any | None = None
    recording = {
        "status": "not_attempted",
        "reason_code": "execution_not_started" if record_video else "gpu_headless_recording_not_validated",
        "files": [],
    }
    case_document["recording"] = recording
    _update_manifest(run_directory, recording=recording)
    atomic_write_json(case_directory / "case.json", case_document)
    try:
        # The Determined/NVIDIA metadata gate above must pass before this first
        # Warp runtime operation.  Discovery does not choose a host GPU index.
        visibility = require_single_warp_visible_gpu(gpu_runtime, permit=gpu_permit)
        persist_execution_state(
            visible_gpu_count=visibility.visible_gpu_count,
            process_visible_gpu_count=visibility.visible_gpu_count,
            gpu_count=visibility.visible_gpu_count,
            selected_logical_device="cuda:0",
            container_visibility={
                "NVIDIA_VISIBLE_DEVICES": visibility.nvidia_visible_devices,
                "CUDA_VISIBLE_DEVICES": visibility.cuda_visible_devices,
            },
        )
        gpu_audit = initialize_logical_cuda_gpu(visibility, on_audit=persist_execution_state)
        persist_execution_state(
            warp_device=gpu_audit.logical_device,
            selected_logical_device=gpu_audit.logical_device,
            cuda_context_initialized=True,
            gpu={
                "model": gpu_audit.model,
                "uuid": gpu_audit.uuid,
                "allocated_uuid": allocation.nvidia_visible_device_uuid,
                "logical_device": gpu_audit.logical_device,
            },
            nvidia_driver_version=gpu_audit.nvidia_driver_version,
            cuda_driver_api_version=gpu_audit.cuda_driver_api_version,
            cuda_toolkit_version=gpu_audit.cuda_toolkit_version,
            warp_version=gpu_audit.warp_version,
            uuid_verification_status=gpu_audit.uuid_verification_status,
            uuid_warning=gpu_audit.uuid_warning,
            allocated_gpu_uuid=allocation.nvidia_visible_device_uuid,
            runtime_gpu_uuid=gpu_audit.uuid,
        )
        if record_video:
            from .experiments.gpu_recording import preflight_gpu_rendering

            recording = {"status": "running", "stage": "renderer_preflight", "files": []}
            case_document["recording"] = recording
            _update_manifest(run_directory, recording=recording)
            atomic_write_json(case_directory / "case.json", case_document)
            preflight_gpu_rendering(gpu_permit)
            recording["stage"] = "scene_setup"
            _update_manifest(run_directory, recording=recording)
            atomic_write_json(case_directory / "case.json", case_document)
        from .experiments.drop import (
            create_drop_scene,
            measure_drop_geometry,
            simulate_drop_case_without_recording,
        )

        asset_path = snapshot["package_root"] / snapshot["entrypoint"]
        geometry = measure_drop_geometry(
            asset_path,
            profile=profile,
            clearance_scale=1.0,
            gpu_permit=gpu_permit,
        )
        case_document["condition"].update(
            {
                "clearance_m": geometry.clearance,
                "characteristic_length_m": geometry.characteristic_length,
                "effective_length_m": geometry.effective_length,
            }
        )
        atomic_write_json(case_directory / "case.json", case_document)
        _append_run_log(
            run_directory,
            "Starting Newton SolverMuJoCo with native contacts on MJWarp logical cuda:0",
        )
        scene = create_drop_scene(
            asset_path,
            profile=profile,
            clearance=geometry.clearance,
            measured_bounds=geometry.initial_bounds,
            gpu_permit=gpu_permit,
        )
        actual_device = require_scene_on_logical_gpu(scene)
        persist_execution_state(
            model_device_verified=True,
            model_compute_device=actual_device,
        )

        def mark_physics_started() -> None:
            # This callback runs only after the first solver step returns.  Until
            # then, actual_compute_device and cuda_used must remain unclaimed.
            persist_execution_state(
                physics_device=actual_device,
                actual_compute_device=actual_device,
                cuda_used=True,
                gpu_physics_started=True,
                gpu_physics_verified=True,
                completed_physics_steps=1,
            )

        def step_started(number: int) -> None:
            changes = {
                "solver_step_started": True, "solver_step_completed": False,
                "attempted_physics_step": number,
            }
            environment["execution"].update(changes)
            case_document["execution"].update(changes)
            if number == 1:
                persist_execution_state(**changes)

        def step_completed(number: int) -> None:
            changes = {"solver_step_completed": True, "completed_physics_steps": number}
            environment["execution"].update(changes)
            case_document["execution"].update(changes)
            if number % 100 == 0:
                persist_execution_state(**changes)

        simulate = simulate_drop_case_without_recording
        recording_arguments: dict[str, Any] = {}
        if record_video:
            from .experiments.gpu_recording import record_gpu_drop_case

            simulate = record_gpu_drop_case
            recording = {"status": "running", "stage": "renderer_setup", "files": []}
            case_document["recording"] = recording
            _update_manifest(run_directory, recording=recording)
            atomic_write_json(case_directory / "case.json", case_document)

            def recording_progress(stage: str) -> None:
                recording["stage"] = stage

            recording_arguments = {
                "output_directory": case_directory,
                "on_recording_progress": recording_progress,
            }
        result = simulate(
            scene,
            profile=profile,
            duration_seconds=duration,
            on_physics_started=mark_physics_started,
            on_step_started=step_started,
            on_step_completed=step_completed,
            **recording_arguments,
        )
        recording = result["recording"]
        expected_steps = round(duration / profile.physics_dt)
        if result.get("physics_steps") != expected_steps:
            raise RuntimeError(
                "GPU integration smoke did not report the complete bounded step count"
            )
        persist_execution_state(
            actual_compute_device=actual_device,
            cuda_used=True,
            gpu_physics_started=True,
            gpu_physics_completed=True,
            gpu_physics_verified=True,
            completed_physics_steps=expected_steps,
        )
        finished_at = utc_now()
        case_document.update(result)
        case_document["status"] = "succeeded"
        case_document["finished_at"] = finished_at
        atomic_write_json(case_directory / "case.json", case_document)
        _append_run_log(
            run_directory,
            "GPU integration smoke completed; structured physics result saved; "
            + ("headless video saved" if record_video else
               "video intentionally absent pending GPU headless rendering validation"),
        )
        result_files = [
            "run.log",
            f"{case_relative}/case.json",
            "checksums.sha256",
        ]
        if record_video:
            result_files.extend([
                "preview.jpg", f"{case_relative}/video.mp4",
                f"{case_relative}/poster.jpg", f"{case_relative}/final.jpg",
            ])
        _update_manifest(
            run_directory,
            finished_at=finished_at,
            result_files=result_files,
            exit_code=0,
            failure_summary=None,
            environment=environment,
            recording=recording,
        )
        update_run_status(
            run_directory,
            status="succeeded",
            phase="finished",
            progress=("1/1 GPU video smoke case complete" if record_video else
                      "1/1 GPU integration smoke case complete (structured result only)"),
            current_case=None,
            completed_cases=1,
            total_cases=1,
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
            "actual_compute_device": actual_device,
            "visible_gpu_count": gpu_audit.visible_gpu_count,
            "cpu_fallback": False,
            "recording_status": recording["status"],
        }
    except (Exception, KeyboardInterrupt) as exc:
        failed_at = utc_now()
        interrupted = isinstance(exc, KeyboardInterrupt)
        terminal_status = "interrupted" if interrupted else "failed"
        expected_steps = round(duration / profile.physics_dt)
        if scene is not None:
            completed_steps = max(
                0,
                min(expected_steps, scene.completed_physics_steps),
            )
            case_document["execution"]["completed_physics_steps"] = completed_steps
            environment["execution"]["completed_physics_steps"] = completed_steps
            if (
                completed_steps >= 1
                and case_document["execution"].get("model_device_verified") is True
            ):
                actual_device = case_document["execution"]["model_compute_device"]
                failure_audit = {
                    "physics_device": actual_device,
                    "actual_compute_device": actual_device,
                    "cuda_used": True,
                    "gpu_physics_started": True,
                    "gpu_physics_verified": True,
                    "gpu_physics_completed": completed_steps == expected_steps,
                }
                case_document["execution"].update(failure_audit)
                environment["execution"].update(failure_audit)
        safe_failure = (
            "GPU integration smoke interrupted (KeyboardInterrupt)"
            if interrupted
            else f"GPU integration smoke failed ({type(exc).__name__}); CPU fallback was not attempted"
        )
        _append_run_log(run_directory, safe_failure)
        case_document["status"] = terminal_status
        case_document["finished_at"] = failed_at
        case_document["failure"] = {
            "code": (
                "gpu_integration_smoke_interrupted"
                if interrupted
                else "gpu_integration_smoke_failed"
            ),
            "message": safe_failure,
        }
        if record_video and recording["status"] == "running":
            recording = {
                "status": "interrupted" if interrupted else "failed",
                "stage": getattr(exc, "stage", recording.get("stage", "unknown")),
                "reason_code": "gpu_video_smoke_failed", "files": [],
            }
            diagnostics = getattr(exc, "diagnostics", None)
            if diagnostics is not None:
                recording["diagnostics"] = diagnostics
        case_document["recording"] = recording
        atomic_write_json(case_directory / "case.json", case_document)
        failed_files = ["run.log", f"{case_relative}/case.json", "checksums.sha256"]
        if record_video:
            for relative in (
                "preview.jpg", f"{case_relative}/video.mp4",
                f"{case_relative}/poster.jpg", f"{case_relative}/final.jpg",
            ):
                if (run_directory / relative).is_file():
                    failed_files.append(relative)
        _update_manifest(
            run_directory,
            finished_at=failed_at,
            result_files=failed_files,
            exit_code=130 if interrupted else 1,
            failure_summary=safe_failure,
            environment=environment,
            recording=recording,
        )
        update_run_status(
            run_directory,
            status=terminal_status,
            phase=terminal_status,
            progress=safe_failure,
            current_case=None,
            completed_cases=0,
            total_cases=1,
            result_files=failed_files,
            failure_summary=safe_failure,
        )
        write_run_checksums(run_directory)
        _update_batch_status(
            data_root,
            batch_id=batch_id,
            run_id=run_id,
            status=terminal_status,
        )
        install_static_site(data_root)
        build_result_index(data_root)
        raise


def run_cpu_smoke_slope(
    data_root: DataRoot,
    *,
    asset_identity: str,
    asset_version: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run one non-authoritative fixed-angle slope case through native contacts."""

    prepare_cpu_smoke_environment()
    from .experiments.drop import render_asset_cover
    from .experiments.slope import (
        create_slope_scene,
        measure_slope_geometry,
        record_slope_case,
    )

    git_commit = resolve_git_commit(git_commit)
    data_root.require_initialized()
    snapshot = snapshot_asset_version(data_root, asset_identity, asset_version)
    readiness = snapshot["readiness"]["slope_friction"]
    if readiness["status"] != "ready":
        reasons = ", ".join(readiness["reason_codes"]) or "unknown"
        raise ValueError(f"Asset version is not ready for slope_friction: {reasons}")
    _require_single_body_slope_smoke(readiness)

    profile = get_profile(SMOKE_PROFILE)
    duration = profile.case_duration_seconds
    if duration is None:
        raise ValueError("The smoke profile does not define a case duration")
    geometry = measure_slope_geometry(
        snapshot["package_root"] / snapshot["entrypoint"],
        profile=profile,
        angle_degrees=SLOPE_SMOKE_ANGLE_DEGREES,
    )
    batch = create_batch_scaffold(
        data_root,
        asset_identity=snapshot["identity"],
        asset_version=snapshot["version"],
        templates=["slope_friction"],
        profile_name=profile.name,
        request_summary={
            "purpose": "local_cpu_smoke",
            "authoritative": False,
            "single_case": True,
            "slope_angle_degrees": SLOPE_SMOKE_ANGLE_DEGREES,
            "case_duration_seconds": duration,
        },
    )
    batch_id = batch["batch_id"]
    run_id = batch["runs"][0]["run_id"]
    public_asset = {key: value for key, value in snapshot.items() if key != "package_root"}
    run_directory = create_run_scaffold(
        data_root,
        run_id=run_id,
        batch_id=batch_id,
        template="slope_friction",
        asset=public_asset,
        profile=profile.expanded(),
        git_commit=git_commit,
    )
    case_id = "slope-25deg"
    case_relative = "cases/001-slope-25deg"
    case_directory = run_directory / case_relative
    case_directory.mkdir()
    atomic_write_text(
        run_directory / "run.log",
        f"{utc_now()} CPU slope smoke run created; fixed 25-degree single case; "
        "MuJoCo/Warp physics locked to CPU; CUDA hidden; "
        "rendering device pending Viewer initialization\n",
    )
    started_at = utc_now()
    case_document: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case_id,
        "label": "25° 坡度（CPU 冒烟）",
        "status": "running",
        "condition": {
            "slope_angle_degrees": SLOPE_SMOKE_ANGLE_DEGREES,
            "characteristic_length_m": geometry.characteristic_length,
            "effective_length_m": geometry.effective_length,
            "ramp_length_m": geometry.ramp_length,
            "ramp_width_m": geometry.ramp_width,
            "ramp_thickness_m": geometry.ramp_thickness,
            "initial_surface_gap_m": geometry.surface_gap,
            "initial_velocity_mps": [0.0, 0.0, 0.0],
            "initial_angular_velocity_rps": [0.0, 0.0, 0.0],
        },
        "authoritative": False,
        "interpretation": (
            "Development smoke observation only; moved/stayed_near_start/inconclusive "
            "is not a formal friction conclusion."
        ),
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
        progress="Recording the only 25-degree CPU slope smoke case",
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
        _append_run_log(run_directory, "Starting native-contact MuJoCo CPU slope case")
        scene = create_slope_scene(
            snapshot["package_root"] / snapshot["entrypoint"],
            profile=profile,
            geometry=geometry,
        )
        result = record_slope_case(
            scene,
            profile=profile,
            geometry=geometry,
            output_directory=case_directory,
            duration_seconds=duration,
        )
        _append_run_log(run_directory, _rendering_log_message(result["rendering"]))
        finished_at = utc_now()
        case_document.update(result)
        case_document["status"] = "succeeded"
        case_document["finished_at"] = finished_at
        atomic_write_json(case_directory / "case.json", case_document)
        _append_run_log(run_directory, "CPU slope smoke case completed successfully")
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
            f"{case_relative}/case.json",
            f"{case_relative}/video.mp4",
            f"{case_relative}/poster.jpg",
            f"{case_relative}/final.jpg",
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
            progress="1/1 CPU slope smoke case complete",
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
            "slope_angle_degrees": SLOPE_SMOKE_ANGLE_DEGREES,
            "development_outcome": result["development_outcome"],
        }
    except (Exception, KeyboardInterrupt) as exc:
        failed_at = utc_now()
        interrupted = isinstance(exc, KeyboardInterrupt)
        terminal_status = "interrupted" if interrupted else "failed"
        safe_failure = (
            "CPU smoke slope interrupted (KeyboardInterrupt)"
            if interrupted
            else f"CPU smoke slope failed ({type(exc).__name__})"
        )
        _append_run_log(run_directory, safe_failure)
        case_document["status"] = terminal_status
        case_document["finished_at"] = failed_at
        case_document["failure"] = {
            "code": (
                "cpu_smoke_slope_interrupted"
                if interrupted
                else "cpu_smoke_slope_failed"
            ),
            "message": safe_failure,
        }
        atomic_write_json(case_directory / "case.json", case_document)
        failed_files = ["run.log", f"{case_relative}/case.json", "checksums.sha256"]
        _update_manifest(
            run_directory,
            finished_at=failed_at,
            result_files=failed_files,
            exit_code=130 if interrupted else 1,
            failure_summary=safe_failure,
        )
        update_run_status(
            run_directory,
            status=terminal_status,
            phase=terminal_status,
            progress=(
                "The CPU slope smoke case was interrupted"
                if interrupted
                else "The CPU slope smoke case failed"
            ),
            current_case=None,
            completed_cases=0,
            total_cases=1,
            result_files=failed_files,
            failure_summary=safe_failure,
        )
        write_run_checksums(run_directory)
        _update_batch_status(
            data_root,
            batch_id=batch_id,
            run_id=run_id,
            status=terminal_status,
        )
        install_static_site(data_root)
        build_result_index(data_root)
        raise
