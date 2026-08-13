from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.usd
import numpy as np
import warp as wp

from asset_viewer.app import (
    build_model,
    parse_args as parse_view_args,
)
from asset_viewer.camera import AssetBounds, compute_asset_bounds, frame_camera_on_bounds

from ..profiles import ExperimentProfile
from ..video import atomic_write_jpeg
from .recording import (
    CpuOnlyViewerGL,
    _headless_viewer,
    _render_frame,
    configure_warp_cpu_only,
    record_simulation_video,
    recording_frame_counts,
)


@dataclass
class DropScene:
    model: newton.Model
    state: newton.State
    state_next: newton.State
    control: Any
    contacts: Any
    solver: newton.solvers.SolverMuJoCo
    sim_time: float = 0.0


@dataclass(frozen=True)
class DropGeometry:
    characteristic_length: float
    effective_length: float
    clearance: float
    initial_bounds: AssetBounds


def _viewer_arguments(
    asset_path: Path,
    *,
    z_offset: float,
    profile: ExperimentProfile,
    include_ground: bool = True,
) -> Any:
    arguments = [
            str(asset_path),
            "--simulate",
            "--start-running",
            "--z",
            str(z_offset),
            "--fps",
            str(profile.video_fps),
            "--substeps",
            str(profile.physics_steps_per_video_frame),
            "--solver",
            "mujoco",
            "--iterations",
            str(profile.iterations),
            "--no-self-collisions",
            "--no-collapse-fixed-joints",
            "--floating",
            "--usd-root-mode",
            "floating",
            "--no-print-traction-force",
            "--no-joint-ui",
            "--headless",
            "--width",
            str(profile.video_width),
            "--height",
            str(profile.video_height),
        ]
    if not include_ground:
        arguments.append("--no-ground")
    return parse_view_args(arguments)


def mujoco_usd_schema_resolvers() -> list[Any]:
    """Resolve generic Newton fields and the asset's authored ``mjc:*`` fields."""

    return [newton.usd.SchemaResolverNewton(), newton.usd.SchemaResolverMjc()]


def measure_drop_geometry(
    asset_path: Path,
    *,
    profile: ExperimentProfile,
    clearance_scale: float = 1.0,
) -> DropGeometry:
    configure_warp_cpu_only()
    if clearance_scale <= 0.0 or not math.isfinite(clearance_scale):
        raise ValueError("clearance_scale must be finite and greater than zero")
    args = _viewer_arguments(asset_path, z_offset=0.0, profile=profile)
    model, state, _joint_panel = build_model(
        args,
        asset_path,
        usd_schema_resolvers=mujoco_usd_schema_resolvers(),
    )
    bounds = compute_asset_bounds(model, state)
    characteristic_length = bounds.max_extent
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("Asset collision bounds do not define a positive characteristic length")
    effective_length = float(np.clip(characteristic_length, 0.10, 1.00))
    clearance = clearance_scale * effective_length
    return DropGeometry(
        characteristic_length=characteristic_length,
        effective_length=effective_length,
        clearance=clearance,
        initial_bounds=bounds,
    )


def create_drop_scene(
    asset_path: Path,
    *,
    profile: ExperimentProfile,
    clearance: float,
    measured_bounds: AssetBounds,
) -> DropScene:
    configure_warp_cpu_only()
    if clearance <= 0.0 or not math.isfinite(clearance):
        raise ValueError("clearance must be finite and greater than zero")
    z_offset = clearance - float(measured_bounds.minimum[2])
    args = _viewer_arguments(asset_path, z_offset=z_offset, profile=profile)
    model, state, _joint_panel = build_model(
        args,
        asset_path,
        usd_schema_resolvers=mujoco_usd_schema_resolvers(),
    )
    solver = newton.solvers.SolverMuJoCo(
        model,
        iterations=profile.iterations,
        use_mujoco_cpu=profile.use_mujoco_cpu,
        use_mujoco_contacts=profile.use_mujoco_contacts,
    )
    return DropScene(
        model=model,
        state=state,
        state_next=model.state(),
        control=model.control(),
        contacts=model.contacts(),
        solver=solver,
    )


def step_drop_scene(scene: DropScene, profile: ExperimentProfile) -> None:
    scene.state.clear_forces()
    # Native MuJoCo contact generation is deliberately enabled in the solver.
    # Do not call model.collide() here or replace the asset's authored contact data.
    scene.solver.step(
        scene.state,
        scene.state_next,
        scene.control,
        scene.contacts,
        profile.physics_dt,
    )
    scene.state, scene.state_next = scene.state_next, scene.state
    scene.sim_time += profile.physics_dt


def _camera_bounds(bounds: AssetBounds) -> AssetBounds:
    minimum = bounds.minimum.copy()
    maximum = bounds.maximum.copy()
    minimum[2] = min(0.0, float(minimum[2]))
    padding = max(0.25 * bounds.max_extent, 0.05)
    minimum[:2] -= padding
    maximum[:2] += padding
    return AssetBounds(minimum, maximum)


def render_asset_cover(
    asset_path: Path,
    *,
    profile: ExperimentProfile,
    output_path: Path,
) -> dict[str, str]:
    """Render an asset-only cover without a ground plane or experiment geometry."""

    configure_warp_cpu_only()
    args = _viewer_arguments(
        asset_path,
        z_offset=0.0,
        profile=profile,
        include_ground=False,
    )
    model, state, _joint_panel = build_model(
        args,
        asset_path,
        usd_schema_resolvers=mujoco_usd_schema_resolvers(),
    )
    bounds = compute_asset_bounds(model, state)
    viewer, rendering = _headless_viewer(profile)
    try:
        viewer.set_model(model)
        viewer.set_camera(wp.vec3(1.0, -1.0, 1.0), pitch=-22.0, yaw=42.0)
        frame_camera_on_bounds(viewer.camera, bounds, padding=1.45)
        viewer.show_visual = True
        viewer.show_collision = False
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        atomic_write_jpeg(output_path, viewer.get_frame(render_ui=False).numpy())
    finally:
        viewer.close()
    return rendering


def record_drop_case(
    scene: DropScene,
    *,
    profile: ExperimentProfile,
    output_directory: Path,
    duration_seconds: float,
    initial_hold_seconds: float = 0.5,
) -> dict[str, Any]:
    initial_bounds = compute_asset_bounds(scene.model, scene.state)
    recording = record_simulation_video(
        scene,
        profile=profile,
        output_directory=output_directory,
        duration_seconds=duration_seconds,
        initial_hold_seconds=initial_hold_seconds,
        camera_bounds=_camera_bounds(initial_bounds),
        step_scene=step_drop_scene,
    )

    final_bounds = compute_asset_bounds(scene.model, scene.state)
    body_q = np.asarray(scene.state.body_q.numpy(), dtype=np.float64)
    body_qd = np.asarray(scene.state.body_qd.numpy(), dtype=np.float64)
    finite = bool(np.isfinite(body_q).all() and np.isfinite(body_qd).all())
    if not finite:
        raise RuntimeError("MuJoCo produced a non-finite body state")
    return {
        **recording,
        "initial_bounds": {
            "minimum": initial_bounds.minimum.tolist(),
            "maximum": initial_bounds.maximum.tolist(),
        },
        "final_bounds": {
            "minimum": final_bounds.minimum.tolist(),
            "maximum": final_bounds.maximum.tolist(),
        },
        "finite": finite,
    }
