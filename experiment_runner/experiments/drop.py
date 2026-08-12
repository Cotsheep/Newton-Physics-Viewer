from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.viewer
import numpy as np
import warp as wp

from asset_viewer.app import (
    build_model,
    parse_args as parse_view_args,
)
from asset_viewer.camera import AssetBounds, compute_asset_bounds, frame_camera_on_bounds

from ..cpu_safety import require_software_opengl_renderer
from ..profiles import ExperimentProfile
from ..video import H264VideoWriter, atomic_write_jpeg


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


def configure_warp_cpu_only() -> None:
    """Override Warp's CUDA-first default before creating any model arrays."""

    wp.set_device("cpu")
    device = wp.get_device()
    if not bool(getattr(device, "is_cpu", False)):
        raise RuntimeError("CPU smoke could not lock Warp to the CPU device")


def recording_frame_counts(
    *,
    duration_seconds: float,
    initial_hold_seconds: float,
    fps: int,
) -> tuple[int, int, int]:
    """Return total, hold, and simulated frames without consuming physics time for the hold."""

    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise ValueError("duration_seconds must be finite and greater than zero")
    if initial_hold_seconds < 0.0 or not math.isfinite(initial_hold_seconds):
        raise ValueError("initial_hold_seconds must be finite and non-negative")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    simulation_frames = round(duration_seconds * fps)
    hold_frames = round(initial_hold_seconds * fps)
    return hold_frames + simulation_frames, hold_frames, simulation_frames


def _opengl_identity() -> tuple[str, str]:
    from OpenGL import GL

    def decoded(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or "unknown"

    return decoded(GL.glGetString(GL.GL_RENDERER)), decoded(GL.glGetString(GL.GL_VENDOR))


def _headless_viewer(profile: ExperimentProfile) -> tuple[newton.viewer.ViewerGL, dict[str, str]]:
    viewer = newton.viewer.ViewerGL(
        width=profile.video_width,
        height=profile.video_height,
        headless=True,
        paused=True,
    )
    try:
        renderer, vendor = _opengl_identity()
        software_required = os.environ.get("NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL") == "1"
        if software_required:
            renderer = require_software_opengl_renderer(renderer)
        return viewer, {
            "device": "software-cpu" if software_required else "system-opengl",
            "renderer": renderer,
            "vendor": vendor,
        }
    except Exception:
        viewer.close()
        raise


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
    model, state, _joint_panel = build_model(args, asset_path)
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
    model, state, _joint_panel = build_model(args, asset_path)
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


def _render_frame(viewer: newton.viewer.ViewerGL, scene: DropScene) -> np.ndarray:
    viewer.begin_frame(scene.sim_time)
    viewer.log_state(scene.state)
    viewer.end_frame()
    return viewer.get_frame(render_ui=False).numpy()


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
    model, state, _joint_panel = build_model(args, asset_path)
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
    total_frames, hold_frames, simulation_frames = recording_frame_counts(
        duration_seconds=duration_seconds,
        initial_hold_seconds=initial_hold_seconds,
        fps=profile.video_fps,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    initial_bounds = compute_asset_bounds(scene.model, scene.state)
    configure_warp_cpu_only()
    viewer, rendering = _headless_viewer(profile)
    viewer.set_model(scene.model)
    viewer.set_camera(wp.vec3(1.0, -1.0, 1.0), pitch=-22.0, yaw=42.0)
    frame_camera_on_bounds(viewer.camera, _camera_bounds(initial_bounds), padding=1.45)
    viewer.show_visual = True
    viewer.show_collision = False

    preview_every = max(1, round(profile.preview_interval_seconds * profile.video_fps))
    video_path = output_directory / "video.mp4"

    try:
        # Compile shaders and upload meshes before opening the output video.
        poster = _render_frame(viewer, scene)
        atomic_write_jpeg(output_directory / "poster.jpg", poster)
        atomic_write_jpeg(
            output_directory.parent.parent / "preview.jpg",
            poster,
            quality=82,
            size=(profile.preview_width, profile.preview_height),
        )
        with H264VideoWriter(
            video_path,
            width=profile.video_width,
            height=profile.video_height,
            fps=profile.video_fps,
        ) as writer:
            for frame_index in range(total_frames):
                if frame_index >= hold_frames:
                    for _ in range(profile.physics_steps_per_video_frame):
                        step_drop_scene(scene, profile)
                frame = _render_frame(viewer, scene)
                writer.write(frame)
                if frame_index % preview_every == 0:
                    atomic_write_jpeg(
                        output_directory.parent.parent / "preview.jpg",
                        frame,
                        quality=82,
                        size=(profile.preview_width, profile.preview_height),
                    )
        final_frame = _render_frame(viewer, scene)
        atomic_write_jpeg(output_directory / "final.jpg", final_frame)
        atomic_write_jpeg(
            output_directory.parent.parent / "preview.jpg",
            final_frame,
            quality=82,
            size=(profile.preview_width, profile.preview_height),
        )
    finally:
        viewer.close()

    final_bounds = compute_asset_bounds(scene.model, scene.state)
    body_q = np.asarray(scene.state.body_q.numpy(), dtype=np.float64)
    body_qd = np.asarray(scene.state.body_qd.numpy(), dtype=np.float64)
    finite = bool(np.isfinite(body_q).all() and np.isfinite(body_qd).all())
    if not finite:
        raise RuntimeError("MuJoCo produced a non-finite body state")
    return {
        "duration_seconds": duration_seconds,
        "initial_hold_seconds": initial_hold_seconds,
        "video_duration_seconds": duration_seconds + initial_hold_seconds,
        "video_frames": total_frames,
        "physics_steps": simulation_frames * profile.physics_steps_per_video_frame,
        "rendering": rendering,
        "initial_bounds": {
            "minimum": initial_bounds.minimum.tolist(),
            "maximum": initial_bounds.maximum.tolist(),
        },
        "final_bounds": {
            "minimum": final_bounds.minimum.tolist(),
            "maximum": final_bounds.maximum.tolist(),
        },
        "finite": finite,
        "files": {
            "video": "video.mp4",
            "poster": "poster.jpg",
            "final": "final.jpg",
        },
    }
