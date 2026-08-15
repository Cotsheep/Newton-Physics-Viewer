from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import newton
import newton.viewer
import numpy as np
import warp as wp

from asset_viewer.camera import AssetBounds, frame_camera_on_bounds

from ..cpu_safety import CpuSmokeEnvironmentUnavailable, require_software_opengl_renderer
from ..profiles import ExperimentProfile
from ..video import H264VideoWriter, atomic_write_jpeg


class CpuOnlyViewerGL(newton.viewer.ViewerGL):
    """Avoid Newton's CUDA-pinned VBO staging buffer on a CPU-only render path."""

    def _build_packed_vbo_arrays(self) -> None:
        if self.device.is_cpu:
            self._packed_groups = []
            self._capsule_keys = set()
            self._packed_write_indices = None
            self._packed_world_xforms = None
            self._packed_vbo_xforms = None
            self._packed_vbo_xforms_host = None
            return
        super()._build_packed_vbo_arrays()


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


def _headless_viewer(
    profile: ExperimentProfile,
) -> tuple[newton.viewer.ViewerGL, dict[str, str]]:
    viewer = CpuOnlyViewerGL(
        width=profile.video_width,
        height=profile.video_height,
        headless=True,
        paused=True,
    )
    try:
        renderer, vendor = _opengl_identity()
        software_required = os.environ.get("NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL") == "1"
        if software_required:
            try:
                renderer = require_software_opengl_renderer(renderer)
            except CpuSmokeEnvironmentUnavailable as exc:
                exc.vendor = vendor
                raise
        return viewer, {
            "device": "software-cpu" if software_required else "system-opengl",
            "renderer": renderer,
            "vendor": vendor,
        }
    except Exception:
        viewer.close()
        raise


def _render_frame(viewer: newton.viewer.ViewerGL, scene: Any) -> np.ndarray:
    viewer.begin_frame(scene.sim_time)
    viewer.log_state(scene.state)
    viewer.end_frame()
    return viewer.get_frame(render_ui=False).numpy()


def record_simulation_video(
    scene: Any,
    *,
    profile: ExperimentProfile,
    output_directory: Path,
    duration_seconds: float,
    initial_hold_seconds: float,
    camera_bounds: AssetBounds,
    step_scene: Callable[[Any, ExperimentProfile], None],
    validate_scene: Callable[[Any], None] | None = None,
    camera_pitch: float = -22.0,
    camera_yaw: float = 42.0,
    camera_padding: float = 1.45,
) -> dict[str, Any]:
    """Record the shared hold/step/render/encode lifecycle for one CPU case."""

    total_frames, hold_frames, simulation_frames = recording_frame_counts(
        duration_seconds=duration_seconds,
        initial_hold_seconds=initial_hold_seconds,
        fps=profile.video_fps,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    configure_warp_cpu_only()
    viewer, rendering = _headless_viewer(profile)
    preview_every = max(1, round(profile.preview_interval_seconds * profile.video_fps))
    try:
        viewer.set_model(scene.model)
        viewer.set_camera(
            wp.vec3(1.0, -1.0, 1.0),
            pitch=camera_pitch,
            yaw=camera_yaw,
        )
        frame_camera_on_bounds(viewer.camera, camera_bounds, padding=camera_padding)
        viewer.show_visual = True
        viewer.show_collision = False

        poster = _render_frame(viewer, scene)
        atomic_write_jpeg(output_directory / "poster.jpg", poster)
        atomic_write_jpeg(
            output_directory.parent.parent / "preview.jpg",
            poster,
            quality=82,
            size=(profile.preview_width, profile.preview_height),
        )
        with H264VideoWriter(
            output_directory / "video.mp4",
            width=profile.video_width,
            height=profile.video_height,
            fps=profile.video_fps,
        ) as writer:
            for frame_index in range(total_frames):
                if frame_index >= hold_frames:
                    for _ in range(profile.physics_steps_per_video_frame):
                        step_scene(scene, profile)
                    if validate_scene is not None:
                        validate_scene(scene)
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

    return {
        "duration_seconds": duration_seconds,
        "initial_hold_seconds": initial_hold_seconds,
        "video_duration_seconds": duration_seconds + initial_hold_seconds,
        "video_frames": total_frames,
        "physics_steps": simulation_frames * profile.physics_steps_per_video_frame,
        "rendering": rendering,
        "files": {
            "video": "video.mp4",
            "poster": "poster.jpg",
            "final": "final.jpg",
        },
    }
