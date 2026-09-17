"""Required recording for the bounded GPU smoke; no CPU or display fallback."""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ..gpu_safety import GpuExecutionPermit, require_gpu_execution_permit, require_scene_on_logical_gpu
from ..profiles import ExperimentProfile, get_profile


class GpuRecordingError(RuntimeError):
    def __init__(self, stage: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        self.stage = stage
        self.diagnostics = diagnostics
        message = f"GPU recording failed during {stage}; no rendering fallback was attempted"
        if diagnostics is not None:
            message += "; EGL diagnostics=" + json.dumps(diagnostics, sort_keys=True)
        super().__init__(message)


class EglDeviceMappingError(RuntimeError):
    """A failed match with bounded, non-secret device-query evidence."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "Cannot uniquely map EGL to the allocated logical CUDA device; "
            + json.dumps(diagnostics, sort_keys=True)
        )


def _egl_device_for_cuda_zero(permit: GpuExecutionPermit) -> tuple[int, int]:
    """Resolve an EGL index from CUDA's device handle, never a host GPU number.

    EGL_NV_device_cuda defines EGL_CUDA_DEVICE_NV as a CUDA API device handle.
    CUDA has already been restricted and verified to expose only logical cuda:0.
    https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt
    """
    require_gpu_execution_permit(permit)
    from pyglet.libs.egl import egl, eglext
    from pyglet.libs.egl.lib import link_EGL

    query_string = link_EGL(
        "eglQueryDeviceStringEXT", ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_int],
    )
    query_attribute = link_EGL(
        "eglQueryDeviceAttribEXT", egl.EGLBoolean,
        [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)],
    )
    count = egl.EGLint()
    count_ok = bool(eglext.eglQueryDevicesEXT(0, None, ctypes.byref(count)))
    evidence: dict[str, Any] = {"egl_device_count": count.value, "devices": []}
    if not count_ok or not 0 < count.value <= 64:
        evidence["reason_code"] = "egl_device_enumeration_unavailable"
        raise EglDeviceMappingError(evidence)
    devices = (eglext.EGLDeviceEXT * count.value)()
    if not eglext.eglQueryDevicesEXT(count.value, devices, ctypes.byref(count)):
        evidence["reason_code"] = "egl_device_enumeration_failed"
        raise EglDeviceMappingError(evidence)
    evidence["egl_device_count"] = count.value
    matches = []
    for index in range(count.value):
        extensions = (query_string(devices[index], 0x3055) or b"").split()  # EGL_EXTENSIONS
        supports_mapping = b"EGL_NV_device_cuda" in extensions
        item = {
            "egl_index": index,
            "supports_cuda_mapping": supports_mapping,
            "software_device": b"EGL_MESA_device_software" in extensions,
            "cuda_attribute_query_ok": None,
            "cuda_device": None,
        }
        evidence["devices"].append(item)
        if not supports_mapping:
            continue
        cuda_device = ctypes.c_ssize_t(-1)
        query_ok = bool(query_attribute(devices[index], 0x323A, ctypes.byref(cuda_device)))
        item["cuda_attribute_query_ok"] = query_ok
        if query_ok:
            item["cuda_device"] = cuda_device.value
        if query_ok and cuda_device.value == 0:
            matches.append((index, devices[index]))
    if len(matches) != 1:
        if len(matches) > 1:
            reason = "ambiguous_logical_cuda_zero"
        elif not any(item["supports_cuda_mapping"] for item in evidence["devices"]):
            reason = "no_cuda_device_extension"
        elif not any(item["cuda_attribute_query_ok"] for item in evidence["devices"]):
            reason = "cuda_attribute_query_failed"
        else:
            reason = "no_logical_cuda_zero"
        evidence["reason_code"] = reason
        raise EglDeviceMappingError(evidence)
    return matches[0]


def preflight_gpu_rendering(permit: GpuExecutionPermit) -> tuple[int, int]:
    """Check the graphics path after CUDA initialization, before model compilation."""
    require_gpu_execution_permit(permit)
    try:
        return _prepare_egl_device(permit)
    except Exception as exc:
        from ..graphics_diagnostics import collect_graphics_loader_evidence

        evidence = dict(getattr(exc, "diagnostics", None) or {
            "reason_code": "egl_preflight_failed", "error_type": type(exc).__name__,
            "message": str(exc)[:2048],
        })
        try:
            evidence["graphics_loader"] = collect_graphics_loader_evidence()
        except Exception as diagnostic_error:
            # Diagnostic collection must never replace the original failure.
            evidence["graphics_loader"] = {"collection_error": type(diagnostic_error).__name__}
        raise GpuRecordingError("renderer_preflight", diagnostics=evidence) from exc


def _prepare_egl_device(permit: GpuExecutionPermit) -> tuple[int, int]:
    if sys.platform != "linux":
        raise RuntimeError("GPU video smoke requires a Linux Determined trial")
    # EGL and PyOpenGL must be configured before either library creates a context.
    if "pyglet.gl" in sys.modules or "OpenGL.GL" in sys.modules:
        raise RuntimeError("GPU video smoke requires a fresh process before OpenGL import")
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE", "").lower() in {"1", "true"}:
        raise RuntimeError("Software OpenGL settings cannot be reused for GPU recording")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyglet

    pyglet.options["headless"] = True
    return _egl_device_for_cuda_zero(permit)


def _open_gpu_viewer(scene: Any, profile: ExperimentProfile) -> tuple[Any, dict[str, Any]]:
    require_gpu_execution_permit(scene.gpu_permit, profile)
    require_scene_on_logical_gpu(scene)
    egl_index, egl_device = _prepare_egl_device(scene.gpu_permit)
    import pyglet

    pyglet.options["headless_device"] = egl_index

    import newton.viewer
    import warp as wp
    from pyglet.libs.egl import egl
    from pyglet.libs.egl.lib import link_EGL
    from .recording import _opengl_identity

    wp.set_device("cuda:0")
    viewer = newton.viewer.ViewerGL(
        width=profile.video_width, height=profile.video_height,
        headless=True, paused=True,
    )
    try:
        renderer, vendor = _opengl_identity()
        if "nvidia" not in vendor.lower():
            raise RuntimeError("GPU video requires an NVIDIA OpenGL renderer")
        query_display = link_EGL(
            "eglQueryDisplayAttribEXT", egl.EGLBoolean,
            [egl.EGLDisplay, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)],
        )
        current_device = ctypes.c_ssize_t()
        if (
            not query_display(egl.eglGetCurrentDisplay(), 0x322C, ctypes.byref(current_device))
            or current_device.value != egl_device
        ):
            raise RuntimeError("OpenGL context is not on the selected EGL device")
        return viewer, {
            "device": "cuda:0", "backend": "egl", "renderer": renderer, "vendor": vendor,
            "egl_cuda_device_matched": True,
            "allocated_gpu_uuid": scene.gpu_permit.allocation.nvidia_visible_device_uuid,
        }
    except BaseException:
        viewer.close()
        raise


def record_gpu_drop_case(
    scene: Any, *, profile: ExperimentProfile, output_directory: Path,
    duration_seconds: float,
    on_physics_started: Callable[[], None] | None = None,
    on_step_started: Callable[[int], None] | None = None,
    on_step_completed: Callable[[int], None] | None = None,
    on_recording_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Render the initial hold and every 20th completed step of the real GPU case."""
    require_gpu_execution_permit(scene.gpu_permit, profile)
    require_scene_on_logical_gpu(scene)
    if profile != get_profile("mujoco-warp-cuda-dt1ms-video-smoke-v1"):
        raise ValueError("GPU recording requires the fixed video smoke profile")
    if duration_seconds != profile.case_duration_seconds or scene.completed_physics_steps:
        raise ValueError("GPU recording permits exactly one fresh fixed-duration case")

    import numpy as np
    import warp as wp
    from asset_viewer.camera import compute_asset_bounds, frame_camera_on_bounds
    from .drop import _camera_bounds, simulate_drop_case_without_recording
    from .recording import _render_frame, recording_frame_counts
    from ..video import H264VideoWriter, atomic_write_jpeg

    hold_seconds = 0.5
    total_frames, hold_frames, _ = recording_frame_counts(
        duration_seconds=duration_seconds, initial_hold_seconds=hold_seconds, fps=profile.video_fps,
    )
    viewer = None
    stage = "renderer_setup"
    started = time.monotonic()

    def progress(value: str) -> None:
        nonlocal stage
        stage = value
        if on_recording_progress is not None:
            on_recording_progress(value)

    def deadline() -> None:
        if time.monotonic() - started > profile.wall_time_limit_seconds:
            raise TimeoutError("GPU video smoke exceeded its recording wall-time limit")

    try:
        progress("renderer_setup")
        viewer, rendering = _open_gpu_viewer(scene, profile)
        viewer.set_model(scene.model)
        viewer.set_camera(wp.vec3(1.0, -1.0, 1.0), pitch=-22.0, yaw=42.0)
        frame_camera_on_bounds(
            viewer.camera, _camera_bounds(compute_asset_bounds(scene.model, scene.state)), padding=1.45,
        )
        viewer.show_visual = True
        viewer.show_collision = False

        def frame() -> Any:
            deadline()
            rgb = _render_frame(viewer, scene)
            if rgb.shape != (profile.video_height, profile.video_width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError("Unexpected GPU frame dimensions or pixel format")
            return rgb

        progress("frame_readback")
        poster = frame()  # Exercise CUDA/GL interop before starting physics.
        output_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_jpeg(output_directory / "poster.jpg", poster)
        preview = output_directory.parent.parent / "preview.jpg"
        atomic_write_jpeg(preview, poster, size=(profile.preview_width, profile.preview_height))
        progress("encoding")
        with H264VideoWriter(
            output_directory / "video.mp4", width=profile.video_width,
            height=profile.video_height, fps=profile.video_fps,
        ) as writer:
            for _ in range(hold_frames):
                deadline()
                writer.write(poster)
            last_frame = poster
            frames_written = hold_frames

            def completed(number: int) -> None:
                nonlocal last_frame, frames_written
                if on_step_completed is not None:
                    on_step_completed(number)
                if number % profile.physics_steps_per_video_frame == 0:
                    progress("frame_readback")
                    last_frame = frame()
                    progress("encoding")
                    writer.write(last_frame)
                    frames_written += 1
                    if number % round(profile.preview_interval_seconds / profile.physics_dt) == 0:
                        atomic_write_jpeg(
                            preview, last_frame, size=(profile.preview_width, profile.preview_height),
                        )
                    progress("simulating")

            progress("simulating")
            result = simulate_drop_case_without_recording(
                scene, profile=profile, duration_seconds=duration_seconds,
                on_physics_started=on_physics_started, on_step_started=on_step_started,
                on_step_completed=completed,
            )
            if frames_written != total_frames:
                raise RuntimeError("GPU recording did not capture the complete frame count")
            progress("encoder_finalize")
        deadline()
        atomic_write_jpeg(output_directory / "final.jpg", last_frame)
        atomic_write_jpeg(preview, last_frame, size=(profile.preview_width, profile.preview_height))
        progress("renderer_close")
        closing_viewer, viewer = viewer, None
        closing_viewer.close()
        deadline()
        files = {"video": "video.mp4", "poster": "poster.jpg", "final": "final.jpg"}
        return {
            **result, "initial_hold_seconds": hold_seconds,
            "video_duration_seconds": duration_seconds + hold_seconds,
            "video_frames": frames_written, "rendering": rendering, "files": files,
            "wall_elapsed_seconds": time.monotonic() - started,
            "recording": {"status": "succeeded", "files": list(files.values())},
        }
    except Exception as exc:
        raise GpuRecordingError(stage, diagnostics=getattr(exc, "diagnostics", None)) from exc
    finally:
        if viewer is not None:
            viewer.close()
