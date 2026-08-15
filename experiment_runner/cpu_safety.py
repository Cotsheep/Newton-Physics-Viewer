from __future__ import annotations

import os


_SOFTWARE_RENDERER_MARKERS = (
    "llvmpipe",
    "softpipe",
    "swrast",
    "software rasterizer",
)


class CpuSmokeEnvironmentUnavailable(RuntimeError):
    """An explicit host capability gap that may skip opt-in real E2E tests."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        renderer: str | None = None,
        vendor: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.renderer = renderer
        self.vendor = vendor


def prepare_cpu_smoke_environment() -> None:
    """Hide CUDA and require software OpenGL before importing the simulation stack."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if os.name == "posix":
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
        # Mesa loads the swrast DRI driver, whose renderer is normally llvmpipe.
        os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"
        os.environ["NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL"] = "1"


def require_software_opengl_renderer(renderer: str | bytes | None) -> str:
    if isinstance(renderer, bytes):
        normalized = renderer.decode("utf-8", errors="replace")
    elif isinstance(renderer, str):
        normalized = renderer
    else:
        normalized = ""
    if not any(marker in normalized.casefold() for marker in _SOFTWARE_RENDERER_MARKERS):
        raise CpuSmokeEnvironmentUnavailable(
            "software_opengl_not_verified",
            "CPU smoke requires a verified software OpenGL renderer; refusing hardware rendering",
            renderer=normalized or None,
        )
    return normalized
