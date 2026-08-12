from __future__ import annotations

import os


_SOFTWARE_RENDERER_MARKERS = (
    "llvmpipe",
    "softpipe",
    "swrast",
    "software rasterizer",
)


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
        raise RuntimeError(
            "CPU smoke requires a verified software OpenGL renderer; refusing hardware rendering"
        )
    return normalized
