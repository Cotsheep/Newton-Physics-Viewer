"""Bounded loader evidence for a failed GPU rendering preflight.

No driver installation, context creation, GPU selection, or environment changes.
Library loading is attempted only after the original EGL failure is captured.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Mapping


_ENVIRONMENT_KEYS = (
    "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS",
    "LIBGL_ALWAYS_SOFTWARE", "MESA_LOADER_DRIVER_OVERRIDE", "EGL_PLATFORM",
    "PYOPENGL_PLATFORM", "PYGLET_HEADLESS",
)
_LIBRARIES = ("libEGL.so.1", "libEGL_nvidia.so.0", "libGLdispatch.so.0")


def _read_bounded(path: Path, limit: int = 16384) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("diagnostic input exceeds size limit")
    return value


def collect_graphics_loader_evidence(
    *, environ: Mapping[str, str] | None = None,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> dict[str, Any]:
    """Report a whitelist of loader settings, vendor files and dlopen errors.

    Successful dlopen proves dependencies loaded, not that a usable GPU graphics
    context exists. A failed dlopen does not distinguish a missing library from
    its missing dependencies; retain the actual error for that distinction.
    """
    environ = os.environ if environ is None else environ
    report: dict[str, Any] = {
        "environment": {key: environ[key][:2048] for key in _ENVIRONMENT_KEYS if key in environ},
        "library_loads": {}, "vendor_configs": [],
    }
    try:
        kernel = _read_bounded(Path("/proc/driver/nvidia/version"))
        match = re.search(r"Kernel Module\s+([0-9][0-9.]*)", kernel)
        report["nvidia_driver_version"] = match.group(1) if match else "unknown"
    except (OSError, ValueError):
        report["nvidia_driver_version"] = "unknown"

    # Docker can append this key to the OCI environment after the YAML is read.
    # Retain duplicate initial entries without printing other environment values.
    try:
        initial = _read_bounded(Path("/proc/self/environ"), 131072)
        report["initial_driver_capabilities"] = [
            entry.split("=", 1)[1][:256] for entry in initial.split("\0")
            if entry.startswith("NVIDIA_DRIVER_CAPABILITIES=")
        ][:8]
    except (OSError, ValueError):
        report["initial_driver_capabilities"] = None

    # Capture loaded paths before diagnostic dlopen changes the loaded set.
    try:
        maps = _read_bounded(Path("/proc/self/maps"), 2097152)
        report["loaded_graphics_paths_before_probe"] = sorted({
            fields[-1] for line in maps.splitlines()
            if len(fields := line.split(maxsplit=5)) == 6
            and re.search(r"/lib(?:EGL|GLdispatch|nvidia-(?:egl|gl))", fields[-1])
        })[:32]
    except (OSError, ValueError):
        report["loaded_graphics_paths_before_probe"] = None

    if "__EGL_VENDOR_LIBRARY_FILENAMES" in environ:
        report["vendor_selection"] = "explicit_files"
        paths = [Path(value) for value in environ["__EGL_VENDOR_LIBRARY_FILENAMES"].split(":") if value][:16]
    else:
        report["vendor_selection"] = (
            "explicit_directories" if "__EGL_VENDOR_LIBRARY_DIRS" in environ
            else "standard_directories_inspected"
        )
        directories = environ.get(
            "__EGL_VENDOR_LIBRARY_DIRS", "/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d",
        )
        paths = []
        for value in directories.split(":")[:16]:
            if value:
                try:
                    paths.extend(islice(Path(value).glob("*.json"), 16 - len(paths)))
                except OSError:
                    report.setdefault("unreadable_vendor_directories", []).append(value[:2048])
            if len(paths) >= 16:
                break
    for path in paths:
        item: dict[str, Any] = {"path": str(path)[:2048]}
        try:
            data = json.loads(_read_bounded(path))
            library = data["ICD"]["library_path"]
            if not isinstance(library, str):
                raise ValueError("invalid vendor library path")
            item["library_path"] = library[:2048]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            item["error"] = type(exc).__name__
        report["vendor_configs"].append(item)

    for name in _LIBRARIES:
        try:
            loader(name)
        except OSError as exc:
            report["library_loads"][name] = {"loaded": False, "error": str(exc)[:2048]}
        else:
            report["library_loads"][name] = {"loaded": True}
    return report
