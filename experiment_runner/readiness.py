"""Read-only deployment checks. Never import a simulation or CUDA runtime."""
from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from .config import DEFAULT_SERVER_CONFIG, resolve_data_root
from .paths import source_root
from .release import source_state
from .storage import LAYOUT


def locked_runtime_versions(root: Path | None = None) -> dict[str, str]:
    root = root or source_root()
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (root / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    packages = {p["name"]: p for p in lock["package"]}
    pending = [item.split("==")[0] for item in project["project"]["dependencies"]]
    versions = {}
    while pending:
        name = pending.pop()
        if name in versions:
            continue
        package = packages[name]
        versions[name] = package["version"]
        pending.extend(item["name"] for item in package.get("dependencies", []))
    return versions


def runtime_version_check(root: Path | None = None) -> dict:
    expected = locked_runtime_versions(root)
    actual = {}
    for name in expected:
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual[name] = "not-installed"
    return {"expected": expected, "actual": actual, "matches": actual == expected}


def require_locked_runtime() -> dict:
    report = runtime_version_check()
    if sys.version_info[:2] != (3, 12) or not report["matches"]:
        raise ValueError("GPU execution requires Python 3.12 and the exact locked runtime dependencies")
    return report


def check_readiness(
    data_root: Path | None = None, *, config_path: Path = DEFAULT_SERVER_CONFIG,
    port: int = 8765,
) -> dict:
    checks = []

    def add(key, label, ok, detail, *, results=False):
        checks.append({"id": key, "label": label, "status": "ready" if ok else "blocked",
                       "required_for_results": results, "detail": detail})

    add("python", "Python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0], results=True)
    add("package", "项目包", importlib.util.find_spec("experiment_runner.cli") is not None,
        "experiment_runner.cli", results=True)
    command = shutil.which("newton-test-remote")
    add("command", "服务器命令", command is not None, "available" if command else "not found", results=True)
    try:
        versions = runtime_version_check()
        add("dependencies", "锁定依赖版本", versions["matches"], versions)
    except (OSError, KeyError, ValueError) as exc:
        add("dependencies", "锁定依赖版本", False, str(exc))
    state = source_state()
    if state["git_commit"] is None:
        try:
            import json
            from .release import MANIFEST, require_gpu_source
            manifest = json.loads((source_root() / MANIFEST).read_text(encoding="utf-8"))
            state = require_gpu_source(manifest["git_commit"])
        except (OSError, KeyError, ValueError, TypeError):
            pass
    add("source", "Git 与工作树", state["git_commit"] is not None
        and state["worktree_dirty"] is False and state["source_sha256"] is not None, state)
    try:
        root = resolve_data_root(data_root, config_path=config_path)
        root.require_existing_owned_directory()
        root.require_initialized()
        add("storage", "数据目录配置、所有权与隔离", True, str(root.path), results=True)
        writable = all(os.access(root.location(key), os.W_OK | os.X_OK) for key in LAYOUT)
        add("writable", "固定目录完整且可写", writable, list(LAYOUT), results=True)
    except (OSError, ValueError) as exc:
        add("storage", "数据目录配置、所有权与隔离", False, str(exc), results=True)
        add("writable", "固定目录完整且可写", False, "数据目录未就绪", results=True)
        add("disk", "磁盘容量与余量", False, "数据目录未就绪")
    else:
        try:
            usage = shutil.disk_usage(root.path)
            add("disk", "磁盘容量与余量", usage.free >= 1024**3,
                {"total_bytes": usage.total, "free_bytes": usage.free, "smoke_min_free_bytes": 1024**3})
        except OSError as exc:
            add("disk", "磁盘容量与余量", False, str(exc))
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        if executable == "ffmpeg" and path is None:
            try:
                import imageio_ffmpeg
                path = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError, OSError):
                pass
        try:
            if path is None:
                raise ValueError(f"{executable} not found")
            args = [path, "-hide_banner", "-encoders"] if executable == "ffmpeg" else [path, "-version"]
            result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", check=True, timeout=10)
            ok = executable == "ffprobe" or "libx264" in result.stdout
            add(executable, "FFmpeg / H.264" if executable == "ffmpeg" else "ffprobe", ok,
                "available" if ok else "libx264 unavailable")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            add(executable, executable, False, str(exc))
    try:
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("Result port must be between 1024 and 65535")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
        add("loopback", "回环结果服务端口", True, f"127.0.0.1:{port}", results=True)
    except (OSError, ValueError) as exc:
        add("loopback", "回环结果服务端口", False, str(exc), results=True)
    add("gpu_policy", "GPU 执行边界", False,
        "未探测 GPU；仅在人工提交的 Determined 单 GPU trial 内运行 integration smoke；正式批次关闭")
    return {"schema_version": 1, "checks": checks,
            "results_ready": all(c["status"] == "ready" for c in checks if c["required_for_results"]),
            "gpu_probed": False, "formal_gpu_enabled": False}
