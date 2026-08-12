from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .storage import DataRoot, atomic_write_text


DEFAULT_SERVER_CONFIG = Path.home() / ".config" / "newton-test" / "server.toml"
DEFAULT_CONTROLLER_CONFIG = Path.home() / ".config" / "newton-test" / "controller.toml"


@dataclass(frozen=True)
class ServerConfig:
    data_root: Path
    web_port: int = 8765


@dataclass(frozen=True)
class ControllerConfig:
    """Non-sensitive preferences used by the local interactive controller."""

    viewer_source: Path | None = None
    data_root: Path | None = None
    ssh_alias: str | None = None
    local_port: int = 8765
    remote_port: int = 8765
    open_browser: bool = True


def _controller_port(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid controller config: {field} must be an integer")
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid controller config: {field} must be an integer") from exc
    if not 1024 <= port <= 65535:
        raise ValueError(f"Invalid controller config: {field} must be between 1024 and 65535")
    return port


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid controller config: {field} must be text")
    stripped = value.strip()
    return stripped or None


def _optional_path(value: object, *, field: str) -> Path | None:
    text = _optional_text(value, field=field)
    return Path(text).expanduser().resolve(strict=False) if text is not None else None


def load_controller_config(
    path: Path = DEFAULT_CONTROLLER_CONFIG,
) -> ControllerConfig:
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError:
        return ControllerConfig()
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid controller config: {path}") from exc

    paths = document.get("paths", {})
    remote = document.get("remote", {})
    web = document.get("web", {})
    if not all(isinstance(section, dict) for section in (paths, remote, web)):
        raise ValueError(f"Invalid controller config: {path}")

    open_browser = web.get("open_browser", True)
    if not isinstance(open_browser, bool):
        raise ValueError("Invalid controller config: web.open_browser must be true or false")

    return ControllerConfig(
        viewer_source=_optional_path(paths.get("viewer_source"), field="paths.viewer_source"),
        data_root=_optional_path(paths.get("data_root"), field="paths.data_root"),
        ssh_alias=_optional_text(remote.get("ssh_alias"), field="remote.ssh_alias"),
        local_port=_controller_port(remote.get("local_port", 8765), field="remote.local_port"),
        remote_port=_controller_port(remote.get("remote_port", 8765), field="remote.remote_port"),
        open_browser=open_browser,
    )


def save_controller_config(
    config: ControllerConfig,
    path: Path = DEFAULT_CONTROLLER_CONFIG,
) -> None:
    local_port = _controller_port(config.local_port, field="remote.local_port")
    remote_port = _controller_port(config.remote_port, field="remote.remote_port")
    if not isinstance(config.open_browser, bool):
        raise ValueError("Invalid controller config: web.open_browser must be true or false")

    content = "[paths]\n"
    if config.viewer_source is not None:
        content += (
            "viewer_source = "
            f"{json.dumps(str(config.viewer_source), ensure_ascii=False)}\n"
        )
    if config.data_root is not None:
        content += (
            "data_root = "
            f"{json.dumps(str(config.data_root), ensure_ascii=False)}\n"
        )
    content += "\n[remote]\n"
    if config.ssh_alias is not None:
        content += f"ssh_alias = {json.dumps(config.ssh_alias, ensure_ascii=False)}\n"
    content += f"local_port = {local_port}\nremote_port = {remote_port}\n"
    content += "\n[web]\n"
    content += f"open_browser = {'true' if config.open_browser else 'false'}\n"

    atomic_write_text(path, content, mode=0o600)
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_server_config(path: Path = DEFAULT_SERVER_CONFIG) -> ServerConfig:
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Server storage is not configured; expected config file: {path}"
        ) from exc

    try:
        data_root = Path(document["storage"]["data_root"])
        web_port = int(document.get("web", {}).get("port", 8765))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid server config: {path}") from exc
    if not 1024 <= web_port <= 65535:
        raise ValueError("Configured web.port must be between 1024 and 65535")
    return ServerConfig(data_root=data_root, web_port=web_port)


def save_server_config(
    config: ServerConfig,
    path: Path = DEFAULT_SERVER_CONFIG,
) -> None:
    if not 1024 <= config.web_port <= 65535:
        raise ValueError("web_port must be between 1024 and 65535")
    content = (
        "[storage]\n"
        f"data_root = {json.dumps(str(config.data_root), ensure_ascii=False)}\n\n"
        "[web]\n"
        f"port = {config.web_port}\n"
    )
    atomic_write_text(path, content, mode=0o600)
    if os.name == "posix":
        os.chmod(path, 0o600)


def resolve_data_root(
    explicit: Path | None,
    *,
    config_path: Path = DEFAULT_SERVER_CONFIG,
) -> DataRoot:
    if explicit is not None:
        return DataRoot(explicit)
    environment = os.environ.get("NEWTON_DATA_ROOT")
    if environment:
        return DataRoot(Path(environment))
    return DataRoot(load_server_config(config_path).data_root)
