from __future__ import annotations

import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit

from .storage import DataRoot, atomic_write_bytes


STATIC_FILES = ("index.html", "app.js", "styles.css")
_RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


def install_static_site(data_root: DataRoot) -> None:
    """Install the versioned, framework-free browser assets into data_root/web."""

    source_root = Path(__file__).with_name("web_static")
    target_root = data_root.location("web")
    target_root.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_FILES:
        atomic_write_bytes(target_root / filename, (source_root / filename).read_bytes())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


class ResultRequestHandler(BaseHTTPRequestHandler):
    server_version = "NewtonResultServer/1"
    protocol_version = "HTTP/1.1"
    data_root: DataRoot

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_POST(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _decode_path(self) -> str | None:
        try:
            raw_path = urlsplit(self.path).path
            decoded = unquote_to_bytes(raw_path).decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            return None
        if "\0" in decoded or "\\" in decoded or not decoded.startswith("/"):
            return None
        segments = decoded.split("/")
        if any(segment in {".", ".."} for segment in segments):
            return None
        return decoded

    def _map_path(self) -> Path | None:
        decoded = self._decode_path()
        if decoded is None:
            return None
        static_name = {
            "/": "index.html",
            "/index.html": "index.html",
            "/app.js": "app.js",
            "/styles.css": "styles.css",
            "/index.json": "index.json",
        }.get(decoded)
        if static_name is not None:
            base = self.data_root.location("web").resolve(strict=True)
            candidate = (base / static_name).resolve(strict=True)
            return candidate if _is_relative_to(candidate, base) else None

        prefix = "/runs/"
        if not decoded.startswith(prefix):
            return None
        relative = decoded[len(prefix) :]
        if not relative or any(segment == "" for segment in relative.split("/")):
            return None
        base = self.data_root.location("runs").resolve(strict=True)
        try:
            candidate = base.joinpath(*relative.split("/")).resolve(strict=True)
        except (FileNotFoundError, OSError):
            return None
        if not _is_relative_to(candidate, base) or not candidate.is_file():
            return None
        return candidate

    def _parse_range(self, size: int) -> tuple[int, int] | None | bool:
        header = self.headers.get("Range")
        if header is None:
            return None
        match = _RANGE_PATTERN.fullmatch(header.strip())
        if match is None:
            return False
        start_text, end_text = match.groups()
        if not start_text and not end_text:
            return False
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            if suffix <= 0:
                return False
            start = max(size - suffix, 0)
            end = size - 1
        if start >= size or start < 0 or end < start:
            return False
        return start, min(end, size - 1)

    def _serve(self, *, send_body: bool) -> None:
        try:
            path = self._map_path()
        except (FileNotFoundError, OSError):
            path = None
        if path is None:
            self.send_error(404, "Result file not found")
            return

        size = path.stat().st_size
        requested_range = self._parse_range(size)
        if requested_range is False:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if requested_range is None:
            status = 200
            start, end = 0, max(size - 1, -1)
        else:
            status = 206
            start, end = requested_range
        length = max(end - start + 1, 0)

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; media-src 'self'; "
            "script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'",
        )
        if path.suffix.lower() == ".json":
            self.send_header("Cache-Control", "no-store")
        elif path.suffix.lower() == ".mp4":
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if not send_body or length == 0:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_result_handler(data_root: DataRoot) -> type[ResultRequestHandler]:
    return type(
        "ConfiguredResultRequestHandler",
        (ResultRequestHandler,),
        {"data_root": data_root},
    )


def create_result_server(
    data_root: DataRoot,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("The result server may bind only to a loopback address")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    data_root.require_initialized()
    install_static_site(data_root)
    return ThreadingHTTPServer((host, port), make_result_handler(data_root))


def serve_results(
    data_root: DataRoot,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    exit_when_stdin_closes: bool = False,
) -> None:
    server = create_result_server(data_root, host=host, port=port)
    if exit_when_stdin_closes:
        def stop_at_stdin_eof() -> None:
            try:
                while sys.stdin.buffer.read(1024):
                    pass
            finally:
                server.shutdown()

        threading.Thread(
            target=stop_at_stdin_eof,
            name="result-server-ssh-lifetime",
            daemon=True,
        ).start()
    try:
        print(f"Result browser: http://{host}:{server.server_port}/", flush=True)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
