from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import BinaryIO

import imageio_ffmpeg
import numpy as np
from PIL import Image

from .storage import atomic_write_bytes


class H264VideoWriter:
    """Stream RGB frames to an atomically-published browser-compatible MP4."""

    def __init__(
        self,
        output: Path,
        *,
        width: int,
        height: int,
        fps: int,
        crf: int = 18,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temporary),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._closed = False

    @property
    def stdin(self) -> BinaryIO:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input stream is unavailable")
        return self.process.stdin

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed video")
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Expected an RGB frame, got shape {rgb.shape}")
        self.stdin.write(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr
            else ""
        )
        if self.process.stderr is not None:
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code:
            if self.temporary.exists():
                self.temporary.unlink()
            raise RuntimeError(f"FFmpeg failed: {stderr.strip()}")
        os.replace(self.temporary, self.output)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.process.kill()
        self.process.wait()
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass  # The killed encoder may have closed its input first.
        if self.process.stderr is not None:
            self.process.stderr.close()
        if self.temporary.exists():
            self.temporary.unlink()

    def __enter__(self) -> H264VideoWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def jpeg_bytes(
    frame: np.ndarray,
    *,
    quality: int = 90,
    size: tuple[int, int] | None = None,
) -> bytes:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected an RGB frame, got shape {rgb.shape}")
    image = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8), mode="RGB")
    if size is not None:
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("JPEG dimensions must be greater than zero")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def atomic_write_jpeg(
    path: Path,
    frame: np.ndarray,
    *,
    quality: int = 90,
    size: tuple[int, int] | None = None,
) -> None:
    atomic_write_bytes(path, jpeg_bytes(frame, quality=quality, size=size))
