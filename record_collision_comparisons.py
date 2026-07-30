"""Record deterministic before/after Artiverse collision comparison videos.

The script uses the same Newton model builder and solver configuration as
``view.py``.  Each video contains ten seconds of a normal drop followed by a
reset, a one-second collider inspection hold, and four seconds of collider-only
drop playback.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence
from xml.etree import ElementTree as ET

import imageio_ffmpeg
import newton
import newton.viewer
import numpy as np
import warp as wp

from asset_viewer.app import build_model, configure_warp_cpu_fallback, parse_args as parse_view_args
from asset_viewer.assets import DEFAULT_SOURCE
from asset_viewer.camera import AssetBounds, compute_asset_bounds, frame_camera_on_bounds
from asset_viewer.solvers import create_solver


DEFAULT_DATA_ROOT = DEFAULT_SOURCE
FIX_SUFFIX = " - fix"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_SUBSTEPS = 4
DEFAULT_DROP_SECONDS = 10.0
DEFAULT_COLLISION_SECONDS = 5.0
DEFAULT_SCREEN_SECONDS = 3.0


@dataclass(frozen=True)
class AssetPair:
    source_dir: Path
    fixed_dir: Path
    before_urdf: Path
    after_urdf: Path
    relative_parts: tuple[str, ...]

    @property
    def folder_name(self) -> str:
        return " ".join(self.relative_parts)


@dataclass
class DropScene:
    model: newton.Model
    state: newton.State
    state_next: newton.State
    control: object
    contacts: object
    solver: object
    joint_panel: object
    sim_time: float = 0.0


@dataclass(frozen=True)
class DropResult:
    finite: bool
    initial_min_z: float
    final_min_z: float
    final_max_z: float
    max_speed: float
    stable_on_ground: bool


def _single_urdf(asset_dir: Path) -> Path:
    collider_dir = asset_dir / "urdf_w_collider"
    urdfs = sorted(collider_dir.glob("*.urdf"))
    if len(urdfs) != 1:
        raise ValueError(f"Expected one URDF in {collider_dir}; found {len(urdfs)}")
    return urdfs[0]


def asset_pair(source_dir: Path, data_root: Path = DEFAULT_DATA_ROOT) -> AssetPair:
    source_dir = source_dir.resolve(strict=True)
    fixed_dir = source_dir.with_name(f"{source_dir.name}{FIX_SUFFIX}")
    if not fixed_dir.is_dir():
        raise FileNotFoundError(f"Missing fixed asset directory: {fixed_dir}")
    try:
        relative = source_dir.relative_to(data_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"Asset is not under the Artiverse data root: {source_dir}") from exc
    if len(relative.parts) < 3:
        raise ValueError(f"Expected category/provider/model path under {data_root}: {source_dir}")
    return AssetPair(
        source_dir=source_dir,
        fixed_dir=fixed_dir,
        before_urdf=_single_urdf(source_dir),
        after_urdf=_single_urdf(fixed_dir),
        relative_parts=tuple(relative.parts[:3]),
    )


def discover_pairs(data_root: Path = DEFAULT_DATA_ROOT) -> list[AssetPair]:
    pairs: list[AssetPair] = []
    for fixed_dir in sorted(data_root.rglob(f"*{FIX_SUFFIX}"), key=lambda path: str(path).lower()):
        if not fixed_dir.is_dir():
            continue
        source_dir = fixed_dir.with_name(fixed_dir.name[: -len(FIX_SUFFIX)])
        if source_dir.is_dir():
            pairs.append(asset_pair(source_dir, data_root))
    return pairs


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def urdf_geometry_counts(urdf_path: Path) -> tuple[int, int]:
    root = ET.parse(urdf_path).getroot()
    visuals = sum(1 for element in root.iter() if _local_name(element.tag) == "visual")
    collisions = sum(1 for element in root.iter() if _local_name(element.tag) == "collision")
    return visuals, collisions


def _viewer_args(urdf_path: Path, *, z: float, fps: int, substeps: int) -> argparse.Namespace:
    return parse_view_args(
        [
            str(urdf_path),
            "--simulate",
            "--start-running",
            "--z",
            str(z),
            "--fps",
            str(fps),
            "--substeps",
            str(substeps),
            "--solver",
            "mujoco",
            "--iterations",
            "10",
            "--no-self-collisions",
            "--no-collapse-fixed-joints",
            "--floating",
            "--no-print-traction-force",
            "--no-joint-ui",
        ]
    )


def calibrated_z(after_urdf: Path, *, fps: int, substeps: int) -> tuple[float, AssetBounds]:
    args = _viewer_args(after_urdf, z=0.0, fps=fps, substeps=substeps)
    model, state, _joint_panel = build_model(args, after_urdf)
    bounds = compute_asset_bounds(model, state)
    extent = max(bounds.max_extent, 1.0e-3)
    clearance = float(np.clip(0.75 * extent, 0.15, 1.0))
    return clearance - float(bounds.minimum[2]), bounds


def build_drop_scene(urdf_path: Path, *, z: float, fps: int, substeps: int) -> DropScene:
    args = _viewer_args(urdf_path, z=z, fps=fps, substeps=substeps)
    model, state, joint_panel = build_model(args, urdf_path)
    solver = create_solver(model, args.solver, args.iterations)
    return DropScene(
        model=model,
        state=state,
        state_next=model.state(),
        control=model.control(),
        contacts=model.contacts(),
        solver=solver,
        joint_panel=joint_panel,
    )


def step_scene(scene: DropScene, *, frame_dt: float, substeps: int) -> None:
    sim_dt = frame_dt / substeps
    for _ in range(substeps):
        scene.state.clear_forces()
        scene.model.collide(scene.state, scene.contacts)
        scene.solver.step(
            scene.state,
            scene.state_next,
            scene.control,
            scene.contacts,
            sim_dt,
        )
        scene.state, scene.state_next = scene.state_next, scene.state
    scene.sim_time += frame_dt


def reset_scene(scene: DropScene) -> None:
    scene.joint_panel.set_state(scene.state)
    scene.joint_panel.reset()
    scene.solver.reset(scene.state)
    scene.state_next = scene.model.state()
    scene.sim_time = 0.0


def simulate_drop(
    urdf_path: Path,
    *,
    z: float,
    fps: int = DEFAULT_FPS,
    substeps: int = DEFAULT_SUBSTEPS,
    seconds: float = DEFAULT_SCREEN_SECONDS,
) -> DropResult:
    scene = build_drop_scene(urdf_path, z=z, fps=fps, substeps=substeps)
    initial = compute_asset_bounds(scene.model, scene.state)
    for _ in range(round(seconds * fps)):
        step_scene(scene, frame_dt=1.0 / fps, substeps=substeps)

    body_q = np.asarray(scene.state.body_q.numpy(), dtype=np.float64)
    body_qd = np.asarray(scene.state.body_qd.numpy(), dtype=np.float64)
    finite = bool(np.isfinite(body_q).all() and np.isfinite(body_qd).all())
    if finite:
        final = compute_asset_bounds(scene.model, scene.state)
        max_speed = float(np.linalg.norm(body_qd[:, :3], axis=1).max(initial=0.0))
        object_height = max(float(initial.extents[2]), 1.0e-3)
        lower_tolerance = max(0.02, 0.15 * object_height)
        upper_tolerance = max(0.08, 0.30 * object_height)
        stable = (
            -lower_tolerance <= float(final.minimum[2]) <= upper_tolerance
            and max_speed <= 1.0
        )
        return DropResult(
            finite=True,
            initial_min_z=float(initial.minimum[2]),
            final_min_z=float(final.minimum[2]),
            final_max_z=float(final.maximum[2]),
            max_speed=max_speed,
            stable_on_ground=stable,
        )

    return DropResult(
        finite=False,
        initial_min_z=float(initial.minimum[2]),
        final_min_z=math.nan,
        final_max_z=math.nan,
        max_speed=math.inf,
        stable_on_ground=False,
    )


@contextlib.contextmanager
def suppress_process_output():
    """Temporarily suppress Python and native-library stdout/stderr noise."""
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        stdout_copy = os.dup(1)
        stderr_copy = os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_copy, 1)
            os.dup2(stderr_copy, 2)
            os.close(stdout_copy)
            os.close(stderr_copy)


def screen_candidates(
    data_root: Path,
    *,
    pairs: Sequence[AssetPair] | None,
    limit: int,
    fps: int,
    substeps: int,
    seconds: float,
    counts_only: bool,
) -> int:
    rows: list[tuple[float, AssetPair, int, int]] = []
    for pair in pairs if pairs is not None else discover_pairs(data_root):
        _before_visuals, before_collisions = urdf_geometry_counts(pair.before_urdf)
        _after_visuals, after_collisions = urdf_geometry_counts(pair.after_urdf)
        reduction = before_collisions / max(after_collisions, 1)
        rows.append((reduction, pair, before_collisions, after_collisions))

    rows.sort(key=lambda row: (-row[0], row[1].folder_name.lower()))
    if counts_only:
        print("asset\tbefore_shapes\tafter_shapes\treduction")
        for reduction, pair, before_collisions, after_collisions in rows[:limit]:
            print(
                f"{pair.folder_name}\t{before_collisions}\t{after_collisions}\t"
                f"{reduction:.3f}"
            )
        return 0

    print("asset\tbefore_shapes\tafter_shapes\tbefore_min_z\tafter_min_z\tafter_speed\tafter_stable")
    failures = 0
    for _reduction, pair, before_collisions, after_collisions in rows[:limit]:
        try:
            with suppress_process_output():
                z, _bounds = calibrated_z(pair.after_urdf, fps=fps, substeps=substeps)
                before = simulate_drop(
                    pair.before_urdf,
                    z=z,
                    fps=fps,
                    substeps=substeps,
                    seconds=seconds,
                )
                after = simulate_drop(
                    pair.after_urdf,
                    z=z,
                    fps=fps,
                    substeps=substeps,
                    seconds=seconds,
                )
        except Exception as exc:
            failures += 1
            print(f"{pair.folder_name}\t{before_collisions}\t{after_collisions}\tERROR\t{exc}")
            continue
        print(
            f"{pair.folder_name}\t{before_collisions}\t{after_collisions}\t"
            f"{before.final_min_z:.6g}\t{after.final_min_z:.6g}\t"
            f"{after.max_speed:.6g}\t{after.stable_on_ground}"
        )
    return failures


def _camera_bounds(initial: AssetBounds) -> AssetBounds:
    minimum = initial.minimum.copy()
    maximum = initial.maximum.copy()
    minimum[2] = min(float(minimum[2]), 0.0)
    horizontal_padding = max(0.15 * initial.max_extent, 0.02)
    minimum[:2] -= horizontal_padding
    maximum[:2] += horizontal_padding
    return AssetBounds(minimum, maximum)


class RawVideoWriter:
    def __init__(self, output: Path, *, width: int, height: int, fps: int) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
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
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.output = output
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )

    @property
    def stdin(self) -> BinaryIO:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input stream is not available")
        return self.process.stdin

    def write(self, frame: np.ndarray) -> None:
        self.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg failed for {self.output}: {stderr.strip()}")

    def __enter__(self) -> RawVideoWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self.process.kill()
            self.process.wait()
            return
        self.close()


def _render_frame(viewer: newton.viewer.ViewerGL, scene: DropScene) -> np.ndarray:
    viewer.begin_frame(scene.sim_time)
    viewer.log_state(scene.state)
    viewer.end_frame()
    return viewer.get_frame(render_ui=False).numpy()


def record_video(
    urdf_path: Path,
    output: Path,
    *,
    z: float,
    width: int,
    height: int,
    fps: int,
    substeps: int,
    drop_seconds: float,
    collision_seconds: float,
    framing_bounds: AssetBounds | None = None,
) -> None:
    scene = build_drop_scene(urdf_path, z=z, fps=fps, substeps=substeps)
    initial_bounds = compute_asset_bounds(scene.model, scene.state)
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True, paused=True)
    viewer.set_model(scene.model)
    viewer.set_camera(wp.vec3(1.0, -1.0, 1.0), pitch=-25.0, yaw=40.0)
    frame_camera_on_bounds(
        viewer.camera,
        framing_bounds if framing_bounds is not None else _camera_bounds(initial_bounds),
        padding=1.4,
    )
    viewer.show_visual = True
    viewer.show_collision = False
    frame_dt = 1.0 / fps

    try:
        # Warm up mesh uploads and shader compilation outside the encoded video.
        _render_frame(viewer, scene)
        with RawVideoWriter(output, width=width, height=height, fps=fps) as writer:
            for _ in range(round(drop_seconds * fps)):
                step_scene(scene, frame_dt=frame_dt, substeps=substeps)
                writer.write(_render_frame(viewer, scene))

            reset_scene(scene)
            viewer.show_collision = True
            viewer.show_visual = False
            collision_frames = round(collision_seconds * fps)
            hold_frames = min(round(fps), collision_frames)
            for _ in range(hold_frames):
                writer.write(_render_frame(viewer, scene))
            for _ in range(collision_frames - hold_frames):
                step_scene(scene, frame_dt=frame_dt, substeps=substeps)
                writer.write(_render_frame(viewer, scene))
    finally:
        viewer.close()


def _record_in_fresh_process(
    urdf_path: Path,
    output: Path,
    *,
    z: float,
    width: int,
    height: int,
    fps: int,
    substeps: int,
    drop_seconds: float,
    collision_seconds: float,
    framing_bounds: AssetBounds,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_record-one-urdf",
        str(urdf_path),
        "--_record-one-output",
        str(output),
        "--_record-one-z",
        str(z),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--substeps",
        str(substeps),
        "--drop-seconds",
        str(drop_seconds),
        "--collision-seconds",
        str(collision_seconds),
        "--_framing-bounds",
        *(str(value) for value in np.concatenate((framing_bounds.minimum, framing_bounds.maximum))),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    if result.returncode:
        detail = result.stderr.strip()
        raise RuntimeError(f"Recording subprocess failed for {urdf_path}: {detail}")


def record_pairs(
    pairs: Iterable[AssetPair],
    output_root: Path,
    *,
    width: int,
    height: int,
    fps: int,
    substeps: int,
    drop_seconds: float,
    collision_seconds: float,
) -> None:
    configure_warp_cpu_fallback()
    output_root.mkdir(parents=True, exist_ok=False)
    for pair in pairs:
        asset_output = output_root / pair.folder_name
        with suppress_process_output():
            z, base_bounds = calibrated_z(pair.after_urdf, fps=fps, substeps=substeps)
        translation = np.array((0.0, 0.0, z), dtype=np.float64)
        framing_bounds = _camera_bounds(
            AssetBounds(
                base_bounds.minimum + translation,
                base_bounds.maximum + translation,
            )
        )
        print(f"RECORD {pair.folder_name} before", flush=True)
        _record_in_fresh_process(
            pair.before_urdf,
            asset_output / "before.mp4",
            z=z,
            width=width,
            height=height,
            fps=fps,
            substeps=substeps,
            drop_seconds=drop_seconds,
            collision_seconds=collision_seconds,
            framing_bounds=framing_bounds,
        )
        print(f"RECORD {pair.folder_name} after", flush=True)
        _record_in_fresh_process(
            pair.after_urdf,
            asset_output / "after.mp4",
            z=z,
            width=width,
            height=height,
            fps=fps,
            substeps=substeps,
            drop_seconds=drop_seconds,
            collision_seconds=collision_seconds,
            framing_bounds=framing_bounds,
        )


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--screen", action="store_true", help="Rank and simulate candidate fixed asset pairs.")
    parser.add_argument("--screen-limit", type=int, default=12)
    parser.add_argument("--screen-seconds", type=_positive, default=DEFAULT_SCREEN_SECONDS)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="With --screen, rank collision-count changes without running physics.",
    )
    parser.add_argument(
        "--asset-dir",
        action="append",
        type=Path,
        default=[],
        help="Source asset directory to record; repeat for each asset.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
    parser.add_argument("--drop-seconds", type=_positive, default=DEFAULT_DROP_SECONDS)
    parser.add_argument("--collision-seconds", type=_positive, default=DEFAULT_COLLISION_SECONDS)
    parser.add_argument("--_record-one-urdf", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_record-one-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_record-one-z", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--_framing-bounds", type=float, nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.screen_limit < 1:
        parser.error("--screen-limit must be at least 1")
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even numbers")
    if args.fps < 1 or args.substeps < 1:
        parser.error("--fps and --substeps must be at least 1")
    if args._record_one_urdf is not None:
        if args._record_one_output is None or args._record_one_z is None:
            parser.error("internal single-record mode requires output and z")
        if not math.isfinite(args._record_one_z):
            parser.error("internal single-record z must be finite")
        if args._framing_bounds is None or not all(math.isfinite(value) for value in args._framing_bounds):
            parser.error("internal single-record mode requires finite framing bounds")
    elif not args.screen and not args.asset_dir:
        parser.error("provide --screen or at least one --asset-dir")
    if args.asset_dir and args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = Path.cwd() / f"Artiverse碰撞体对比视频_{timestamp}"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_warp_cpu_fallback()
    if args._record_one_urdf is not None:
        framing_values = np.asarray(args._framing_bounds, dtype=np.float64)
        with suppress_process_output():
            record_video(
                args._record_one_urdf,
                args._record_one_output,
                z=args._record_one_z,
                width=args.width,
                height=args.height,
                fps=args.fps,
                substeps=args.substeps,
                drop_seconds=args.drop_seconds,
                collision_seconds=args.collision_seconds,
                framing_bounds=AssetBounds(framing_values[:3], framing_values[3:]),
            )
        return 0

    if args.screen:
        selected_pairs = (
            [asset_pair(path, args.data_root) for path in args.asset_dir]
            if args.asset_dir
            else None
        )
        return min(
            screen_candidates(
                args.data_root,
                pairs=selected_pairs,
                limit=args.screen_limit,
                fps=args.fps,
                substeps=args.substeps,
                seconds=args.screen_seconds,
                counts_only=args.counts_only,
            ),
            1,
        )

    pairs = [asset_pair(path, args.data_root) for path in args.asset_dir]
    record_pairs(
        pairs,
        args.output_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        substeps=args.substeps,
        drop_seconds=args.drop_seconds,
        collision_seconds=args.collision_seconds,
    )
    print(f"DONE {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
