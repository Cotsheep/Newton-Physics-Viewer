"""Local adapter for the unmodified official Artiverse asset renderer."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


LOCAL_ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = LOCAL_ROOT / "upstream"
UPSTREAM_VIEWER = UPSTREAM_ROOT / "view_model.py"
BLENDER_CHECK_SCRIPT = LOCAL_ROOT / "check_blender_dependencies.py"
ARTIVERSE_COMMIT = "44f3d41d015018e9b4dff2cbf01fd0892fe6b2c5"
PYGLTFTOOLKIT_COMMIT = "2275d159f8a24b677858489e1257b274e73582ed"


@dataclass(frozen=True)
class ModelInputs:
    model_dir: Path
    model_id: str
    segmented_glb: Path
    articulations_json: Path
    uses_uncorrected_fallback: bool


def resolve_model_inputs(model_dir: Path) -> ModelInputs:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    model_id = model_dir.name
    segmented_glb = model_dir / f"{model_id}.segmented.glb"
    if not segmented_glb.is_file():
        raise FileNotFoundError(f"Official viewer input is missing: {segmented_glb}")

    corrected = model_dir / f"{model_id}.corrected.articulations.json"
    uncorrected = model_dir / f"{model_id}.articulations.json"
    if corrected.is_file():
        articulations_json = corrected
        fallback = False
    elif uncorrected.is_file():
        articulations_json = uncorrected
        fallback = True
    else:
        raise FileNotFoundError(
            "Official viewer needs an articulation file; neither "
            f"{corrected.name} nor {uncorrected.name} exists in {model_dir}"
        )

    return ModelInputs(
        model_dir=model_dir,
        model_id=model_id,
        segmented_glb=segmented_glb,
        articulations_json=articulations_json,
        uses_uncorrected_fallback=fallback,
    )


def viewer_environment() -> dict[str, str]:
    try:
        import static_ffmpeg
    except ImportError:
        pass
    else:
        static_ffmpeg.add_paths(weak=True)

    env = os.environ.copy()
    executable_dir = Path(sys.executable).resolve().parent
    candidates = [executable_dir, executable_dir / "Scripts"]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*(str(path) for path in candidates), existing_path])
    return env


def find_blenderproc(env: dict[str, str] | None = None) -> str | None:
    search_path = (env or viewer_environment()).get("PATH")
    return shutil.which("blenderproc", path=search_path)


def find_ffmpeg(env: dict[str, str] | None = None) -> str | None:
    search_path = (env or viewer_environment()).get("PATH")
    return shutil.which("ffmpeg", path=search_path)


@contextmanager
def staged_official_input(inputs: ModelInputs) -> Iterator[tuple[Path, Path]]:
    """Create the filenames and path style hard-coded by the official launcher."""
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_run_", dir=LOCAL_ROOT) as temp_dir:
        run_root = Path(temp_dir)
        staged_model = run_root / inputs.model_id
        staged_output = run_root / "output"
        staged_model.mkdir()

        shutil.copy2(inputs.segmented_glb, staged_model / f"{inputs.model_id}.segmented.glb")
        shutil.copy2(
            inputs.articulations_json,
            staged_model / f"{inputs.model_id}.corrected.articulations.json",
        )
        yield staged_model, staged_output


def official_command(
    staged_model: Path,
    staged_output: Path,
    *,
    fps: int,
    ambient_strength: float,
    light_size_scale: float,
) -> list[str]:
    return [
        sys.executable,
        str(UPSTREAM_VIEWER),
        "--model_path",
        staged_model.as_posix(),
        "--output_dir",
        staged_output.as_posix(),
        "--fps",
        str(fps),
        "--ambient_strength",
        str(ambient_strength),
        "--light_size_scale",
        str(light_size_scale),
    ]


def check_installation(inputs: ModelInputs) -> None:
    env = viewer_environment()
    if not UPSTREAM_VIEWER.is_file():
        raise FileNotFoundError(f"Official viewer snapshot is missing: {UPSTREAM_VIEWER}")

    completed = subprocess.run(
        [sys.executable, str(UPSTREAM_VIEWER), "--help"],
        cwd=UPSTREAM_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Official viewer import check failed:\n{detail}")

    blenderproc = find_blenderproc(env)
    if blenderproc is None:
        raise RuntimeError(
            "BlenderProc is not installed for this Python environment. "
            "Install upstream/requirements.txt before rendering."
        )

    ffmpeg = find_ffmpeg(env)
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is not available. Install the local requirements so "
            "static-ffmpeg can provide the executable used by the official renderer."
        )

    blender_check = subprocess.run(
        [blenderproc, "run", str(BLENDER_CHECK_SCRIPT)],
        cwd=UPSTREAM_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    marker = "ARTIVERSE_BLENDER_DEPENDENCIES_OK"
    if blender_check.returncode or marker not in blender_check.stdout:
        detail = blender_check.stderr.strip() or blender_check.stdout.strip()
        raise RuntimeError(
            "Official viewer dependencies are missing from Blender's own Python "
            "environment. Run the documented 'blenderproc pip install' command.\n"
            f"{detail}"
        )

    articulation_note = (
        "uncorrected pre-release fallback"
        if inputs.uses_uncorrected_fallback
        else "official corrected annotation"
    )
    print(f"Artiverse official commit: {ARTIVERSE_COMMIT}")
    print(f"pygltftoolkit commit: {PYGLTFTOOLKIT_COMMIT}")
    print(f"Model: {inputs.model_id}")
    print(f"Geometry: {inputs.segmented_glb}")
    print(f"Articulation: {inputs.articulations_json} ({articulation_note})")
    print(f"BlenderProc: {blenderproc}")
    print(f"ffmpeg: {ffmpeg}")
    print("Blender Python dependency check: OK")
    print("Official viewer import check: OK")


def run_official_viewer(args: argparse.Namespace) -> None:
    inputs = resolve_model_inputs(args.model_path)
    check_installation(inputs)
    if args.check:
        return

    output_dir = args.output_dir.expanduser().resolve()
    env = viewer_environment()
    with staged_official_input(inputs) as (staged_model, staged_output):
        command = official_command(
            staged_model,
            staged_output,
            fps=args.fps,
            ambient_strength=args.ambient_strength,
            light_size_scale=args.light_size_scale,
        )
        subprocess.run(command, cwd=UPSTREAM_ROOT, env=env, check=True)

        variant_names = ("videos", "videos_right", "videos_textured", "videos_textured_right")
        rendered_by_variant = {
            variant: {
                path.name
                for path in (staged_output / variant).glob("*.webm")
                if path.is_file()
            }
            for variant in variant_names
        }
        expected_names = rendered_by_variant["videos"]
        incomplete = {
            variant: names
            for variant, names in rendered_by_variant.items()
            if not names or names != expected_names
        }
        if not expected_names or incomplete:
            raise RuntimeError(
                "The official launcher did not produce the same non-empty WEBM set "
                f"for all four variants: {rendered_by_variant}. Review the "
                "BlenderProc output above for the underlying render error."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged_output, output_dir, dirs_exist_ok=True)
        print(f"Copied official viewer output to: {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vendored, unmodified Artiverse official asset renderer."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Artiverse model directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered videos.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--ambient-strength", type=float, default=0.08)
    parser.add_argument("--light-size-scale", type=float, default=1.8)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate input and dependencies without launching Blender renders.",
    )
    return parser.parse_args(argv)


def main() -> None:
    run_official_viewer(parse_args())


if __name__ == "__main__":
    main()
