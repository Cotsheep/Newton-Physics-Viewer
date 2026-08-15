from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "smoke_asset" / "newton-mujoco.usda"
WORKER = PROJECT_ROOT / "tests" / "smoke_e2e_worker.py"
PREFLIGHT = PROJECT_ROOT / "tests" / "smoke_e2e_preflight.py"
OPT_IN_ENVIRONMENT = "NEWTON_TEST_RUN_CPU_SMOKE_E2E"
FFPROBE_ENVIRONMENT = "NEWTON_TEST_FFPROBE"
TEMP_ROOT_ENVIRONMENT = "NEWTON_TEST_E2E_TEMP_ROOT"
PREFLIGHT_UNAVAILABLE_EXIT = 77
PREFLIGHT_UNAVAILABLE_REASONS = {"software_opengl_not_verified"}
EXPECTED_ARTIFACTS = {
    "asset-cover.jpg",
    "preview.jpg",
    "run.log",
    "manifest.json",
    "status.json",
    "checksums.sha256",
}


def _preflight_report(stdout: str) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError("software OpenGL preflight emitted invalid JSON") from exc
        if not isinstance(value, dict):
            raise AssertionError("software OpenGL preflight report must be a JSON object")
        reports.append(value)
    if len(reports) != 1:
        raise AssertionError("software OpenGL preflight must emit exactly one structured report")
    report = reports[0]
    for field in ("status", "reason_code", "renderer", "vendor"):
        if field not in report:
            raise AssertionError(f"software OpenGL preflight report is missing {field}")
    return report


def _interpret_preflight(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Accept success, skip only an explicit environment gap, and fail everything else."""

    diagnostics = f"{completed.stdout}\n{completed.stderr}"
    if "traceback (most recent call last):" in diagnostics.casefold():
        raise AssertionError("software OpenGL preflight emitted a traceback")

    try:
        report = _preflight_report(completed.stdout)
    except AssertionError as exc:
        detail = (completed.stderr or completed.stdout).strip()
        raise AssertionError(f"{exc}: {detail or 'no diagnostic output'}") from exc

    if completed.returncode == PREFLIGHT_UNAVAILABLE_EXIT:
        reason_code = report.get("reason_code")
        if (
            report.get("status") != "unavailable"
            or reason_code not in PREFLIGHT_UNAVAILABLE_REASONS
        ):
            raise AssertionError(
                "software OpenGL preflight used the unavailable exit code without "
                "a recognized structured environment reason"
            )
        renderer = report.get("renderer") or "未记录"
        vendor = report.get("vendor") or "未记录"
        raise unittest.SkipTest(
            f"software OpenGL unavailable [{reason_code}]: renderer={renderer}, vendor={vendor}"
        )

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise AssertionError(
            f"software OpenGL preflight failed with exit code {completed.returncode}: "
            f"{detail or 'no diagnostic output'}"
        )
    if report.get("status") != "available" or report.get("reason_code") is not None:
        raise AssertionError("successful software OpenGL preflight reported an invalid status")
    if not isinstance(report.get("renderer"), str) or not report["renderer"].strip():
        raise AssertionError("successful software OpenGL preflight did not report a renderer")
    if not isinstance(report.get("vendor"), str) or not report["vendor"].strip():
        raise AssertionError("successful software OpenGL preflight did not report a vendor")
    return report


def _require_ffmpeg_libx264(
    *,
    importer=importlib.import_module,
    runner=subprocess.run,
) -> str:
    """Return the bundled FFmpeg path after checking its encoder list."""

    try:
        imageio_ffmpeg = importer("imageio_ffmpeg")
    except ImportError as exc:
        raise AssertionError("FFmpeg preflight could not import imageio_ffmpeg") from exc
    except OSError as exc:
        raise AssertionError(f"FFmpeg dependency import failed with an OS error: {exc}") from exc

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError) as exc:
        raise AssertionError(
            f"FFmpeg preflight could not resolve the bundled executable: {exc}"
        ) from exc

    try:
        encoders = runner(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("FFmpeg encoder query timed out after 30 seconds") from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"FFmpeg encoder query failed with exit code {exc.returncode}"
        ) from exc
    except OSError as exc:
        raise AssertionError(f"FFmpeg encoder query could not start: {exc}") from exc
    except subprocess.SubprocessError as exc:
        raise AssertionError(f"FFmpeg encoder query failed: {exc}") from exc
    if "libx264" not in encoders.stdout:
        raise unittest.SkipTest("FFmpeg does not provide the required libx264 encoder")
    return ffmpeg


class CpuSmokePreflightProtocolTests(unittest.TestCase):
    def _completed(
        self,
        returncode: int,
        stdout: str,
        stderr: str = "",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["python", "smoke_e2e_preflight.py"],
            returncode,
            stdout,
            stderr,
        )

    def test_available_report_continues(self) -> None:
        report = _interpret_preflight(
            self._completed(
                0,
                json.dumps(
                    {
                        "status": "available",
                        "reason_code": None,
                        "renderer": "llvmpipe",
                        "vendor": "Mesa",
                    }
                ),
            )
        )
        self.assertEqual(report["renderer"], "llvmpipe")

    def test_explicit_structured_environment_gap_skips(self) -> None:
        with self.assertRaisesRegex(unittest.SkipTest, "software_opengl_not_verified"):
            _interpret_preflight(
                self._completed(
                    PREFLIGHT_UNAVAILABLE_EXIT,
                    json.dumps(
                        {
                            "status": "unavailable",
                            "reason_code": "software_opengl_not_verified",
                            "renderer": "NVIDIA GeForce",
                            "vendor": "NVIDIA",
                        }
                    ),
                )
            )

    def test_import_error_is_a_failure_not_a_skip(self) -> None:
        with self.assertRaises(AssertionError):
            _interpret_preflight(
                self._completed(1, "", "ImportError: cannot import name '_headless_viewer'")
            )

    def test_invalid_json_is_a_failure_not_a_skip(self) -> None:
        with self.assertRaises(AssertionError):
            _interpret_preflight(self._completed(0, "{not-json}"))

    def test_unknown_nonzero_exit_is_a_failure_not_a_skip(self) -> None:
        with self.assertRaises(AssertionError):
            _interpret_preflight(
                self._completed(
                    2,
                    json.dumps(
                        {
                            "status": "unavailable",
                            "reason_code": "software_opengl_not_verified",
                            "renderer": None,
                            "vendor": None,
                        }
                    ),
                    "unknown failure",
                )
            )

    def test_traceback_with_structured_unavailable_report_is_a_failure(self) -> None:
        try:
            _interpret_preflight(
                self._completed(
                    PREFLIGHT_UNAVAILABLE_EXIT,
                    json.dumps(
                        {
                            "status": "unavailable",
                            "reason_code": "software_opengl_not_verified",
                            "renderer": None,
                            "vendor": None,
                        }
                    ),
                    "Traceback (most recent call last):\nRuntimeError: viewer regression",
                )
            )
        except unittest.SkipTest as exc:
            self.fail(f"traceback was incorrectly converted to skip: {exc}")
        except AssertionError as exc:
            self.assertIn("traceback", str(exc).casefold())
        else:
            self.fail("traceback was incorrectly accepted")


class FfmpegEncoderPreflightTests(unittest.TestCase):
    @staticmethod
    def _module() -> mock.Mock:
        module = mock.Mock()
        module.get_ffmpeg_exe.return_value = "ffmpeg"
        return module

    def _assert_failure_not_skip(
        self,
        *,
        importer: mock.Mock,
        runner: mock.Mock,
        diagnostic: str,
    ) -> None:
        try:
            _require_ffmpeg_libx264(importer=importer, runner=runner)
        except unittest.SkipTest as exc:
            self.fail(f"FFmpeg failure was incorrectly converted to skip: {exc}")
        except AssertionError as exc:
            self.assertIn(diagnostic, str(exc).casefold())
        else:
            self.fail("FFmpeg failure was incorrectly accepted")

    def test_imageio_ffmpeg_import_error_is_a_failure_not_a_skip(self) -> None:
        self._assert_failure_not_skip(
            importer=mock.Mock(side_effect=ImportError("broken locked dependency")),
            runner=mock.Mock(),
            diagnostic="import",
        )

    def test_ffmpeg_start_oserror_is_a_failure_not_a_skip(self) -> None:
        self._assert_failure_not_skip(
            importer=mock.Mock(return_value=self._module()),
            runner=mock.Mock(side_effect=OSError("cannot start executable")),
            diagnostic="start",
        )

    def test_ffmpeg_timeout_is_a_failure_not_a_skip(self) -> None:
        self._assert_failure_not_skip(
            importer=mock.Mock(return_value=self._module()),
            runner=mock.Mock(side_effect=subprocess.TimeoutExpired(["ffmpeg"], 30)),
            diagnostic="timed out",
        )

    def test_ffmpeg_nonzero_exit_is_a_failure_not_a_skip(self) -> None:
        self._assert_failure_not_skip(
            importer=mock.Mock(return_value=self._module()),
            runner=mock.Mock(side_effect=subprocess.CalledProcessError(3, ["ffmpeg"])),
            diagnostic="exit code 3",
        )

    def test_successful_query_without_libx264_is_an_explicit_skip(self) -> None:
        completed = subprocess.CompletedProcess(["ffmpeg"], 0, " h264 ", "")
        try:
            _require_ffmpeg_libx264(
                importer=mock.Mock(return_value=self._module()),
                runner=mock.Mock(return_value=completed),
            )
        except unittest.SkipTest as exc:
            self.assertIn("libx264", str(exc))
        else:
            self.fail("missing libx264 did not skip the real E2E")

    def test_successful_query_with_libx264_continues(self) -> None:
        completed = subprocess.CompletedProcess(["ffmpeg"], 0, " V..... libx264 ", "")
        ffmpeg = _require_ffmpeg_libx264(
            importer=mock.Mock(return_value=self._module()),
            runner=mock.Mock(return_value=completed),
        )
        self.assertEqual(ffmpeg, "ffmpeg")


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return False


def _parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    return entries


def _ffprobe_executable() -> str | None:
    explicit = os.environ.get(FFPROBE_ENVIRONMENT)
    if explicit:
        path = Path(explicit)
        return str(path) if path.is_file() else None
    return shutil.which("ffprobe")


def _ffprobe_path_argument(executable: str, path: Path) -> str:
    if os.name != "nt" and executable.casefold().endswith(".exe"):
        converted = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=10,
        )
        return converted.stdout.strip()
    return str(path)


def _probe_video(path: Path) -> dict[str, Any]:
    executable = _ffprobe_executable()
    if executable is None:
        raise RuntimeError("ffprobe became unavailable after E2E preflight")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            _ffprobe_path_argument(executable, path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


@unittest.skipUnless(
    os.environ.get(OPT_IN_ENVIRONMENT) == "1",
    f"set {OPT_IN_ENVIRONMENT}=1 to run real CPU smoke E2E tests",
)
class RealCpuSmokeE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.name == "nt":
            raise unittest.SkipTest(
                "Windows cannot currently prove that its OpenGL renderer is software-only"
            )
        if _ffprobe_executable() is None:
            raise unittest.SkipTest(
                f"ffprobe is required; install it or set {FFPROBE_ENVIRONMENT}"
            )
        _require_ffmpeg_libx264()
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "LIBGL_ALWAYS_SOFTWARE": "true",
                "MESA_LOADER_DRIVER_OVERRIDE": "swrast",
                "NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL": "1",
            }
        )
        try:
            preflight = subprocess.run(
                [sys.executable, str(PREFLIGHT)],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AssertionError(f"software OpenGL preflight could not run: {exc}") from exc
        report = _interpret_preflight(preflight)
        renderer = report["renderer"].casefold()
        if not any(marker in renderer for marker in ("llvmpipe", "softpipe", "swrast")):
            raise AssertionError(f"software OpenGL preflight accepted an unsafe renderer: {renderer}")

    def _run_and_validate(self, template: str) -> tuple[dict[str, Any], dict[str, Any]]:
        temp_root_value = os.environ.get(TEMP_ROOT_ENVIRONMENT)
        temp_root = Path(temp_root_value) if temp_root_value else None
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"newton-test-{template}-",
            dir=temp_root,
        ) as temporary:
            root = Path(temporary)
            data_root = root / "data"
            summary_path = root / "summary.json"
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "-1",
                    "LIBGL_ALWAYS_SOFTWARE": "true",
                    "MESA_LOADER_DRIVER_OVERRIDE": "swrast",
                    "NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL": "1",
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    template,
                    str(data_root),
                    str(FIXTURE),
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=300,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result = summary["result"]
            self.assertEqual(summary["acceptance_status"], "accepted")
            self.assertEqual(result["status"], "succeeded")
            self.assertFalse(result["authoritative"])

            run_directory = data_root / "runtime" / "runs" / result["run_id"]
            manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            case_directories = list((run_directory / "cases").iterdir())
            self.assertEqual(len(case_directories), 1)
            case_directory = case_directories[0]
            case = json.loads((case_directory / "case.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["template"], template)
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(case["status"], "succeeded")
            self.assertFalse(manifest["authoritative"])
            self.assertFalse(status["authoritative"])
            self.assertFalse(case["authoritative"])
            self.assertTrue(case["finite"])
            self.assertTrue(_finite_numbers(manifest))
            self.assertTrue(_finite_numbers(case))
            self.assertEqual(manifest["profile"]["physics_dt"], 0.001)
            self.assertEqual(result["profile"], "mujoco-cpu-wsl-smoke-v1")
            self.assertEqual(manifest["profile"]["name"], result["profile"])
            self.assertTrue(manifest["profile"]["use_mujoco_cpu"])
            self.assertTrue(manifest["profile"]["use_mujoco_contacts"])
            self.assertEqual(manifest["profile"]["video_fps"], 50)
            self.assertEqual(manifest["profile"]["video_width"], 640)
            self.assertEqual(manifest["profile"]["video_height"], 360)
            self.assertEqual(case["physics_steps"], 2000)
            self.assertEqual(case["video_frames"], 125)
            self.assertEqual(case["initial_hold_seconds"], 0.5)
            self.assertEqual(case["duration_seconds"], 2.0)
            self.assertEqual(case["video_duration_seconds"], 2.5)
            self.assertEqual(manifest["environment"]["execution"]["physics_device"], "cpu")
            self.assertEqual(manifest["environment"]["execution"]["warp_device"], "cpu")
            self.assertFalse(manifest["environment"]["execution"]["cuda_used"])
            self.assertEqual(
                manifest["environment"]["execution"]["rendering_device"],
                "software-cpu",
            )
            renderer = manifest["environment"]["execution"]["opengl_renderer"].casefold()
            self.assertTrue(
                any(marker in renderer for marker in ("llvmpipe", "softpipe", "swrast")),
                renderer,
            )

            for relative in EXPECTED_ARTIFACTS:
                path = run_directory / relative
                self.assertTrue(path.is_file() and path.stat().st_size > 0, relative)
            for filename in ("case.json", "video.mp4", "poster.jpg", "final.jpg"):
                path = case_directory / filename
                self.assertTrue(path.is_file() and path.stat().st_size > 0, filename)
            for image_path, expected_size in (
                (run_directory / "asset-cover.jpg", (640, 360)),
                (run_directory / "preview.jpg", (320, 180)),
                (case_directory / "poster.jpg", (640, 360)),
                (case_directory / "final.jpg", (640, 360)),
            ):
                with Image.open(io.BytesIO(image_path.read_bytes())) as image:
                    image.load()
                    self.assertEqual(image.size, expected_size)
            self.assertNotEqual(
                (run_directory / "asset-cover.jpg").read_bytes(),
                (case_directory / "poster.jpg").read_bytes(),
            )

            checksums = _parse_checksums(run_directory / "checksums.sha256")
            self.assertIn("asset-cover.jpg", checksums)
            self.assertIn(f"cases/{case_directory.name}/video.mp4", checksums)
            for relative, expected_digest in checksums.items():
                digest = hashlib.sha256((run_directory / relative).read_bytes()).hexdigest()
                self.assertEqual(digest, expected_digest, relative)

            probe = _probe_video(case_directory / "video.mp4")
            stream = probe["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual((stream["width"], stream["height"]), (640, 360))
            numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
            self.assertAlmostEqual(numerator / denominator, 50.0, places=6)
            if stream.get("nb_frames") not in (None, "N/A"):
                self.assertEqual(int(stream["nb_frames"]), 125)
            self.assertAlmostEqual(float(probe["format"]["duration"]), 2.5, delta=0.04)

            index = json.loads((data_root / "web" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["assets"][0]["runs"][0]["template"], template)
            return case, manifest

    def test_real_drop_smoke_end_to_end(self) -> None:
        self._run_and_validate("drop")

    def test_real_slope_smoke_end_to_end(self) -> None:
        case, manifest = self._run_and_validate("slope_friction")
        self.assertEqual(case["slope_angle_degrees"], 25.0)
        self.assertEqual(case["condition"]["slope_angle_degrees"], 25.0)
        self.assertIn(
            case["development_outcome"],
            {"moved", "stayed_near_start", "inconclusive"},
        )
        self.assertEqual(len(case["initial_position"]), 3)
        self.assertEqual(len(case["final_position"]), 3)
        self.assertEqual(len(case["final_linear_velocity"]), 3)
        self.assertTrue(math.isfinite(case["displacement_along_slope"]))
        self.assertFalse(manifest["profile"]["authoritative"])


if __name__ == "__main__":
    unittest.main()
