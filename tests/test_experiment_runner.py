from __future__ import annotations

import json
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from experiment_runner.assets import (
    ASSET_ENTRYPOINT,
    accept_asset,
    inspect_template_readiness,
    normalize_asset_identity,
)
from experiment_runner.config import (
    ControllerConfig,
    load_controller_config,
    save_controller_config,
)
from experiment_runner.controller import (
    _interactive,
    _require_data_root,
    build_remote_readiness_command,
    build_remote_result_command,
    discover_inbox_packages,
    open_remote_results,
    run_local_smoke,
)
from experiment_runner.profiles import describe_profiles, get_profile
from experiment_runner.provenance import resolve_git_commit
from experiment_runner.results import (
    build_result_index,
    create_run_scaffold,
    update_run_status,
)
from experiment_runner.storage import DataRoot, LAYOUT, atomic_write_json, read_json
from experiment_runner.web import create_result_server, install_static_site
from experiment_runner.video import jpeg_bytes


try:
    from pxr import Usd  # noqa: F401
except ImportError:
    HAS_USD = False
else:
    HAS_USD = True


READY_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "Box" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 1
        point3f physics:centerOfMass = (0, 0, 0)
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        quatf physics:principalAxes = (1, 0, 0, 0)

        def Cube "Geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MjcGeomAPI", "MaterialBindingAPI"]
        )
        {
            bool physics:collisionEnabled = true
            double size = 0.2
            rel material:binding:physics = </World/PhysicsMaterial>
            uniform int mjc:condim = 3
            uniform double[] mjc:solref = [0.02, 1.0]
            uniform double[] mjc:solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        }
    }

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "MjcMaterialAPI"]
    )
    {
        float physics:dynamicFriction = 0.5
        uniform double mjc:rollingfriction = 0.01
    }
}
"""

DROP_ONLY_USDA = READY_USDA.replace(
    "        float physics:dynamicFriction = 0.5\n"
    "        uniform double mjc:rollingfriction = 0.01\n",
    "",
)


class DataRootTests(unittest.TestCase):
    def test_initializes_fixed_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "newton-data")
            root.initialize()
            root.require_initialized()
            self.assertTrue(all((root.path / relative).is_dir() for relative in LAYOUT.values()))

    def test_rejects_unrelated_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root_path = Path(temporary) / "newton-data"
            root_path.mkdir()
            (root_path / "unrelated.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                DataRoot(root_path).initialize()

    def test_rejects_source_worktree_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "separate"):
                DataRoot(source / "data").initialize(source_root=source)

    def test_rechecks_source_worktree_overlap_after_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            root = DataRoot(source / "data")
            root.initialize()
            with self.assertRaisesRegex(ValueError, "separate"):
                root.require_initialized(source_root=source)

    def test_server_data_root_must_exist_before_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "administrator-has-not-created-this")
            with self.assertRaisesRegex(ValueError, "already exist"):
                root.require_existing_owned_directory()

    def test_atomic_json_is_utf8_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "value.json"
            atomic_write_json(target, {"name": "斜坡", "value": 25})
            self.assertEqual(read_json(target), {"name": "斜坡", "value": 25})
            self.assertFalse(any(target.parent.glob("*.tmp")))


class ControllerConfigTests(unittest.TestCase):
    def test_missing_config_uses_non_sensitive_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_controller_config(Path(temporary) / "missing.toml")
        self.assertEqual(config, ControllerConfig())

    def test_round_trips_paths_alias_ports_and_browser_preference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "controller.toml"
            expected = ControllerConfig(
                viewer_source=directory / "viewer-assets",
                data_root=directory / "experiment-data",
                ssh_alias="physics-server",
                local_port=8876,
                remote_port=8877,
                open_browser=False,
            )
            save_controller_config(expected, path)
            actual = load_controller_config(path)
            serialized = path.read_text(encoding="utf-8")
        self.assertEqual(actual, expected)
        self.assertNotIn("password", serialized)


class ControllerAssetDiscoveryTests(unittest.TestCase):
    def test_discovers_inbox_packages_by_required_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            package = root.location("inbox") / "kitchen" / "box-01"
            package.mkdir(parents=True)
            (package / ASSET_ENTRYPOINT).write_text("#usda 1.0\n", encoding="utf-8")
            (root.location("inbox") / "notes").mkdir()
            self.assertEqual(discover_inbox_packages(root), ["kitchen/box-01"])


class ProfileTests(unittest.TestCase):
    def test_profiles_have_integral_frame_steps(self) -> None:
        self.assertEqual(get_profile("mujoco-native-dt1ms-v1").physics_steps_per_video_frame, 20)
        smoke = get_profile("mujoco-cpu-wsl-smoke-v1")
        self.assertEqual(smoke.physics_steps_per_video_frame, 20)
        self.assertFalse(smoke.authoritative)
        self.assertTrue(smoke.use_mujoco_cpu)
        self.assertTrue(smoke.use_mujoco_contacts)

    def test_profiles_command_distinguishes_runnable_smoke_from_reserved_formal_profile(self) -> None:
        from experiment_runner.cli import main

        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            return_code = main(["profiles"])

        self.assertEqual(return_code, 0)
        profiles = json.loads(output.getvalue())
        formal = profiles["mujoco-native-dt1ms-v1"]
        smoke = profiles["mujoco-cpu-wsl-smoke-v1"]

        self.assertTrue(formal["authoritative"])
        self.assertEqual(formal["availability"]["status"], "reserved_not_runnable")
        self.assertFalse(formal["availability"]["runnable"])
        self.assertEqual(formal["availability"]["entrypoints"], [])

        self.assertFalse(smoke["authoritative"])
        self.assertEqual(smoke["availability"]["status"], "development_smoke_only")
        self.assertTrue(smoke["availability"]["runnable"])
        self.assertEqual(
            smoke["availability"]["entrypoints"],
            ["smoke-drop", "smoke-slope"],
        )
        smoke["availability"]["entrypoints"].append("mutated-by-caller")
        self.assertEqual(
            describe_profiles()["mujoco-cpu-wsl-smoke-v1"]["availability"]["entrypoints"],
            ["smoke-drop", "smoke-slope"],
        )

    def test_profiles_help_describes_registration_and_availability_not_approval(self) -> None:
        from experiment_runner.cli import build_parser

        help_text = build_parser().format_help()
        self.assertIn("registered profiles and execution availability", help_text)
        self.assertNotIn("approved experiment profiles", help_text)

    def test_smoke_preview_is_resized_to_profile_dimensions(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        encoded = jpeg_bytes(frame, size=(320, 180))
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(image.size, (320, 180))

    def test_initial_hold_is_added_to_physics_duration(self) -> None:
        from experiment_runner.experiments.drop import recording_frame_counts

        total_frames, hold_frames, simulation_frames = recording_frame_counts(
            duration_seconds=2.0,
            initial_hold_seconds=0.5,
            fps=50,
        )
        self.assertEqual((total_frames, hold_frames, simulation_frames), (125, 25, 100))

    def test_cpu_smoke_environment_hides_cuda_and_requires_software_opengl(self) -> None:
        from experiment_runner.cpu_safety import prepare_cpu_smoke_environment

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("experiment_runner.cpu_safety.os.name", "posix"):
                prepare_cpu_smoke_environment()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                self.assertEqual(os.environ["LIBGL_ALWAYS_SOFTWARE"], "true")
                self.assertEqual(os.environ["MESA_LOADER_DRIVER_OVERRIDE"], "swrast")
                self.assertEqual(os.environ["NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL"], "1")

    def test_software_renderer_check_fails_closed(self) -> None:
        from experiment_runner.cpu_safety import require_software_opengl_renderer

        self.assertEqual(
            require_software_opengl_renderer("llvmpipe (LLVM 19.1.1, 256 bits)"),
            "llvmpipe (LLVM 19.1.1, 256 bits)",
        )
        with self.assertRaisesRegex(RuntimeError, "software OpenGL"):
            require_software_opengl_renderer("NVIDIA GeForce RTX 3090")

    def test_warp_default_is_explicitly_locked_to_cpu(self) -> None:
        from experiment_runner.experiments import drop

        cpu_device = mock.Mock(is_cpu=True)
        with mock.patch.object(drop.wp, "set_device") as set_device:
            with mock.patch.object(drop.wp, "get_device", return_value=cpu_device):
                drop.configure_warp_cpu_only()
        set_device.assert_called_once_with("cpu")

    def test_asset_cover_arguments_exclude_the_ground_plane(self) -> None:
        from experiment_runner.experiments.drop import _viewer_arguments

        arguments = _viewer_arguments(
            Path("asset.usda"),
            z_offset=0.0,
            profile=get_profile("mujoco-cpu-wsl-smoke-v1"),
            include_ground=False,
        )
        self.assertFalse(arguments.ground)

    def test_cpu_viewer_skips_cuda_pinned_buffers(self) -> None:
        from experiment_runner.experiments.drop import CpuOnlyViewerGL

        viewer = object.__new__(CpuOnlyViewerGL)
        viewer.device = mock.Mock(is_cpu=True)
        viewer._packed_groups = ["stale"]
        viewer._capsule_keys = {"stale"}
        viewer._packed_write_indices = object()
        viewer._packed_world_xforms = object()
        viewer._packed_vbo_xforms = object()
        viewer._packed_vbo_xforms_host = object()
        viewer._build_packed_vbo_arrays()
        self.assertEqual(viewer._packed_groups, [])
        self.assertEqual(viewer._capsule_keys, set())
        self.assertIsNone(viewer._packed_write_indices)
        self.assertIsNone(viewer._packed_world_xforms)
        self.assertIsNone(viewer._packed_vbo_xforms)
        self.assertIsNone(viewer._packed_vbo_xforms_host)


class ProvenanceTests(unittest.TestCase):
    def test_explicit_commit_must_match_readable_source_repository(self) -> None:
        completed = mock.Mock(returncode=0, stdout="a" * 40 + "\n", stderr="")
        with mock.patch("experiment_runner.provenance.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_git_commit("b" * 40, source_root=Path("source"))


class AssetIdentityTests(unittest.TestCase):
    def test_normalizes_separators_and_unicode(self) -> None:
        self.assertEqual(normalize_asset_identity("kitchen\\microwave-01"), "kitchen/microwave-01")

    def test_rejects_absolute_parent_and_empty_segments(self) -> None:
        for value in ("/tmp/asset", "C:/asset", "../asset", "a//b", "a/./b", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_asset_identity(value)


@unittest.skipUnless(HAS_USD, "OpenUSD bindings are not installed")
class AssetAcceptanceTests(unittest.TestCase):
    def _write_package(self, root: DataRoot, identity: str, content: str) -> Path:
        package = root.location("inbox").joinpath(*identity.split("/"))
        package.mkdir(parents=True)
        (package / "newton-mujoco.usda").write_text(content, encoding="utf-8")
        return package

    def test_readiness_is_per_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "newton-mujoco.usda"
            entrypoint.write_text(DROP_ONLY_USDA, encoding="utf-8")
            readiness = inspect_template_readiness(entrypoint)
            self.assertEqual(readiness["drop"]["status"], "ready")
            self.assertEqual(readiness["slope_friction"]["status"], "not_ready")

    def test_slope_requires_collision_shape_physics_material_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "newton-mujoco.usda"
            entrypoint.write_text(
                READY_USDA.replace(
                    "            rel material:binding:physics = </World/PhysicsMaterial>\n",
                    "",
                ),
                encoding="utf-8",
            )
            readiness = inspect_template_readiness(entrypoint)
            self.assertEqual(readiness["drop"]["status"], "ready")
            self.assertIn(
                "missing_physics_material_binding",
                readiness["slope_friction"]["reason_codes"],
            )

    def test_accepts_and_moves_ready_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            identity = "kitchen/box-01"
            source = self._write_package(root, identity, READY_USDA)
            report = accept_asset(
                root,
                identity,
                git_commit=resolve_git_commit(),
                enforce_readonly=False,
            )
            self.assertEqual(report["status"], "accepted")
            self.assertFalse(source.exists())
            destination = root.location("assets").joinpath(
                *identity.split("/"),
                report["asset_version"],
                "newton-mujoco.usda",
            )
            self.assertTrue(destination.is_file())
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotIn(str(root.path), serialized)

    def test_acceptance_report_always_records_a_complete_git_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            self._write_package(root, "props/versioned", READY_USDA)
            with mock.patch(
                "experiment_runner.assets.resolve_git_commit",
                return_value="c" * 40,
            ):
                report = accept_asset(root, "props/versioned", enforce_readonly=False)
        self.assertEqual(report["git_commit"], "c" * 40)

    def test_partial_ready_asset_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            self._write_package(root, "props/drop-only", DROP_ONLY_USDA)
            report = accept_asset(root, "props/drop-only", enforce_readonly=False)
            self.assertEqual(report["status"], "partially_ready")
            self.assertEqual(report["ready_templates"], ["drop"])

    def test_exact_duplicate_stays_in_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            identity = "props/duplicate"
            self._write_package(root, identity, READY_USDA)
            first = accept_asset(root, identity, enforce_readonly=False)
            self.assertEqual(first["status"], "accepted")
            duplicate_source = self._write_package(root, identity, READY_USDA)
            duplicate = accept_asset(root, identity, enforce_readonly=False)
            self.assertEqual(duplicate["status"], "duplicate")
            self.assertTrue(duplicate_source.is_dir())

    def test_case_only_identity_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            self._write_package(root, "Props/Box", READY_USDA)
            accepted = accept_asset(root, "Props/Box", enforce_readonly=False)
            self.assertEqual(accepted["status"], "accepted")
            source = self._write_package(root, "props/box", READY_USDA)
            collision = accept_asset(root, "props/box", enforce_readonly=False)
            self.assertEqual(collision["status"], "failed")
            self.assertEqual(collision["error"]["code"], "asset_identity_case_collision")
            self.assertTrue(source.is_dir())


class CpuSmokeOutputTests(unittest.TestCase):
    def test_asset_cover_is_rendered_independently_from_case_poster(self) -> None:
        from experiment_runner.smoke import run_cpu_smoke_drop

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            package = root.location("assets") / "fixtures" / "box" / ("a" * 64)
            package.mkdir(parents=True)
            asset_path = package / ASSET_ENTRYPOINT
            asset_path.write_text("#usda 1.0\n", encoding="utf-8")
            snapshot = {
                "identity": "fixtures/box",
                "version": "a" * 64,
                "entrypoint": ASSET_ENTRYPOINT,
                "dependencies": [],
                "readiness": {
                    "drop": {"status": "ready", "reason_codes": []},
                    "slope_friction": {"status": "not_ready", "reason_codes": ["fixture"]},
                },
                "package_root": package,
            }
            geometry = mock.Mock(
                clearance=0.2,
                characteristic_length=0.2,
                effective_length=0.2,
                initial_bounds=mock.Mock(),
            )

            def write_cover(*_args, output_path: Path, **_kwargs):
                output_path.write_bytes(b"asset-only-cover")
                return {"device": "software-cpu", "renderer": "llvmpipe", "vendor": "Mesa"}

            def write_case(_scene, *, output_directory: Path, **_kwargs):
                (output_directory / "video.mp4").write_bytes(b"video")
                (output_directory / "poster.jpg").write_bytes(b"case-poster")
                (output_directory / "final.jpg").write_bytes(b"case-final")
                (output_directory.parent.parent / "preview.jpg").write_bytes(b"preview")
                return {
                    "duration_seconds": 2.0,
                    "rendering": {
                        "device": "software-cpu",
                        "renderer": "llvmpipe",
                        "vendor": "Mesa",
                    },
                }

            environment = dict(os.environ)
            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch("experiment_runner.smoke.resolve_git_commit", return_value="b" * 40):
                    with mock.patch("experiment_runner.smoke.snapshot_asset_version", return_value=snapshot):
                        with mock.patch(
                            "experiment_runner.experiments.drop.measure_drop_geometry",
                            return_value=geometry,
                        ):
                            with mock.patch(
                                "experiment_runner.experiments.drop.create_drop_scene",
                                return_value=object(),
                            ):
                                with mock.patch(
                                    "experiment_runner.experiments.drop.render_asset_cover",
                                    side_effect=write_cover,
                                ) as render_cover:
                                    with mock.patch(
                                        "experiment_runner.experiments.drop.record_drop_case",
                                        side_effect=write_case,
                                    ):
                                        result = run_cpu_smoke_drop(
                                            root,
                                            asset_identity="fixtures/box",
                                            asset_version="a" * 64,
                                        )

            run_directory = root.location("runs") / result["run_id"]
            cover = (run_directory / "asset-cover.jpg").read_bytes()
            poster = (run_directory / "cases" / "001-medium" / "poster.jpg").read_bytes()
            self.assertEqual(cover, b"asset-only-cover")
            self.assertNotEqual(cover, poster)
            render_cover.assert_called_once()


class ResultBrowserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = DataRoot(Path(self.temporary.name) / "data")
        self.root.initialize()
        self.run_id = "drop-20260807T120000Z-12345678"
        run_directory = create_run_scaffold(
            self.root,
            run_id=self.run_id,
            batch_id="batch-20260807T120000Z-12345678",
            template="drop",
            asset={
                "identity": "kitchen/box-01",
                "version": "a" * 64,
            },
            profile=get_profile("mujoco-cpu-wsl-smoke-v1").expanded(),
            git_commit="b" * 40,
        )
        manifest_path = run_directory / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["started_at"] = "2026-08-07T12:00:00Z"
        manifest["finished_at"] = "2026-08-07T12:00:02Z"
        atomic_write_json(manifest_path, manifest)

        case_directory = run_directory / "cases" / "001-medium"
        case_directory.mkdir()
        atomic_write_json(
            case_directory / "case.json",
            {
                "schema_version": 1,
                "case_id": "drop-medium",
                "label": "中等高度",
                "status": "succeeded",
                "condition": {"clearance_m": 0.4},
                "duration_seconds": 2.0,
                "started_at": "2026-08-07T12:00:00Z",
                "finished_at": "2026-08-07T12:00:02Z",
            },
        )
        (case_directory / "video.mp4").write_bytes(b"0123456789abcdef")
        (case_directory / "poster.jpg").write_bytes(b"poster")
        (case_directory / "final.jpg").write_bytes(b"final")
        (run_directory / "asset-cover.jpg").write_bytes(b"asset-only-cover")
        update_run_status(
            run_directory,
            status="succeeded",
            phase="finished",
            progress="1/1 cases complete",
            current_case=None,
            completed_cases=1,
            total_cases=1,
            result_files=["cases/001-medium/video.mp4"],
        )
        install_static_site(self.root)
        build_result_index(self.root)

    def test_index_is_asset_first_and_contains_inline_video(self) -> None:
        index = read_json(self.root.location("web") / "index.json")
        self.assertEqual(len(index["assets"]), 1)
        asset = index["assets"][0]
        self.assertEqual(asset["identity"], "kitchen/box-01")
        self.assertEqual(asset["cover_url"], f"/runs/{self.run_id}/asset-cover.jpg")
        test_case = asset["runs"][0]["cases"][0]
        self.assertEqual(
            test_case["video_url"],
            f"/runs/{self.run_id}/cases/001-medium/video.mp4",
        )

    def test_manifest_records_reproducible_runtime_environment(self) -> None:
        manifest = read_json(
            self.root.location("runs") / self.run_id / "manifest.json"
        )
        environment = manifest["environment"]
        self.assertEqual(environment["software"]["python"], "3.12.13")
        for package in ("newton", "warp-lang", "mujoco", "mujoco-warp"):
            self.assertIn(package, environment["software"])
        self.assertIn("machine", environment["hardware"])
        self.assertEqual(environment["execution"]["physics_device"], "cpu")
        self.assertFalse(environment["execution"]["cuda_used"])

    def test_read_only_server_supports_head_and_byte_ranges(self) -> None:
        server = create_result_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"

        with urlopen(f"{base}/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"Newton", response.read())

        video_url = f"{base}/runs/{self.run_id}/cases/001-medium/video.mp4"
        range_request = Request(video_url, headers={"Range": "bytes=3-7"})
        with urlopen(range_request, timeout=5) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 3-7/16")
            self.assertEqual(response.read(), b"34567")

        head_request = Request(video_url, method="HEAD")
        with urlopen(head_request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Content-Length"], "16")
            self.assertEqual(response.read(), b"")

        with self.assertRaises(HTTPError) as post_error:
            urlopen(Request(f"{base}/index.json", method="POST"), timeout=5)
        self.assertEqual(post_error.exception.code, 405)

        with self.assertRaises(HTTPError) as traversal_error:
            urlopen(f"{base}/runs/%2e%2e/web/index.json", timeout=5)
        self.assertEqual(traversal_error.exception.code, 404)

        with self.assertRaises(HTTPError) as private_root_error:
            urlopen(f"{base}/assets/anything", timeout=5)
        self.assertEqual(private_root_error.exception.code, 404)

        multiple_range = Request(video_url, headers={"Range": "bytes=0-1,4-5"})
        with self.assertRaises(HTTPError) as range_error:
            urlopen(multiple_range, timeout=5)
        self.assertEqual(range_error.exception.code, 416)


class ControllerTests(unittest.TestCase):
    def test_first_run_setup_can_be_skipped_and_menu_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "controller.toml"
            answers = iter(["", "", "", "", "", "", "8", "0"])
            with mock.patch("builtins.input", side_effect=lambda _prompt="": next(answers)):
                with mock.patch("builtins.print") as output:
                    return_code = _interactive(config_path=config_path)
        self.assertEqual(return_code, 0)
        rendered = "\n".join(" ".join(map(str, call.args)) for call in output.call_args_list)
        self.assertIn("当前已可用", rendered)
        self.assertIn("尚未开放", rendered)

    def test_remote_readiness_command_is_fixed_and_read_only(self) -> None:
        command = build_remote_readiness_command(
            ssh_executable="ssh.exe",
            host_alias="physics-server",
        )
        self.assertEqual(command[-4:], ["physics-server", "command", "-v", "newton-test-remote"])
        self.assertNotIn("sudo", command)
        self.assertNotIn("install", command)

    def test_remote_browser_command_uses_alias_and_loopback_only(self) -> None:
        command = build_remote_result_command(
            ssh_executable="ssh.exe",
            host_alias="physics-server",
            local_port=8765,
            remote_port=8877,
        )
        self.assertIn("physics-server", command)
        self.assertIn("127.0.0.1:8765:127.0.0.1:8877", command)
        self.assertNotIn("0.0.0.0", command)
        self.assertNotIn("StrictHostKeyChecking=no", command)
        self.assertEqual(command[-7:], [
            "newton-test-remote",
            "serve-results",
            "--host",
            "127.0.0.1",
            "--port",
            "8877",
            "--exit-when-stdin-closes",
        ])

    def test_remote_browser_command_rejects_shell_text(self) -> None:
        with self.assertRaises(ValueError):
            build_remote_result_command(
                ssh_executable="ssh.exe",
                host_alias="server;whoami",
                local_port=8765,
                remote_port=8765,
            )

    def test_local_smoke_passes_asset_selection_as_subprocess_data(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch(
            "experiment_runner.controller.subprocess.run",
            return_value=completed,
        ) as run:
            return_code = run_local_smoke(
                Path("D:/newton-data"),
                asset_identity="fixtures/blue box",
                asset_version="a" * 64,
            )
        self.assertEqual(return_code, 0)
        command = run.call_args.args[0]
        self.assertIn("fixtures/blue box", command)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertFalse(run.call_args.kwargs["check"])

    def test_remote_browser_timeout_always_stops_ssh_process(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 143
        with mock.patch("experiment_runner.controller.shutil.which", return_value="ssh.exe"):
            with mock.patch("experiment_runner.controller.subprocess.Popen", return_value=process):
                with mock.patch(
                    "experiment_runner.controller._wait_for_local_port",
                    side_effect=RuntimeError("timeout"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "timeout"):
                        open_remote_results(host_alias="physics-server", open_browser=False)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_reusing_data_root_rechecks_viewer_and_source_separation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            data = source / "data"
            DataRoot(data).initialize()
            config = ControllerConfig(viewer_source=Path(temporary) / "viewer", data_root=data)
            with mock.patch("experiment_runner.controller._project_root", return_value=source):
                with self.assertRaisesRegex(ValueError, "separate"):
                    _require_data_root(config)

    def test_server_storage_parser_requires_repeated_admin_approved_path(self) -> None:
        from experiment_runner.cli import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["configure-storage", "D:/server-data"])

    def test_every_server_command_rechecks_data_root_ownership(self) -> None:
        from experiment_runner.cli import _data_root

        args = mock.Mock(
            data_root=Path("D:/server-data"),
            config=Path("D:/server-config.toml"),
        )
        with mock.patch.object(DataRoot, "require_existing_owned_directory") as ownership:
            _data_root(args)
        ownership.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
