from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from asset_viewer.camera import AssetBounds
from experiment_runner.config import ControllerConfig
from experiment_runner.profiles import get_profile
from experiment_runner.results import build_result_index, create_run_scaffold, update_run_status
from experiment_runner.storage import DataRoot, atomic_write_json, read_json


FIXTURE = Path(__file__).parent / "fixtures" / "smoke_asset" / "newton-mujoco.usda"


class SlopeGeometryTests(unittest.TestCase):
    def test_basis_points_downhill_and_gravity_has_positive_down_slope_component(self) -> None:
        from experiment_runner.experiments.slope import slope_basis

        down_slope, cross_slope, surface_normal = slope_basis(25.0)

        np.testing.assert_allclose(np.linalg.norm(down_slope), 1.0)
        np.testing.assert_allclose(np.linalg.norm(cross_slope), 1.0)
        np.testing.assert_allclose(np.linalg.norm(surface_normal), 1.0)
        np.testing.assert_allclose(np.dot(down_slope, surface_normal), 0.0, atol=1.0e-12)
        np.testing.assert_allclose(cross_slope, (0.0, 1.0, 0.0), atol=1.0e-12)
        self.assertGreater(down_slope[0], 0.0)
        self.assertLess(down_slope[2], 0.0)
        self.assertGreater(np.dot((0.0, 0.0, -9.81), down_slope), 0.0)

    def test_planned_asset_is_above_ramp_without_penetration(self) -> None:
        from experiment_runner.experiments.slope import (
            asset_bounds_corners,
            plan_slope_geometry,
            transform_asset_points,
        )

        bounds = AssetBounds(
            minimum=np.array((-0.05, -0.04, -0.03)),
            maximum=np.array((0.05, 0.04, 0.07)),
        )
        geometry = plan_slope_geometry(bounds, angle_degrees=25.0)
        world_points = transform_asset_points(
            asset_bounds_corners(bounds),
            geometry.asset_translation,
            geometry.asset_rotation_xyzw,
        )
        relative = world_points - geometry.ramp_center
        normal_distance = relative @ geometry.surface_normal
        down_slope_distance = relative @ geometry.down_slope
        cross_slope_distance = relative @ geometry.cross_slope

        np.testing.assert_allclose(
            normal_distance.min() - geometry.ramp_thickness / 2.0,
            geometry.surface_gap,
            atol=1.0e-10,
        )
        self.assertGreater(geometry.surface_gap, 0.0)
        self.assertGreaterEqual(down_slope_distance.min(), -geometry.ramp_length / 2.0)
        self.assertLessEqual(down_slope_distance.max(), geometry.ramp_length / 2.0)
        self.assertGreaterEqual(cross_slope_distance.min(), -geometry.ramp_width / 2.0)
        self.assertLessEqual(cross_slope_distance.max(), geometry.ramp_width / 2.0)

    def test_along_slope_measurement_and_development_outcome_are_non_formal(self) -> None:
        from experiment_runner.experiments.slope import (
            classify_development_outcome,
            measure_along_slope_displacement,
            slope_basis,
        )

        down_slope, _cross, _normal = slope_basis(25.0)
        initial = np.array((0.1, 0.2, 0.3))
        final = initial + down_slope * 0.2
        self.assertAlmostEqual(
            measure_along_slope_displacement(initial, final, down_slope),
            0.2,
        )
        self.assertEqual(classify_development_outcome(0.2, effective_length=0.1), "moved")
        self.assertEqual(
            classify_development_outcome(0.001, effective_length=0.1),
            "stayed_near_start",
        )
        self.assertEqual(
            classify_development_outcome(0.003, effective_length=0.1),
            "inconclusive",
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            classify_development_outcome(math.nan, effective_length=0.1)

    def test_reference_surface_priority_is_lower_without_int32_overflow(self) -> None:
        from experiment_runner.experiments.slope import reference_surface_priorities

        self.assertEqual(reference_surface_priorities([2, 0, 1]), (0, -1))
        with self.assertRaisesRegex(ValueError, "minimum MuJoCo contact priority"):
            reference_surface_priorities([np.iinfo(np.int32).min])

    def test_real_scene_starts_at_rest_above_the_ramp(self) -> None:
        from experiment_runner.experiments.slope import create_slope_scene, measure_slope_geometry

        profile = get_profile("mujoco-cpu-wsl-smoke-v1")
        geometry = measure_slope_geometry(FIXTURE, profile=profile, angle_degrees=25.0)
        scene = create_slope_scene(FIXTURE, profile=profile, geometry=geometry)

        velocities = np.asarray(scene.state.body_qd.numpy(), dtype=np.float64)
        self.assertTrue(np.isfinite(velocities).all())
        np.testing.assert_allclose(velocities, 0.0, atol=0.0)
        self.assertGreater(geometry.surface_gap, 0.0)
        self.assertEqual(scene.ramp_shape_index, scene.model.shape_count - 1)
        priorities = np.asarray(scene.model.mujoco.geom_priority.numpy()).reshape(-1)
        self.assertEqual(priorities[scene.ramp_shape_index], scene.ramp_contact_priority)
        self.assertLess(scene.ramp_contact_priority, scene.asset_min_contact_priority)
        friction = np.asarray(scene.model.shape_material_mu.numpy()).reshape(-1)
        rolling = np.asarray(scene.model.shape_material_mu_rolling.numpy()).reshape(-1)
        self.assertAlmostEqual(friction[0], 0.5)
        self.assertAlmostEqual(rolling[0], 0.01)
        self.assertTrue(profile.use_mujoco_cpu)
        self.assertTrue(profile.use_mujoco_contacts)

    def test_create_slope_scene_single_body_fallback_explains_policy_boundary(self) -> None:
        from experiment_runner.experiments.slope import create_slope_scene

        builder = mock.Mock(body_count=2)
        geometry = mock.Mock(asset_translation=np.zeros(3), asset_rotation_xyzw=np.zeros(4))
        with mock.patch("experiment_runner.experiments.slope.configure_warp_cpu_only"):
            with mock.patch(
                "experiment_runner.experiments.slope.newton.ModelBuilder",
                return_value=builder,
            ):
                with mock.patch(
                    "experiment_runner.experiments.slope._wp_transform",
                    return_value=object(),
                ):
                    with mock.patch(
                        "experiment_runner.experiments.slope.mujoco_usd_schema_resolvers",
                        return_value=[],
                    ):
                        with self.assertRaises(ValueError) as raised:
                            create_slope_scene(
                                FIXTURE,
                                profile=get_profile("mujoco-cpu-wsl-smoke-v1"),
                                geometry=geometry,
                            )

        message = str(raised.exception)
        self.assertIn("fixed 25-degree local CPU slope smoke", message)
        self.assertIn("exactly one dynamic rigid body", message)
        self.assertIn("may still pass generic slope_friction readiness", message)
        self.assertIn("not a permanent limit on future formal slope experiments", message)
        self.assertIn("detected 2", message)


class SlopeSmokeRunnerTests(unittest.TestCase):
    def _snapshot(
        self,
        root: DataRoot,
        *,
        ready: bool = True,
        rigid_body_count: int = 1,
    ) -> dict[str, object]:
        package = root.location("assets") / "fixtures" / "box" / ("a" * 64)
        package.mkdir(parents=True)
        (package / "newton-mujoco.usda").write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "identity": "fixtures/box",
            "version": "a" * 64,
            "entrypoint": "newton-mujoco.usda",
            "dependencies": [],
            "readiness": {
                "drop": {"status": "ready", "reason_codes": []},
                "slope_friction": {
                    "status": "ready" if ready else "not_ready",
                    "reason_codes": [] if ready else ["missing_physics_material_binding"],
                    "checks": {"rigid_body_count": rigid_body_count},
                },
            },
            "package_root": package,
        }

    def test_rejects_asset_that_is_not_slope_ready_before_creating_a_run(self) -> None:
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root, ready=False)
            with mock.patch("experiment_runner.smoke.snapshot_asset_version", return_value=snapshot):
                with self.assertRaisesRegex(ValueError, "not ready for slope_friction"):
                    run_cpu_smoke_slope(
                        root,
                        asset_identity="fixtures/box",
                        asset_version="a" * 64,
                    )
            self.assertEqual(list(root.location("runs").iterdir()), [])
            self.assertEqual(list(root.location("batches").iterdir()), [])

    def test_rejects_multi_body_asset_before_geometry_or_run_directories(self) -> None:
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root, rigid_body_count=2)
            with mock.patch(
                "experiment_runner.smoke.snapshot_asset_version",
                return_value=snapshot,
            ):
                with mock.patch(
                    "experiment_runner.experiments.slope.measure_slope_geometry"
                ) as measure:
                    with self.assertRaises(ValueError) as raised:
                        run_cpu_smoke_slope(
                            root,
                            asset_identity="fixtures/box",
                            asset_version="a" * 64,
                        )
            measure.assert_not_called()
            message = str(raised.exception)
            self.assertIn("fixed 25-degree local CPU slope smoke", message)
            self.assertIn("exactly one dynamic rigid body", message)
            self.assertIn("may still pass generic slope_friction readiness", message)
            self.assertIn("not a permanent limit on future formal slope experiments", message)
            self.assertIn("detected 2", message)
            self.assertEqual(list(root.location("runs").iterdir()), [])
            self.assertEqual(list(root.location("batches").iterdir()), [])

    def test_rejects_boolean_body_count_as_unknown_before_creating_directories(self) -> None:
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root, rigid_body_count=True)
            with mock.patch(
                "experiment_runner.smoke.snapshot_asset_version",
                return_value=snapshot,
            ):
                with self.assertRaises(ValueError) as raised:
                    run_cpu_smoke_slope(
                        root,
                        asset_identity="fixtures/box",
                        asset_version="a" * 64,
                    )

            self.assertIn("detected unknown", str(raised.exception))
            self.assertNotIn("detected True", str(raised.exception))
            self.assertEqual(list(root.location("runs").iterdir()), [])
            self.assertEqual(list(root.location("batches").iterdir()), [])

    def test_rejects_missing_body_count_as_unknown_before_creating_directories(self) -> None:
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root)
            del snapshot["readiness"]["slope_friction"]["checks"]["rigid_body_count"]
            with mock.patch(
                "experiment_runner.smoke.snapshot_asset_version",
                return_value=snapshot,
            ):
                with self.assertRaises(ValueError) as raised:
                    run_cpu_smoke_slope(
                        root,
                        asset_identity="fixtures/box",
                        asset_version="a" * 64,
                    )

            self.assertIn("detected unknown", str(raised.exception))
            self.assertEqual(list(root.location("runs").iterdir()), [])
            self.assertEqual(list(root.location("batches").iterdir()), [])

    def test_mocked_slope_run_writes_non_authoritative_schema_and_web_group(self) -> None:
        from experiment_runner.experiments.slope import plan_slope_geometry
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root)
            geometry = plan_slope_geometry(
                AssetBounds(np.full(3, -0.05), np.full(3, 0.05)),
                angle_degrees=25.0,
            )

            def write_cover(*_args, output_path: Path, **_kwargs):
                output_path.write_bytes(b"asset-cover")
                return {"device": "software-cpu", "renderer": "llvmpipe", "vendor": "Mesa"}

            def write_case(_scene, *, output_directory: Path, **_kwargs):
                (output_directory / "video.mp4").write_bytes(b"video")
                (output_directory / "poster.jpg").write_bytes(b"poster")
                (output_directory / "final.jpg").write_bytes(b"final")
                (output_directory.parent.parent / "preview.jpg").write_bytes(b"preview")
                return {
                    "duration_seconds": 2.0,
                    "initial_hold_seconds": 0.5,
                    "video_duration_seconds": 2.5,
                    "video_frames": 125,
                    "physics_steps": 2000,
                    "slope_angle_degrees": 25.0,
                    "initial_position": [0.0, 0.0, 1.0],
                    "final_position": [0.2, 0.0, 0.9],
                    "displacement_along_slope": 0.22,
                    "final_linear_velocity": [0.1, 0.0, -0.04],
                    "finite": True,
                    "development_outcome": "moved",
                    "rendering": {
                        "device": "software-cpu",
                        "renderer": "llvmpipe",
                        "vendor": "Mesa",
                    },
                }

            with mock.patch.dict(os.environ, dict(os.environ), clear=True):
                with mock.patch("experiment_runner.smoke.resolve_git_commit", return_value="b" * 40):
                    with mock.patch("experiment_runner.smoke.snapshot_asset_version", return_value=snapshot):
                        with mock.patch(
                            "experiment_runner.experiments.slope.measure_slope_geometry",
                            return_value=geometry,
                        ):
                            with mock.patch(
                                "experiment_runner.experiments.slope.create_slope_scene",
                                return_value=object(),
                            ):
                                with mock.patch(
                                    "experiment_runner.experiments.drop.render_asset_cover",
                                    side_effect=write_cover,
                                ):
                                    with mock.patch(
                                        "experiment_runner.experiments.slope.record_slope_case",
                                        side_effect=write_case,
                                    ):
                                        result = run_cpu_smoke_slope(
                                            root,
                                            asset_identity="fixtures/box",
                                            asset_version="a" * 64,
                                        )

            run_directory = root.location("runs") / result["run_id"]
            case = read_json(run_directory / "cases" / "001-slope-25deg" / "case.json")
            manifest = read_json(run_directory / "manifest.json")
            index = read_json(root.location("web") / "index.json")
            self.assertEqual(manifest["template"], "slope_friction")
            self.assertEqual(manifest["profile"]["name"], "mujoco-cpu-wsl-smoke-v1")
            self.assertTrue(manifest["profile"]["use_mujoco_cpu"])
            self.assertTrue(manifest["profile"]["use_mujoco_contacts"])
            self.assertFalse(manifest["profile"]["authoritative"])
            self.assertFalse(manifest["authoritative"])
            self.assertFalse(read_json(run_directory / "status.json")["authoritative"])
            self.assertEqual(case["condition"]["slope_angle_degrees"], 25.0)
            self.assertEqual(case["development_outcome"], "moved")
            self.assertFalse(case["authoritative"])
            public_run = index["assets"][0]["runs"][0]
            self.assertEqual(public_run["template"], "slope_friction")
            self.assertFalse(public_run["authoritative"])
            self.assertEqual(public_run["cases"][0]["development_outcome"], "moved")
            run_log = (run_directory / "run.log").read_text(encoding="utf-8")
            self.assertNotIn("software-rendering policy enabled", run_log)
            self.assertIn("MuJoCo/Warp physics locked to CPU", run_log)
            self.assertIn("CUDA hidden", run_log)
            self.assertIn(
                "Viewer rendering initialized; device=software-cpu; "
                "renderer=llvmpipe; vendor=Mesa",
                run_log,
            )

    def test_recording_failure_is_auditable(self) -> None:
        from experiment_runner.experiments.slope import plan_slope_geometry
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root)
            geometry = plan_slope_geometry(
                AssetBounds(np.full(3, -0.05), np.full(3, 0.05)),
                angle_degrees=25.0,
            )
            with mock.patch("experiment_runner.smoke.resolve_git_commit", return_value="b" * 40):
                with mock.patch("experiment_runner.smoke.snapshot_asset_version", return_value=snapshot):
                    with mock.patch(
                        "experiment_runner.experiments.slope.measure_slope_geometry",
                        return_value=geometry,
                    ):
                        with mock.patch(
                            "experiment_runner.experiments.drop.render_asset_cover",
                            return_value={"device": "software-cpu", "renderer": "llvmpipe", "vendor": "Mesa"},
                        ):
                            with mock.patch(
                                "experiment_runner.experiments.slope.create_slope_scene",
                                return_value=object(),
                            ):
                                with mock.patch(
                                    "experiment_runner.experiments.slope.record_slope_case",
                                    side_effect=RuntimeError("non-finite fixture"),
                                ):
                                    with self.assertRaisesRegex(RuntimeError, "non-finite fixture"):
                                        run_cpu_smoke_slope(
                                            root,
                                            asset_identity="fixtures/box",
                                            asset_version="a" * 64,
                                        )

            run_directory = next(root.location("runs").iterdir())
            case = read_json(run_directory / "cases" / "001-slope-25deg" / "case.json")
            manifest = read_json(run_directory / "manifest.json")
            status = read_json(run_directory / "status.json")
            self.assertEqual(case["status"], "failed")
            self.assertEqual(case["failure"]["code"], "cpu_smoke_slope_failed")
            self.assertEqual(manifest["exit_code"], 1)
            self.assertEqual(status["status"], "failed")
            self.assertFalse(manifest["authoritative"])
            self.assertFalse(status["authoritative"])
            self.assertTrue((run_directory / "checksums.sha256").is_file())

    def test_slope_recording_rejects_non_finite_state(self) -> None:
        from experiment_runner.experiments.slope import _require_finite_state

        scene = mock.Mock()
        scene.state.body_q.numpy.return_value = np.array([[math.nan] * 7])
        scene.state.body_qd.numpy.return_value = np.zeros((1, 6))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _require_finite_state(scene)

    def test_keyboard_interrupt_does_not_leave_slope_run_running(self) -> None:
        from experiment_runner.experiments.slope import plan_slope_geometry
        from experiment_runner.smoke import run_cpu_smoke_slope

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = self._snapshot(root)
            geometry = plan_slope_geometry(
                AssetBounds(np.full(3, -0.05), np.full(3, 0.05)),
                angle_degrees=25.0,
            )
            with mock.patch("experiment_runner.smoke.resolve_git_commit", return_value="b" * 40):
                with mock.patch("experiment_runner.smoke.snapshot_asset_version", return_value=snapshot):
                    with mock.patch(
                        "experiment_runner.experiments.slope.measure_slope_geometry",
                        return_value=geometry,
                    ):
                        with mock.patch(
                            "experiment_runner.experiments.drop.render_asset_cover",
                            return_value={"device": "software-cpu", "renderer": "llvmpipe", "vendor": "Mesa"},
                        ):
                            with mock.patch(
                                "experiment_runner.experiments.slope.create_slope_scene",
                                return_value=object(),
                            ):
                                with mock.patch(
                                    "experiment_runner.experiments.slope.record_slope_case",
                                    side_effect=KeyboardInterrupt,
                                ):
                                    with self.assertRaises(KeyboardInterrupt):
                                        run_cpu_smoke_slope(
                                            root,
                                            asset_identity="fixtures/box",
                                            asset_version="a" * 64,
                                        )

            run_directory = next(root.location("runs").iterdir())
            case = read_json(run_directory / "cases" / "001-slope-25deg" / "case.json")
            manifest = read_json(run_directory / "manifest.json")
            status = read_json(run_directory / "status.json")
            self.assertEqual(case["status"], "interrupted")
            self.assertEqual(case["failure"]["code"], "cpu_smoke_slope_interrupted")
            self.assertEqual(manifest["exit_code"], 130)
            self.assertEqual(status["status"], "interrupted")
            self.assertFalse(manifest["authoritative"])
            self.assertFalse(status["authoritative"])


class SlopeCliAndControllerTests(unittest.TestCase):
    def test_remote_cli_exposes_fixed_cpu_non_authoritative_single_case_slope_help(self) -> None:
        from experiment_runner.cli import build_parser

        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        )
        slope_parser = subparsers.choices["smoke-slope"]
        help_text = slope_parser.format_help()
        self.assertIn("CPU", help_text)
        self.assertIn("non-authoritative", help_text)
        self.assertIn("single-case", help_text)
        self.assertIn("slope smoke", help_text)
        self.assertIn("25", help_text)

    def test_remote_cli_dispatches_smoke_slope_to_cpu_runner(self) -> None:
        from experiment_runner.cli import main

        expected = {
            "status": "succeeded",
            "authoritative": False,
            "slope_angle_degrees": 25.0,
        }
        with mock.patch(
            "experiment_runner.smoke.run_cpu_smoke_slope",
            return_value=expected,
        ) as run:
            with mock.patch("experiment_runner.cli._data_root", return_value=mock.sentinel.root):
                with mock.patch("builtins.print") as output:
                    return_code = main(
                        [
                            "smoke-slope",
                            "fixtures/blue-box",
                            "a" * 64,
                            "--data-root",
                            "D:/newton-data",
                        ]
                    )

        self.assertEqual(return_code, 0)
        rendered = "\n".join(" ".join(map(str, call.args)) for call in output.call_args_list)
        self.assertIn("CPU", rendered)
        self.assertIn("不是正式摩擦结论", rendered)
        run.assert_called_once_with(
            mock.sentinel.root,
            asset_identity="fixtures/blue-box",
            asset_version="a" * 64,
            git_commit=None,
        )

    def test_local_slope_launcher_uses_foreground_subprocess_data(self) -> None:
        from experiment_runner.controller import run_local_smoke

        completed = mock.Mock(returncode=0)
        with mock.patch(
            "experiment_runner.controller.subprocess.run",
            return_value=completed,
        ) as run:
            return_code = run_local_smoke(
                Path("D:/newton-data"),
                asset_identity="fixtures/blue box",
                asset_version="a" * 64,
                template="slope_friction",
            )

        self.assertEqual(return_code, 0)
        command = run.call_args.args[0]
        self.assertIn("smoke-slope", command)
        self.assertIn("fixtures/blue box", command)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertFalse(run.call_args.kwargs["check"])

    def test_local_menu_filters_slope_ready_assets_and_cancel_does_not_launch(self) -> None:
        from experiment_runner.controller import _run_slope_smoke_interactive

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            config = ControllerConfig(data_root=root.path)
            rows = [
                {
                    "identity": "drop-only",
                    "version": "a" * 64,
                    "readiness": "故意变化的显示文字",
                    "template_readiness": {
                        "drop": {"status": "ready", "checks": {"rigid_body_count": 1}},
                        "slope_friction": {
                            "status": "not_ready",
                            "checks": {"rigid_body_count": 1},
                        },
                    },
                },
                {
                    "identity": "slope-only",
                    "version": "b" * 64,
                    "readiness": "另一段显示文字",
                    "template_readiness": {
                        "drop": {"status": "not_ready", "checks": {"rigid_body_count": 1}},
                        "slope_friction": {
                            "status": "ready",
                            "checks": {"rigid_body_count": 1},
                        },
                    },
                },
                {
                    "identity": "both",
                    "version": "c" * 64,
                    "readiness": "不参与选择逻辑",
                    "template_readiness": {
                        "drop": {"status": "ready", "checks": {"rigid_body_count": 1}},
                        "slope_friction": {
                            "status": "ready",
                            "checks": {"rigid_body_count": 1},
                        },
                    },
                },
                {
                    "identity": "multi-body",
                    "version": "d" * 64,
                    "readiness": "通用坡度就绪",
                    "template_readiness": {
                        "drop": {"status": "ready", "checks": {"rigid_body_count": 2}},
                        "slope_friction": {
                            "status": "ready",
                            "checks": {"rigid_body_count": 2},
                        },
                    },
                },
            ]
            answers = iter(["1", ""])
            with mock.patch("experiment_runner.controller.list_assets_for_menu", return_value=rows):
                with mock.patch("builtins.input", side_effect=lambda _prompt="": next(answers)):
                    with mock.patch("experiment_runner.controller.run_local_smoke") as run:
                        with mock.patch("builtins.print") as output:
                            _run_slope_smoke_interactive(config)

            run.assert_not_called()
            rendered = "\n".join(" ".join(map(str, call.args)) for call in output.call_args_list)
            self.assertIn("slope-only", rendered)
            self.assertIn("both", rendered)
            self.assertNotIn("drop-only", rendered)
            self.assertNotIn("multi-body", rendered)
            self.assertIn("已取消", rendered)

    def test_menu_asset_rows_keep_structured_readiness_for_policy_decisions(self) -> None:
        from experiment_runner.controller import list_assets_for_menu

        readiness = {
            "drop": {"status": "ready", "reason_codes": [], "checks": {"rigid_body_count": 2}},
            "slope_friction": {
                "status": "ready",
                "reason_codes": [],
                "checks": {"rigid_body_count": 2},
            },
        }
        with mock.patch(
            "experiment_runner.controller.list_asset_versions",
            return_value=[{"identity": "fixtures/multi", "version": "a" * 64}],
        ):
            with mock.patch(
                "experiment_runner.controller.inspect_template_readiness",
                return_value=readiness,
            ):
                data_root = mock.Mock()
                data_root.resolve_managed.return_value = Path("asset.usda")
                rows = list_assets_for_menu(data_root)

        self.assertIs(rows[0]["template_readiness"], readiness)
        self.assertEqual(
            rows[0]["template_readiness"]["slope_friction"]["checks"]["rigid_body_count"],
            2,
        )

    def test_result_index_exposes_authoritative_flag_for_slope_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            run_directory = create_run_scaffold(
                root,
                run_id="slope-friction-20260813T120000Z-12345678",
                batch_id="batch-20260813T120000Z-12345678",
                template="slope_friction",
                asset={"identity": "fixtures/box", "version": "a" * 64},
                profile=get_profile("mujoco-cpu-wsl-smoke-v1").expanded(),
                git_commit="b" * 40,
            )
            case_directory = run_directory / "cases" / "001-slope-25deg"
            case_directory.mkdir()
            atomic_write_json(
                case_directory / "case.json",
                {
                    "schema_version": 1,
                    "case_id": "slope-25deg",
                    "label": "25° 坡度（CPU 冒烟）",
                    "status": "succeeded",
                    "condition": {"slope_angle_degrees": 25.0},
                    "duration_seconds": 2.0,
                    "authoritative": False,
                    "development_outcome": "inconclusive",
                    "slope_angle_degrees": 25.0,
                    "displacement_along_slope": 0.0,
                    "final_linear_velocity": [0.0, 0.0, 0.0],
                    "measurement_units": {
                        "displacement": "m",
                        "linear_velocity": "m/s",
                    },
                    "finite": True,
                },
            )
            update_run_status(
                run_directory,
                status="succeeded",
                phase="finished",
                progress="1/1 CPU slope smoke case complete",
                current_case=None,
                completed_cases=1,
                total_cases=1,
            )
            index = build_result_index(root)
            run = index["assets"][0]["runs"][0]
            self.assertFalse(run["authoritative"])
            self.assertFalse(run["cases"][0]["authoritative"])
            self.assertEqual(run["cases"][0]["development_outcome"], "inconclusive")
            self.assertEqual(run["cases"][0]["displacement_along_slope"], 0.0)
            self.assertEqual(run["cases"][0]["final_linear_velocity"], [0.0, 0.0, 0.0])

    def test_result_index_omits_non_finite_slope_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            run_directory = create_run_scaffold(
                root,
                run_id="slope-friction-20260813T120000Z-deadbeef",
                batch_id="batch-20260813T120000Z-deadbeef",
                template="slope_friction",
                asset={"identity": "fixtures/box", "version": "a" * 64},
                profile=get_profile("mujoco-cpu-wsl-smoke-v1").expanded(),
                git_commit="b" * 40,
            )
            case_directory = run_directory / "cases" / "001-slope-25deg"
            case_directory.mkdir()
            atomic_write_json(
                case_directory / "case.json",
                {
                    "schema_version": 1,
                    "case_id": "slope-25deg",
                    "status": "succeeded",
                    "condition": {},
                    "authoritative": False,
                    "slope_angle_degrees": math.nan,
                    "displacement_along_slope": math.inf,
                    "final_linear_velocity": [0.0, math.nan, 0.0],
                    "finite": False,
                },
            )
            index = build_result_index(root)
            public_case = index["assets"][0]["runs"][0]["cases"][0]

            self.assertNotIn("slope_angle_degrees", public_case)
            self.assertNotIn("displacement_along_slope", public_case)
            self.assertNotIn("final_linear_velocity", public_case)
            self.assertFalse(public_case["finite"])
            json.dumps(index, allow_nan=False)

    def test_web_consumer_prominently_labels_smoke_and_formats_slope_metrics_safely(self) -> None:
        script = (
            Path(__file__).resolve().parent.parent
            / "experiment_runner"
            / "web_static"
            / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn("非正式开发冒烟", script)
        self.assertIn("不能作为正式物理结论", script)
        self.assertIn("Number.isFinite", script)
        self.assertIn("slope_angle_degrees", script)
        self.assertIn("displacement_along_slope", script)
        self.assertIn("development_outcome", script)
        self.assertIn("final_linear_velocity", script)
        self.assertIn('["有限性", formatFiniteState(testCase.finite)]', script)
        self.assertIn("m/s", script)
        self.assertIn("未记录", script)


if __name__ == "__main__":
    unittest.main()
