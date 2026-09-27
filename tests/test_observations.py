from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiment_runner.asset_parameters import snapshot_physics_parameters
from experiment_runner.observations import case_observations, finite_number, slope_displacement_observation
from experiment_runner.profiles import get_profile
from experiment_runner.results import build_result_index, create_run_scaffold
from experiment_runner.storage import DataRoot, atomic_write_json, read_json
from tests.test_experiment_runner import READY_USDA


def measurement(value=0.2):
    return slope_displacement_observation(
        [1, 2, 3], [1 + value, 2, 3], [1, 0, 0],
        body_index=0, body_count=1, end_time_seconds=2.0,
    )


class ObservationTests(unittest.TestCase):
    def test_extreme_or_boolean_values_are_not_measurements(self):
        for value in (10 ** 1000, True, None, "0", math.nan, math.inf):
            self.assertFalse(finite_number(value))

    def test_signed_projection_ignores_cross_slope_and_normal_motion(self):
        down = [0.8, 0, -0.6]
        # 0.5 m downhill + 2 m cross-slope + 1 m along normal.
        result = slope_displacement_observation(
            [0, 0, 0], [1, 2, 0.5], down,
            body_index=0, body_count=1, end_time_seconds=3.0,
        )
        self.assertAlmostEqual(result["value"], 0.5)
        self.assertEqual(result["reference_point"], "body_frame_origin")
        self.assertEqual(result["sampling"], "initial_and_final_state")
        self.assertAlmostEqual(measurement(-0.25)["value"], -0.25)

    def test_rejects_undefined_measurement_inputs(self):
        arguments = dict(initial_position=[0, 0, 0], final_position=[1, 0, 0],
                         down_slope=[1, 0, 0], body_index=0, body_count=1, end_time_seconds=1.0)
        for changes in ({"body_count": 2}, {"body_count": True}, {"body_index": 1},
                        {"end_time_seconds": 0}, {"end_time_seconds": math.inf},
                        {"final_position": [math.nan, 0, 0]}, {"down_slope": [2, 0, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                slope_displacement_observation(**(arguments | changes))

    def test_zero_and_missing_are_distinct(self):
        document = {"status": "succeeded", "finite": True,
                    "observations": {"along_slope_displacement": measurement(0)}}
        result = case_observations(document, "slope_friction")["along_slope_displacement"]
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["value"], 0)
        result = case_observations({"status": "succeeded", "finite": True}, "slope_friction")
        self.assertEqual(result["along_slope_displacement"]["status"], "not_measured")
        self.assertIsNone(result["along_slope_displacement"]["value"])

    def test_bad_or_incomplete_state_suppresses_even_recorded_numbers(self):
        document = {"status": "succeeded", "finite": True,
                    "observations": {"along_slope_displacement": measurement()}}
        for changes, expected in (({"finite": False}, "invalid"), ({"finite": None}, "unverified"),
                                  ({"status": "running"}, "incomplete"),
                                  ({"status": "failed"}, "incomplete"), ({"status": "created"}, "not_started")):
            with self.subTest(changes=changes):
                result = case_observations(document | changes, "slope_friction")["along_slope_displacement"]
                self.assertEqual(result["status"], expected)
                self.assertIsNone(result["value"])

    def test_inconsistent_new_measurement_never_falls_back_to_legacy_number(self):
        for changes in ({"value": 5}, {"value": math.nan}, {"definition": "unknown"},
                        {"unit": "cm"}, {"start_time_seconds": 1}, {"body_count": 2}):
            document = {"status": "succeeded", "finite": True, "displacement_along_slope": 5,
                        "duration_seconds": 2, "observations": {"along_slope_displacement": measurement() | changes}}
            with self.subTest(changes=changes):
                result = case_observations(document, "slope_friction")["along_slope_displacement"]
                self.assertEqual(result["status"], "invalid")
                self.assertIsNone(result["value"])

    def test_legacy_values_do_not_gain_invented_measurement_definition(self):
        document = {"status": "succeeded", "finite": True,
                    "displacement_along_slope": 0.3, "duration_seconds": 2}
        original = copy.deepcopy(document)
        result = case_observations(document, "slope_friction")["along_slope_displacement"]
        self.assertEqual(result["status"], "legacy_record")
        self.assertEqual(result["value"], 0.3)
        self.assertNotIn("reference_point", result)
        self.assertEqual(document, original)

    def test_drop_does_not_infer_rebound_or_penetration_from_camera_bounds(self):
        result = case_observations({"status": "succeeded", "finite": True,
                                    "initial_bounds": {"minimum": [0, 0, 1]},
                                    "final_bounds": {"minimum": [0, 0, -0.1]}}, "drop")
        self.assertTrue(all(item["status"] == "not_measured" and item["value"] is None for item in result.values()))

    def test_record_slope_case_writes_definition_and_actual_physics_window(self):
        from experiment_runner.experiments.slope import record_slope_case

        def array(value):
            return SimpleNamespace(numpy=lambda: np.asarray(value))

        scene = SimpleNamespace(sim_time=0.0, tracked_body_index=0, initial_position=np.zeros(3),
                                asset_min_contact_priority=0, ramp_contact_priority=-1,
                                model=object(), state=SimpleNamespace(
                                    body_q=array([[0.2, 0, 0, 0, 0, 0, 1]]), body_qd=array([[0] * 6])))
        geometry = SimpleNamespace(down_slope=np.array([1, 0, 0]), angle_degrees=25, effective_length=0.1)

        def record(*args, **kwargs):
            scene.sim_time = 2.0
            return {"duration_seconds": 2.0, "video_duration_seconds": 2.5}

        with mock.patch("experiment_runner.experiments.slope.compute_asset_bounds"), mock.patch(
            "experiment_runner.experiments.slope.record_simulation_video", side_effect=record,
        ):
            result = record_slope_case(scene, profile=get_profile("mujoco-cpu-wsl-smoke-v1"),
                                       geometry=geometry, output_directory=Path("unused"), duration_seconds=2)
        observed = result["observations"]["along_slope_displacement"]
        self.assertEqual(observed["end_time_seconds"], 2.0)
        self.assertEqual(result["video_duration_seconds"], 2.5)
        self.assertEqual(result["displacement_along_slope"], observed["value"])


class ParameterSnapshotTests(unittest.TestCase):
    def test_version_snapshot_carries_parameters_without_changing_asset_content(self):
        from experiment_runner.assets import accept_asset, snapshot_asset_version

        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            package = root.location("inbox") / "fixture"
            package.mkdir()
            (package / "newton-mujoco.usda").write_text(READY_USDA, encoding="utf-8")
            report = accept_asset(root, "fixture", enforce_readonly=False)
            snapshot = snapshot_asset_version(root, "fixture", report["asset_version"])
            self.assertEqual(snapshot["version"], report["asset_version"])
            self.assertTrue(snapshot["physics_parameters"]["records"])
            entrypoint = snapshot["package_root"] / snapshot["entrypoint"]
            self.assertEqual(entrypoint.read_text(encoding="utf-8"), READY_USDA)
            self.assertNotIn(str(root.path), json.dumps(snapshot["physics_parameters"]))

    def snapshot(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "newton-mujoco.usda"
            path.write_text(source, encoding="utf-8")
            result = snapshot_physics_parameters(path)
            self.assertEqual(path.read_text(encoding="utf-8"), source)
            return result

    def test_authored_values_and_bound_material_are_recorded_without_claiming_origin(self):
        result = self.snapshot(READY_USDA)
        records = {item["field"]: item for item in result["records"]}
        self.assertEqual(records["mjc:solref"]["value"], [0.02, 1])
        friction = records["physics:dynamicFriction"]
        self.assertEqual(friction["value"], 0.5)
        self.assertEqual(friction["value_prim"], "/World/PhysicsMaterial")
        self.assertEqual(friction["collision_prim"], "/World/Box/Geometry")
        self.assertEqual(friction["source"], "authored_in_package")
        self.assertEqual(result["provenance"], "package_authorship_only")
        self.assertEqual(result["solver_effective_values"], "not_recorded")
        json.dumps(result, allow_nan=False)

    def test_unbound_material_and_unannotated_values_are_not_filled_with_defaults(self):
        source = READY_USDA.replace("rel material:binding:physics = </World/PhysicsMaterial>", "")
        records = {item["field"]: item for item in self.snapshot(source)["records"]}
        self.assertEqual(records["physics:dynamicFriction"]["status"], "material_unbound")
        self.assertIsNone(records["physics:dynamicFriction"]["value"])
        self.assertEqual(records["mjc:priority"]["status"], "not_authored")
        self.assertIsNone(records["mjc:priority"]["value"])

    def test_visual_only_shapes_are_excluded(self):
        result = self.snapshot(READY_USDA.replace('"PhysicsCollisionAPI", ', ""))
        self.assertEqual(result["records"], [])

    def test_animated_parameter_is_not_presented_as_a_constant(self):
        source = READY_USDA.replace(
            "uniform double[] mjc:solref = [0.02, 1.0]",
            "double[] mjc:solref.timeSamples = { 0: [0.02, 1.0], 1: [0.05, 1.0] }",
        )
        records = {item["field"]: item for item in self.snapshot(source)["records"]}
        self.assertEqual(records["mjc:solref"]["status"], "time_varying")
        self.assertIsNone(records["mjc:solref"]["value"])

    def test_multiple_shapes_keep_different_values_and_their_locations(self):
        from pxr import Usd
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "newton-mujoco.usda"
            path.write_text(READY_USDA, encoding="utf-8")
            stage = Usd.Stage.Open(str(path))
            original = stage.GetPrimAtPath("/World/Box/Geometry")
            from pxr import Sdf
            Sdf.CopySpec(stage.GetRootLayer(), original.GetPath(), stage.GetRootLayer(), "/World/Box/Other")
            stage.GetPrimAtPath("/World/Box/Other").GetAttribute("mjc:solref").Set([0.05, 1.0])
            stage.GetRootLayer().Save()
            records = snapshot_physics_parameters(path)["records"]
        values = [item["value"] for item in records if item["field"] == "mjc:solref"]
        self.assertEqual(values, [[0.02, 1.0], [0.05, 1.0]])


class ObservationIndexTests(unittest.TestCase):
    def test_index_uses_recorded_snapshot_and_keeps_run_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            parameters = {"schema_version": 1, "records": [{"field": "mjc:solref", "value": [0.02, 1]}]}
            run = create_run_scaffold(root, run_id="slope-test", batch_id="batch-test",
                                      template="slope_friction", asset={"identity": "fixture", "version": "a" * 64,
                                      "physics_parameters": parameters},
                                      profile=get_profile("mujoco-cpu-wsl-smoke-v1").expanded(), git_commit="b" * 40)
            case_dir = run / "cases" / "001-slope"
            case_dir.mkdir()
            case_path = case_dir / "case.json"
            atomic_write_json(case_path, {"case_id": "slope-25", "status": "succeeded", "finite": True,
                                         "observations": {"along_slope_displacement": measurement()},
                                         "duration_seconds": 2, "video_duration_seconds": 2.5})
            original = case_path.read_bytes()
            manifest_before = (run / "manifest.json").read_bytes()
            public = build_result_index(root)["assets"][0]["runs"][0]
            self.assertEqual(public["physics_parameters"], parameters)
            self.assertEqual(public["git_commit"], "b" * 40)
            self.assertEqual(public["cases"][0]["observations"]["along_slope_displacement"]["status"], "measured")
            self.assertEqual(public["cases"][0]["video_duration_seconds"], 2.5)
            self.assertEqual(case_path.read_bytes(), original)
            self.assertEqual((run / "manifest.json").read_bytes(), manifest_before)
            # No scan of the present asset store to fill a historical snapshot.
            manifest = read_json(run / "manifest.json")
            del manifest["asset"]["physics_parameters"]
            atomic_write_json(run / "manifest.json", manifest)
            public = build_result_index(root)["assets"][0]["runs"][0]
            self.assertNotIn("physics_parameters", public)


if __name__ == "__main__":
    unittest.main()
