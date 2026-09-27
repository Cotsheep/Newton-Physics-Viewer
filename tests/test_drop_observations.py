from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np

from experiment_runner.drop_metrics import DropAccumulator, public_drop_observations
from experiment_runner.experiments.drop_observer import CollisionClearance, DropObserver, UnsupportedDropGeometry


class Array:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def numpy(self):
        return self.value.copy()


def scene_from_xml(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mapping = np.arange(model.nbody).reshape(1, -1) - 1
    poses = np.column_stack((data.xpos[1:], data.xquat[1:, [1, 2, 3, 0]]))
    return SimpleNamespace(
        solver=SimpleNamespace(mj_model=model, mj_data=data, mjc_body_to_newton=Array(mapping)),
        state=SimpleNamespace(body_q=Array(poses)), sim_time=0.0,
    )


def primitive_scene(geometry, *, ground='type="plane" size="10 10 0.1"', body='pos="0 0 1"', extra=""):
    return scene_from_xml(f'''<mujoco><worldbody><geom {ground}/>
        <body {body}><freejoint/><geom {geometry}/>{extra}</body></worldbody></mujoco>''')


def series(clearances, dt=0.001):
    accumulator = DropAccumulator(dt)
    for index, value in enumerate(clearances):
        accumulator.add(index * dt, value)
    return accumulator.result((len(clearances) - 1) * dt)


class DropEventTests(unittest.TestCase):
    def test_first_complete_flight_and_whole_window_penetration(self):
        observations = series([1, 0, 0.2, 0.4, 0.1, 0, -0.03, 0.8, 0])
        height = observations["first_rebound_height"]
        self.assertEqual(height["status"], "measured")
        self.assertEqual(height["value"], 0.4)
        self.assertEqual(height["peak_time_seconds"], 0.003)
        self.assertEqual(observations["first_rebound_ratio"]["value"], 0.4)
        self.assertEqual(observations["maximum_ground_penetration"]["value"], 0.03)
        self.assertEqual(observations["maximum_ground_penetration"]["peak_time_seconds"], 0.006)

    def test_ratio_uses_actual_initial_clearance(self):
        observations = series([0.5, -0.001, 0.1, 0.2, 0])
        self.assertEqual(observations["first_rebound_ratio"]["value"], 0.4)

    def test_no_contact_no_liftoff_and_incomplete_flight_are_not_zero(self):
        for values, status in (([1, .5, .1], "not_observed"),
                               ([1, 0, -0.001, 0.000005], "no_liftoff"),
                               ([1, 0, .2, .1], "incomplete")):
            with self.subTest(status=status):
                observations = series(values)
                self.assertEqual(observations["first_rebound_height"]["status"], status)
                self.assertIsNone(observations["first_rebound_height"]["value"])
                self.assertIsNone(observations["first_rebound_ratio"]["value"])
                self.assertEqual(observations["maximum_ground_penetration"]["status"], "measured")

    def test_zero_penetration_is_a_real_measurement(self):
        observed = series([1, .5, .25])["maximum_ground_penetration"]
        self.assertEqual(observed["status"], "measured")
        self.assertEqual(observed["value"], 0)

    def test_invalid_or_missing_sample_invalidates_the_window(self):
        for time_value, clearance in ((.002, .2), (.001, math.nan)):
            accumulator = DropAccumulator(.001)
            accumulator.add(0, 1)
            with self.assertRaises(ValueError):
                accumulator.add(time_value, clearance)
            self.assertEqual(accumulator.result(.002)["maximum_ground_penetration"]["status"], "invalid")
        self.assertEqual(series([0, .1, 0])["first_rebound_height"]["status"], "invalid")

    def test_truncated_sampling_does_not_claim_complete_maximum(self):
        accumulator = DropAccumulator(.001)
        accumulator.add(0, 1)
        accumulator.add(.001, -.1)
        self.assertEqual(accumulator.result(.005)["maximum_ground_penetration"]["status"], "incomplete")


class CollisionGeometryTests(unittest.TestCase):
    def test_primitive_clearance_matches_mujoco_signed_distance(self):
        for kind, size in (("sphere", ".2"), ("box", ".1 .2 .3"),
                           ("ellipsoid", ".1 .2 .3"), ("capsule", ".1 .3"),
                           ("cylinder", ".1 .3")):
            for height in (1.0, .05):
                with self.subTest(kind=kind, height=height):
                    scene = primitive_scene(f'type="{kind}" size="{size}"', body=f'pos="0 0 {height}" euler="17 31 13"')
                    geometry = CollisionClearance(scene)
                    reference = mujoco.mj_geomDistance(scene.solver.mj_model, scene.solver.mj_data, 0, 1, 10.0, None)
                    self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), reference, places=6)

    def test_compiled_reduced_mesh_hull_matches_solver_distance(self):
        rng = np.random.default_rng(25)
        points = rng.normal(size=(40, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        vertices = " ".join(str(value) for value in points.ravel())
        scene = scene_from_xml(f'''<mujoco><asset><mesh name="hull" maxhullvert="12" vertex="{vertices}"/></asset>
            <worldbody><geom type="plane" size="10 10 1"/>
            <body pos="0 0 2" euler="17 31 13"><freejoint/><geom type="mesh" mesh="hull"/></body>
            </worldbody></mujoco>''')
        geometry = CollisionClearance(scene)
        reference = mujoco.mj_geomDistance(scene.solver.mj_model, scene.solver.mj_data, 0, 1, 10.0, None)
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), reference, places=6)

    def test_rotated_primitives_use_surface_support_not_rotated_box_corners(self):
        angle = math.pi / 4
        for geometry, extent in (
            ('type="sphere" size="0.2"', .2),
            ('type="box" size="0.1 0.2 0.3"', (.1 + .3) / math.sqrt(2)),
            ('type="ellipsoid" size="0.1 0.2 0.3"', math.sqrt((.1**2 + .3**2) / 2)),
            ('type="capsule" size="0.1 0.3"', .1 + .3 / math.sqrt(2)),
            ('type="cylinder" size="0.1 0.3"', (.1 + .3) / math.sqrt(2)),
        ):
            with self.subTest(geometry=geometry):
                scene = primitive_scene(geometry, body=f'pos="0 0 1" quat="{math.cos(angle/2)} 0 {math.sin(angle/2)} 0"')
                clearance = CollisionClearance(scene).measure(scene.state.body_q.numpy())
                self.assertAlmostEqual(clearance, 1 - extent, places=12)

    def test_completed_pose_is_used_instead_of_stale_solver_geometry(self):
        scene = primitive_scene('type="sphere" size="0.1"')
        geometry = CollisionClearance(scene)
        scene.state.body_q.value[0, 2] = 0.05
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), -0.05)
        self.assertAlmostEqual(scene.solver.mj_data.geom_xpos[1, 2], 1.0)

    def test_visual_geometry_cannot_create_false_penetration(self):
        scene = primitive_scene('type="sphere" size="0.1"', extra='<geom type="box" size="5 5 5" contype="0" conaffinity="0" mass="0"/>')
        geometry = CollisionClearance(scene)
        self.assertEqual(len(geometry.shapes), 1)
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), .9)

    def test_multiple_bodies_require_whole_asset_liftoff(self):
        scene = scene_from_xml('''<mujoco><worldbody><geom type="plane" size="10 10 1"/>
            <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body>
            <body pos="1 0 .1"><freejoint/><geom type="sphere" size=".1"/></body>
            </worldbody></mujoco>''')
        geometry = CollisionClearance(scene)
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), 0)
        scene.state.body_q.value[0, 2] = 2
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), 0)

    def test_mesh_compiler_rotation_and_scaling_are_included(self):
        scene = scene_from_xml('''<mujoco><asset><mesh name="tetra" scale="2 3 4"
            vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset><worldbody>
            <geom type="plane" size="10 10 1"/><body pos="0 0 5" euler="0 30 0"><freejoint/>
            <geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>''')
        # Rotating the scaled (2,0,0) vertex by +30 degrees gives z=-1.
        geometry = CollisionClearance(scene)
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), 4.0, places=6)
        scene.state.body_q.value[0, 2] -= 4.2
        self.assertAlmostEqual(geometry.measure(scene.state.body_q.numpy()), -.2, places=6)

    def test_unsupported_scene_is_explicit_not_a_bounds_fallback(self):
        scene = primitive_scene('type="sphere" size=".1"', ground='type="box" size="10 10 .1"')
        with self.assertRaises(UnsupportedDropGeometry):
            CollisionClearance(scene)
        observer = DropObserver(scene, .001)
        self.assertTrue(observer.reason)
        self.assertEqual(observer.result(1)["observations"]["maximum_ground_penetration"]["status"], "unsupported")

    def test_contact_mask_exclusion_and_tilted_ground_are_not_silently_measured(self):
        for scene in (
            primitive_scene('type="sphere" size=".1" contype="2" conaffinity="2"'),
            primitive_scene('type="sphere" size=".1"', ground='type="plane" size="10 10 1" euler="0 15 0"'),
        ):
            with self.assertRaises(UnsupportedDropGeometry):
                CollisionClearance(scene)

    def test_newton_compiled_model_adapter_on_existing_fixture(self):
        from experiment_runner.experiments.drop import create_drop_scene, measure_drop_geometry
        from experiment_runner.profiles import get_profile
        profile = get_profile("mujoco-cpu-wsl-smoke-v1")
        asset = Path(__file__).parent / "fixtures/smoke_asset/newton-mujoco.usda"
        geometry = measure_drop_geometry(asset, profile=profile)
        scene = create_drop_scene(asset, profile=profile, clearance=geometry.clearance, measured_bounds=geometry.initial_bounds)
        self.assertIsNone(scene.drop_observer.reason)
        self.assertAlmostEqual(scene.drop_observer.accumulator.initial, .1, places=6)
        self.assertEqual(scene.completed_physics_steps, 0)


class DropProtocolTests(unittest.TestCase):
    def document(self):
        return {"status": "succeeded", "finite": True, "duration_seconds": .004,
                "physics_steps": 4, "observations": series([.5, 0, .1, .2, 0])}

    def test_valid_complete_and_incomplete_flight_round_trip(self):
        document = self.document()
        result = public_drop_observations(document)
        self.assertEqual(result["first_rebound_height"]["value"], .2)
        for values in ([.5, .4, .3, .2, .1], [.5, 0, -.01, -.001, 0], [.5, 0, .1, .2, .1]):
            document["observations"] = series(values)
            self.assertEqual(public_drop_observations(document), document["observations"])

    def test_bad_units_ratio_events_and_sampling_are_rejected(self):
        for metric, field, value in (
            ("first_rebound_ratio", "value", .99), ("first_rebound_height", "unit", "cm"),
            ("first_rebound_height", "peak_time_seconds", .8), ("maximum_ground_penetration", "value", -1),
            ("first_rebound_height", "sample_count", 100),
        ):
            with self.subTest(field=field):
                document = self.document()
                document["observations"][metric][field] = value
                self.assertTrue(all(item["status"] == "invalid" for item in public_drop_observations(document).values()))

    def test_failed_run_or_non_finite_state_suppresses_measurements(self):
        for changes, expected in (({"status": "failed"}, "incomplete"), ({"finite": False}, "invalid")):
            result = public_drop_observations(self.document() | changes)
            self.assertTrue(all(item["status"] == expected and item["value"] is None for item in result.values()))

    def test_legacy_bounds_never_become_a_measurement(self):
        self.assertTrue(all(item["status"] == "not_measured" for item in public_drop_observations(
            {"status": "succeeded", "finite": True, "final_bounds": {"minimum": [0, 0, -.1]}}).values()))

    def test_gpu_step_hook_runs_for_every_step_without_changing_step_count(self):
        from tests.test_gpu_smoke import RunnerHarness
        with RunnerHarness() as harness:
            observer = mock.Mock()
            observer.result.return_value = {"observations": {}}
            harness.scene.drop_observer = observer
            harness.run()
            self.assertEqual(observer.sample.call_count, 1000)
            observer.result.assert_called_once_with(1.0)
            self.assertEqual(harness.scene.solver.step.call_count, 1000)

    def test_cpu_recording_includes_observation_summary(self):
        from experiment_runner.experiments.drop import record_drop_case
        from experiment_runner.profiles import get_profile
        observer = mock.Mock()
        observer.result.return_value = {"observations": series([.5, 0, .1, 0])}
        scene = SimpleNamespace(drop_observer=observer, model=object(), state=SimpleNamespace(
            body_q=Array([[0, 0, 1, 0, 0, 0, 1]]), body_qd=Array([[0] * 6])))
        from asset_viewer.camera import AssetBounds
        bounds = AssetBounds(np.zeros(3), np.ones(3))
        with mock.patch("experiment_runner.experiments.drop.compute_asset_bounds", return_value=bounds), mock.patch(
            "experiment_runner.experiments.drop.record_simulation_video", return_value={"duration_seconds": .003}
        ):
            result = record_drop_case(scene, profile=get_profile("mujoco-cpu-wsl-smoke-v1"),
                                      output_directory=Path("unused"), duration_seconds=.003)
        self.assertEqual(result["observations"]["first_rebound_height"]["value"], .1)
        observer.result.assert_called_once_with(.003)


if __name__ == "__main__":
    unittest.main()
