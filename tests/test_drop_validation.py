from dataclasses import replace
from pathlib import Path
import os
import unittest
from unittest import mock

import numpy as np

from experiment_runner.drop_validation import DROP_VALIDATION_PROFILE, drop_validation_case
from experiment_runner.profiles import get_profile, describe_profiles


class DropValidationSceneTests(unittest.TestCase):
    def test_collision_clearance_corrects_an_inexact_planning_box_without_stepping(self):
        from asset_viewer.camera import AssetBounds
        from experiment_runner.experiments.drop import create_drop_scene, measure_drop_geometry
        asset = Path(__file__).parent / 'fixtures/smoke_asset/newton-mujoco.usda'
        profile = get_profile('mujoco-cpu-wsl-smoke-v1')
        geometry = measure_drop_geometry(asset, profile=profile)
        bounds = geometry.initial_bounds
        minimum = bounds.minimum.copy()
        minimum[2] -= .05
        scene = create_drop_scene(asset, profile=profile, clearance=geometry.clearance,
            measured_bounds=AssetBounds(minimum, bounds.maximum), neutral_reference=True)
        actual = scene.drop_observer.geometry.measure(scene.state.body_q.numpy())
        self.assertAlmostEqual(actual, geometry.clearance, places=6)
        self.assertEqual(scene.completed_physics_steps, 0)

    def test_four_releases_preserve_asset_and_use_lower_priority_ground(self):
        import mujoco
        from experiment_runner.experiments.drop import create_drop_scene, measure_drop_geometry
        from experiment_runner.experiments.drop_reference import attach_drop_reference
        asset = Path(__file__).parent / 'fixtures/smoke_asset/newton-mujoco.usda'
        before = asset.read_bytes()
        profile = get_profile('mujoco-cpu-wsl-smoke-v1')
        initial_poses = []
        cameras = []
        for height in ('low', 'medium', 'high', 'fixed-1m'):
            spec = drop_validation_case(height)
            geometry = measure_drop_geometry(asset, profile=profile, clearance_scale=spec.clearance_scale,
                                             fixed_clearance_m=spec.fixed_clearance_m)
            scene = create_drop_scene(asset, profile=profile, clearance=geometry.clearance,
                                      measured_bounds=geometry.initial_bounds, neutral_reference=True)
            conditions = attach_drop_reference(scene, geometry)
            expected = spec.fixed_clearance_m if spec.fixed_clearance_m is not None else spec.clearance_scale * geometry.effective_length
            self.assertAlmostEqual(conditions['actual_initial_clearance_m'], expected, places=6)
            self.assertGreaterEqual(conditions['scale_reference']['ruler_top_m'], 1.0 + geometry.initial_bounds.extents[2])
            self.assertLess(conditions['reference_ground_priority'], conditions['asset_min_contact_priority'])
            self.assertEqual(scene.completed_physics_steps, 0)
            np.testing.assert_array_equal(scene.state.body_qd.numpy(), 0)
            initial_poses.append(scene.state.body_q.numpy())
            cameras.append(scene.scale_reference.camera_bounds)
            # Move a separate MuJoCo data object into contact to verify priority
            # actually selects the asset's values, rather than merely recording it.
            m, d = scene.solver.mj_model, mujoco.MjData(scene.solver.mj_model)
            m.geom_solref[0] = [.2, .1]  # Contrasting ground value must not win.
            d.qpos[:] = scene.solver.mj_data.qpos
            d.qpos[2] -= geometry.clearance + .001
            mujoco.mj_forward(m, d)
            self.assertGreater(d.ncon, 0)
            for c in d.contact:
                other = next(int(i) for i in c.geom if i != 0)
                np.testing.assert_allclose(c.solref, m.geom_solref[other])
                np.testing.assert_allclose(c.solimp, m.geom_solimp[other])
        for poses in initial_poses[1:]:
            np.testing.assert_allclose(poses[:, [0,1,3,4,5,6]], initial_poses[0][:, [0,1,3,4,5,6]])
        for bounds in cameras[1:]:
            np.testing.assert_allclose(bounds.minimum, cameras[0].minimum)
            np.testing.assert_allclose(bounds.maximum, cameras[0].maximum)
        self.assertEqual(asset.read_bytes(), before)

    def test_fixed_clearance_is_independent_of_asset_size_and_clamping(self):
        from asset_viewer.camera import AssetBounds
        from experiment_runner.experiments.drop import measure_drop_geometry
        from experiment_runner.experiments.drop_reference import scale_reference
        profile = get_profile('mujoco-cpu-wsl-smoke-v1')
        for size in (0.02, 0.3, 0.495, 2.0):
            bounds = AssetBounds(np.zeros(3), np.full(3, size))
            with self.subTest(size=size), mock.patch('experiment_runner.experiments.drop.build_model',
                    return_value=(mock.Mock(), mock.Mock(), None)), mock.patch(
                    'experiment_runner.experiments.drop.compute_asset_bounds', return_value=bounds):
                geometry = measure_drop_geometry(Path('unused.usda'), profile=profile,
                                                 clearance_scale=None, fixed_clearance_m=1.0)
                self.assertEqual(geometry.clearance, 1.0)
                self.assertEqual(geometry.characteristic_length, size)
                self.assertEqual(geometry.effective_length, min(max(size, 0.1), 1.0))
                placed = AssetBounds(bounds.minimum + [0, 0, 1], bounds.maximum + [0, 0, 1])
                reference = scale_reference(placed, geometry.effective_length)
                self.assertGreaterEqual(reference.camera_bounds.maximum[2], placed.maximum[2])
                self.assertTrue(np.any(np.isclose(reference.starts[:, 2], 1.0)
                                       & np.isclose(reference.ends[:, 2], 1.0)))

    def test_invalid_or_ambiguous_clearance_is_rejected_before_runtime_setup(self):
        from experiment_runner.experiments.drop import measure_drop_geometry
        profile = get_profile('mujoco-cpu-wsl-smoke-v1')
        invalid = [dict(clearance_scale=1.0, fixed_clearance_m=1.0), dict(clearance_scale=None),
                   *[dict(clearance_scale=None, fixed_clearance_m=value)
                     for value in (0, -1, float('nan'), float('inf'))]]
        with mock.patch('experiment_runner.experiments.drop.configure_warp_for_profile') as configure:
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    measure_drop_geometry(Path('unused.usda'), profile=profile, **arguments)
            configure.assert_not_called()

    def test_priority_underflow_is_rejected_without_assignment(self):
        import newton
        from types import SimpleNamespace
        from experiment_runner.experiments.drop_reference import set_neutral_ground_priority
        def arr(values):
            return SimpleNamespace(numpy=lambda: np.array(values))
        priority = mock.Mock()
        priority.numpy.return_value = np.array([np.iinfo(np.int32).min, 0], dtype=np.int32)
        model = SimpleNamespace(shape_body=arr([0,-1]), shape_flags=arr([int(newton.ShapeFlags.COLLIDE_SHAPES)]*2),
                                shape_type=arr([newton.GeoType.BOX,newton.GeoType.PLANE]),
                                mujoco=SimpleNamespace(geom_priority=priority))
        with self.assertRaisesRegex(ValueError, 'safely'):
            set_neutral_ground_priority(model)
        priority.assign.assert_not_called()


class DropValidationBoundaryTests(unittest.TestCase):
    def test_height_required_and_no_arbitrary_duration_or_profile(self):
        from experiment_runner.cli import build_parser
        parser = build_parser()
        for extra in ([], ['--height','all'], ['--height','low','--duration','60'],
                      ['--height','low','--profile','mujoco-native-dt1ms-v1']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parser.parse_args(['validate-drop-gpu','a','b',*extra])
        parsed = parser.parse_args(['validate-drop-gpu','a','b','--height','high'])
        self.assertEqual(parsed.height, 'high')
        parsed = parser.parse_args(['validate-drop-gpu','a','b','--height','fixed-1m'])
        self.assertEqual(parsed.height, 'fixed-1m')
        self.assertFalse(describe_profiles()['mujoco-native-dt1ms-v1']['availability']['runnable'])

    def test_fixed_development_profile_only(self):
        from experiment_runner.gpu_safety import issue_gpu_execution_permit, require_gpu_execution_permit, GpuSmokeSafetyError
        from tests.test_gpu_smoke import TRIAL_ENVIRONMENT
        profile = get_profile(DROP_VALIDATION_PROFILE)
        self.assertEqual(profile.case_duration_seconds, 10)
        self.assertEqual(profile.wall_time_limit_seconds, 900)
        self.assertEqual((profile.physics_dt, profile.iterations), (0.001, 10))
        self.assertEqual((profile.video_width, profile.video_height, profile.video_fps), (1280, 720, 50))
        self.assertEqual(profile.recording_mode, "required")
        self.assertFalse(profile.authoritative)
        with mock.patch.dict(os.environ, TRIAL_ENVIRONMENT, clear=True):
            permit = issue_gpu_execution_permit()
            require_gpu_execution_permit(permit, profile)
            with self.assertRaises(GpuSmokeSafetyError):
                require_gpu_execution_permit(permit, replace(profile, case_duration_seconds=20))


class DropValidationRunnerTests(unittest.TestCase):
    def test_all_levels_reach_scene_with_correct_scale_and_retain_case_results(self):
        from tests.test_gpu_smoke import RunnerHarness
        from experiment_runner.storage import read_json
        for height, scale in [('low', .5), ('medium', 1), ('high', 2), ('fixed-1m', None)]:
            with self.subTest(height=height), RunnerHarness() as harness:
                if height == 'fixed-1m':
                    harness.geometry.clearance = 1.0
                measure = harness.stack.enter_context(mock.patch(
                    'experiment_runner.experiments.drop.measure_drop_geometry', return_value=harness.geometry))
                harness.stack.enter_context(mock.patch('experiment_runner.experiments.drop_reference.attach_drop_reference',
                    return_value={'reference_policy':'asset_priority_over_ground_v1'}))
                def record(scene, **kwargs):
                    self.assertEqual(kwargs['duration_seconds'], 10)
                    kwargs['on_physics_started']()
                    scene.completed_physics_steps = 10000
                    return {'recording':{'status':'succeeded','files':[]}, 'physics_steps':10000,
                            'duration_seconds':10, 'finite':True}
                harness.stack.enter_context(mock.patch('experiment_runner.experiments.gpu_recording.record_gpu_drop_case', side_effect=record))
                result = harness.run(validation_height=height)
                self.assertEqual(measure.call_args.kwargs['clearance_scale'], scale)
                self.assertEqual(measure.call_args.kwargs.get('fixed_clearance_m'), 1.0 if height == 'fixed-1m' else None)
                self.assertTrue(harness.create_scene.call_args.kwargs['neutral_reference'])
                run = harness.root.location('runs') / result['run_id']
                case = read_json(run / f'cases/001-{height}-validation/case.json')
                self.assertEqual(case['condition']['height_level'], height)
                if height == 'fixed-1m':
                    self.assertEqual(harness.create_scene.call_args.kwargs['clearance'], 1.0)
                    self.assertEqual(case['condition']['case_scope'], 'fixed_1m_drop_validation_v1')
                    self.assertIsNone(case['condition']['clearance_scale'])
                    self.assertEqual(case['condition']['fixed_clearance_m'], 1.0)
                    self.assertEqual(case['condition']['clearance_m'], 1.0)
                    batch = read_json(harness.root.location('batches') / result['batch_id'] / 'batch.json')
                    self.assertEqual(batch['request_summary']['drop_fixed_clearance_m'], 1.0)
                self.assertEqual(case['duration_seconds'], 10)
                self.assertEqual(case['execution']['completed_physics_steps'], 10000)
                self.assertEqual(case['status'], 'succeeded')
                self.assertFalse(case['authoritative'])
                self.assertEqual(len(list((run/'cases').iterdir())), 1)

    def test_recording_really_counts_10000_steps_and_525_frames_with_fake_gpu(self):
        import imageio_ffmpeg
        from tests.test_gpu_smoke import RunnerHarness, GpuVideoSmokeTests
        from experiment_runner.storage import read_json
        with RunnerHarness() as harness:
            viewer, frame = GpuVideoSmokeTests().recording_mocks(harness)
            frame.return_value = np.full((720, 1280, 3), 80, dtype=np.uint8)
            harness.stack.enter_context(mock.patch('experiment_runner.experiments.drop_reference.attach_drop_reference', return_value={}))
            result = harness.run(validation_height='fixed-1m')
            run = harness.root.location('runs') / result['run_id']
            case = read_json(run/'cases/001-fixed-1m-validation/case.json')
            frames, duration = imageio_ffmpeg.count_frames_and_secs(str(run/'cases/001-fixed-1m-validation/video.mp4'))
            self.assertEqual((frames, duration), (525, 10.5))
            self.assertEqual(case['physics_steps'], 10000)
            self.assertEqual(harness.scene.solver.step.call_count, 10000)
            self.assertEqual(frame.call_count, 501)
            viewer.close.assert_called_once()

    def test_invalid_level_and_missing_trial_never_discover_gpu(self):
        from tests.test_gpu_smoke import RunnerHarness
        from experiment_runner.gpu_safety import GpuSmokeSafetyError
        with RunnerHarness() as harness:
            with self.assertRaises(ValueError):
                harness.run(validation_height='all')
            with self.assertRaises(GpuSmokeSafetyError):
                harness.run({}, validation_height='low')
            self.assertEqual(harness.runtime.discovery_calls, 0)
            self.assertEqual(list(harness.root.location('runs').iterdir()), [])

    def test_failed_case_keeps_actual_progress_and_never_starts_next_height(self):
        from tests.test_gpu_smoke import RunnerHarness, GpuVideoSmokeTests
        from experiment_runner.storage import read_json
        with RunnerHarness() as harness:
            viewer, frame = GpuVideoSmokeTests().recording_mocks(harness)
            frame.return_value = np.full((720, 1280, 3), 80, dtype=np.uint8)
            harness.stack.enter_context(mock.patch('experiment_runner.experiments.drop_reference.attach_drop_reference', return_value={}))
            calls = 0
            def fail(*args):
                nonlocal calls
                calls += 1
                if calls == 128:
                    raise RuntimeError('test injected failure')
            harness.scene.solver.step.side_effect = fail
            with self.assertRaises(Exception):
                harness.run(validation_height='low')
            runs = list(harness.root.location('runs').iterdir())
            self.assertEqual(len(runs), 1)
            case = read_json(runs[0]/'cases/001-low-validation/case.json')
            self.assertEqual(case['execution']['completed_physics_steps'], 127)
            self.assertEqual(case['status'], 'failed')
            viewer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
