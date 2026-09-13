from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiment_runner.gpu_safety import (
    GpuSmokeSafetyError,
    require_determined_trial_allocation,
    require_single_visible_cuda_gpu,
)
from experiment_runner.profiles import get_profile


ENVIRONMENT = {
    "DET_EXPERIMENT_ID": "1", "DET_TRIAL_ID": "2", "DET_TASK_ID": "task",
    "DET_ALLOCATION_ID": "allocation", "DET_TASK_TYPE": "TRIAL",
    "DET_SLOT_IDS": "[0]", "CUDA_VISIBLE_DEVICES": "0",
    "NVIDIA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc",
}


class SafetyRegressionTests(unittest.TestCase):
    def test_result_server_accepts_only_literal_ipv4_loopback(self):
        from experiment_runner.web import create_result_server
        root = mock.Mock()
        for host in ("localhost", "::1", "0.0.0.0", "192.0.2.1"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "127.0.0.1"):
                create_result_server(root, host=host)
        root.require_initialized.assert_not_called()

    def test_formal_profile_rejected_even_with_gate_issued_permit(self):
        from experiment_runner.experiments import recording
        from experiment_runner.gpu_safety import issue_gpu_execution_permit
        permit = issue_gpu_execution_permit(ENVIRONMENT)
        with mock.patch.object(recording.wp, "set_device") as choose:
            with self.assertRaisesRegex(GpuSmokeSafetyError, "fixed GPU integration"):
                recording.configure_warp_for_profile(get_profile("mujoco-native-dt1ms-v1"), gpu_permit=permit)
        choose.assert_not_called()

    def test_direct_geometry_model_and_recording_calls_reject_before_any_work(self):
        from experiment_runner.experiments import drop, recording
        profile = get_profile("mujoco-warp-cuda-dt1ms-integration-smoke-v1")
        with mock.patch.object(recording.wp, "set_device") as choose, mock.patch.object(drop, "build_model") as build:
            actions = (
                lambda: drop.measure_drop_geometry(Path("unused.usda"), profile=profile),
                lambda: drop.create_drop_scene(Path("unused.usda"), profile=profile, clearance=1, measured_bounds=None),
                lambda: drop.render_asset_cover(Path("unused.usda"), profile=profile, output_path=Path("unused.jpg")),
                lambda: drop.record_drop_case(None, profile=profile, output_directory=Path("unused"), duration_seconds=1),
                lambda: recording._headless_viewer(profile),
            )
            for action in actions:
                with self.assertRaises(GpuSmokeSafetyError):
                    action()
            choose.assert_not_called()
            build.assert_not_called()

    def test_forged_permit_is_rejected_before_runtime_enumeration(self):
        from experiment_runner.gpu_safety import GpuExecutionPermit, require_single_warp_visible_gpu
        for permit in (None, True, object(), GpuExecutionPermit()):
            runtime = mock.Mock()
            with self.assertRaises(GpuSmokeSafetyError):
                require_single_warp_visible_gpu(runtime, ENVIRONMENT, permit=permit)
            runtime.visible_device_count.assert_not_called()

    def test_slot_requires_one_nonnegative_integer(self):
        for slots in ('[-1]', '["0"]', '["unknown"]', '[true]', '[0.0]', '[0,1]'):
            with self.subTest(slots=slots), self.assertRaises(GpuSmokeSafetyError):
                require_determined_trial_allocation({**ENVIRONMENT, "DET_SLOT_IDS": slots})

    def test_runtime_uuid_mismatch_and_malformed_are_rejected(self):
        for uuid in ("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "malformed"):
            runtime = mock.Mock()
            runtime.visible_device_count.return_value = 1
            runtime.initialize_logical_device.return_value = SimpleNamespace(
                alias="cuda:0", ordinal=0, is_cuda=True, uuid=uuid, name="fake"
            )
            with self.subTest(uuid=uuid), self.assertRaises(GpuSmokeSafetyError):
                require_single_visible_cuda_gpu(runtime, ENVIRONMENT)

    def test_shared_gpu_device_selection_requires_permit(self):
        from experiment_runner.experiments import recording
        for name in ("mujoco-native-dt1ms-v1", "mujoco-warp-cuda-dt1ms-integration-smoke-v1"):
            with self.subTest(profile=name), mock.patch.object(recording.wp, "set_device") as choose:
                with self.assertRaises(GpuSmokeSafetyError):
                    recording.configure_warp_for_profile(get_profile(name))
                choose.assert_not_called()

    def test_each_bound_material_must_author_friction(self):
        from pxr import Sdf, Usd
        from experiment_runner.assets import inspect_template_readiness
        fixture = Path(__file__).parent / "fixtures/smoke_asset/newton-mujoco.usda"
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(fixture.read_text(encoding="utf-8"))
        layer = stage.GetRootLayer()
        Sdf.CopySpec(layer, "/World/Box/Geometry", layer, "/World/Box/Geometry2")
        material = stage.DefinePrim("/World/MissingFriction", "Material")
        material.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["PhysicsMaterialAPI"]))
        stage.GetPrimAtPath("/World/Box/Geometry2").GetRelationship(
            "material:binding:physics"
        ).SetTargets(["/World/MissingFriction"])
        with mock.patch.object(Usd.Stage, "Open", return_value=stage):
            readiness = inspect_template_readiness(Path("in-memory.usda"))
        self.assertEqual(readiness["slope_friction"]["status"], "not_ready")

    def test_temporary_viewer_selection_rejects_data_and_source(self):
        from experiment_runner import controller
        from experiment_runner.config import ControllerConfig
        fixture = Path(__file__).parent / "fixtures/minimal_rigid.usda"
        with mock.patch("builtins.input", return_value=str(fixture)), mock.patch.object(
            controller, "launch_viewer"
        ) as launch:
            with self.assertRaises(ValueError):
                controller._open_viewer_interactive(ControllerConfig(data_root=fixture.parent))
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
