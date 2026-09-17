from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiment_runner.assets import ASSET_ENTRYPOINT
from experiment_runner.gpu_safety import (
    GpuSmokeSafetyError,
    require_determined_trial_allocation,
    require_scene_on_logical_gpu,
    require_single_visible_cuda_gpu,
)
from experiment_runner.profiles import get_profile
from experiment_runner.storage import DataRoot, read_json


GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"
TRIAL_ENVIRONMENT = {
    "DET_EXPERIMENT_ID": "101",
    "DET_TRIAL_ID": "202",
    "DET_TASK_ID": "task-303",
    "DET_ALLOCATION_ID": "alloc-404",
    "DET_SLOT_IDS": '[0]',
    "DET_TASK_TYPE": "TRIAL",
    "DET_RESOURCES_ID": "resources-505",
    "DET_TRIAL_RUN_ID": "1",
    "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
    "CUDA_VISIBLE_DEVICES": "0",
}


class FakeCudaRuntime:
    def __init__(self, count: int = 1, *, initialize_error: Exception | None = None) -> None:
        self.count = count
        self.initialize_error = initialize_error
        self.discovery_calls = 0
        self.initialized_devices: list[str] = []
        self.device = SimpleNamespace(
            alias="cuda:0",
            ordinal=0,
            is_cuda=True,
            name="Fake GPU",
            uuid=GPU_UUID,
        )

    def visible_device_count(self) -> int:
        self.discovery_calls += 1
        return self.count

    def initialize_logical_device(self, logical_device: str):
        self.initialized_devices.append(logical_device)
        if self.initialize_error is not None:
            raise self.initialize_error
        return self.device

    def driver_version(self):
        return (575, 57)

    def toolkit_version(self):
        return (12, 9)

    def nvidia_driver_version(self):
        return "575.57.08"


class GpuSafetyTests(unittest.TestCase):
    def test_missing_all_determined_identifiers_is_rejected_before_cuda(self) -> None:
        runtime = FakeCudaRuntime()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GpuSmokeSafetyError, "DET_EXPERIMENT_ID"):
                require_single_visible_cuda_gpu(runtime)
        self.assertEqual(runtime.discovery_calls, 0)
        self.assertEqual(runtime.initialized_devices, [])

    def test_each_required_determined_trial_field_is_fail_closed(self) -> None:
        required = (
            "DET_EXPERIMENT_ID",
            "DET_TRIAL_ID",
            "DET_TASK_ID",
            "DET_ALLOCATION_ID",
            "DET_SLOT_IDS",
            "DET_TASK_TYPE",
        )
        for missing in required:
            with self.subTest(missing=missing):
                environment = dict(TRIAL_ENVIRONMENT)
                environment.pop(missing)
                runtime = FakeCudaRuntime()
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(GpuSmokeSafetyError):
                        require_single_visible_cuda_gpu(runtime)
                self.assertEqual(runtime.discovery_calls, 0)
                self.assertEqual(runtime.initialized_devices, [])

    def test_unknown_core_determined_identifier_is_rejected(self) -> None:
        for key in ("DET_EXPERIMENT_ID", "DET_TRIAL_ID", "DET_TASK_ID"):
            with self.subTest(key=key):
                environment = dict(TRIAL_ENVIRONMENT)
                environment[key] = "unknown"
                with self.assertRaisesRegex(GpuSmokeSafetyError, key):
                    require_determined_trial_allocation(environment)

    def test_determined_slot_list_must_contain_exactly_one_slot(self) -> None:
        for value in ("not-json", "[]", '["0", "1"]', "null"):
            with self.subTest(value=value):
                environment = dict(TRIAL_ENVIRONMENT)
                environment["DET_SLOT_IDS"] = value
                with self.assertRaisesRegex(GpuSmokeSafetyError, "DET_SLOT_IDS"):
                    require_determined_trial_allocation(environment)

    def test_invalid_nvidia_visibility_sentinels_are_rejected_before_cuda(self) -> None:
        for value in (None, "", "all", "none", "void"):
            with self.subTest(value=value):
                environment = dict(TRIAL_ENVIRONMENT)
                if value is None:
                    environment.pop("NVIDIA_VISIBLE_DEVICES")
                else:
                    environment["NVIDIA_VISIBLE_DEVICES"] = value
                runtime = FakeCudaRuntime()
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(
                        GpuSmokeSafetyError, "NVIDIA_VISIBLE_DEVICES"
                    ):
                        require_single_visible_cuda_gpu(runtime)
                self.assertEqual(runtime.discovery_calls, 0)
                self.assertEqual(runtime.initialized_devices, [])

    def test_multiple_nvidia_gpu_uuids_are_rejected(self) -> None:
        environment = dict(TRIAL_ENVIRONMENT)
        environment["NVIDIA_VISIBLE_DEVICES"] = f"{GPU_UUID},{GPU_UUID}"
        with self.assertRaisesRegex(GpuSmokeSafetyError, "exactly one"):
            require_determined_trial_allocation(environment)

    def test_numeric_host_gpu_index_is_not_allocation_evidence(self) -> None:
        environment = dict(TRIAL_ENVIRONMENT)
        environment["NVIDIA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(GpuSmokeSafetyError, "host GPU indices"):
            require_determined_trial_allocation(environment)

    def test_zero_and_multiple_warp_visible_gpus_are_rejected(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count):
                runtime = FakeCudaRuntime(count=count)
                with mock.patch.dict(os.environ, TRIAL_ENVIRONMENT, clear=True):
                    with self.assertRaisesRegex(GpuSmokeSafetyError, "exactly one"):
                        require_single_visible_cuda_gpu(runtime)
                self.assertEqual(runtime.discovery_calls, 1)
                self.assertEqual(runtime.initialized_devices, [])

    def test_complete_trial_metadata_selects_only_container_logical_device(self) -> None:
        runtime = FakeCudaRuntime()
        with mock.patch.dict(os.environ, TRIAL_ENVIRONMENT, clear=True):
            audit = require_single_visible_cuda_gpu(runtime)
        self.assertEqual(runtime.initialized_devices, ["cuda:0"])
        self.assertEqual(audit.logical_device, "cuda:0")
        self.assertEqual(audit.visible_gpu_count, 1)
        self.assertEqual(audit.nvidia_visible_devices, GPU_UUID)

    def test_cuda_initialization_failure_never_attempts_cpu(self) -> None:
        runtime = FakeCudaRuntime(initialize_error=RuntimeError("driver refused context"))
        with mock.patch.dict(os.environ, TRIAL_ENVIRONMENT, clear=True):
            with self.assertRaisesRegex(GpuSmokeSafetyError, "CPU fallback is forbidden"):
                require_single_visible_cuda_gpu(runtime)
        self.assertEqual(runtime.initialized_devices, ["cuda:0"])
        self.assertNotIn("cpu", runtime.initialized_devices)

    def test_cpu_smoke_still_hides_cuda_and_cannot_enter_gpu_path(self) -> None:
        from experiment_runner.cpu_safety import prepare_cpu_smoke_environment

        runtime = FakeCudaRuntime()
        with mock.patch.dict(os.environ, TRIAL_ENVIRONMENT, clear=True):
            prepare_cpu_smoke_environment()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
            with self.assertRaisesRegex(GpuSmokeSafetyError, "CPU smoke safety"):
                require_single_visible_cuda_gpu(runtime)
        self.assertEqual(runtime.discovery_calls, 0)
        self.assertEqual(runtime.initialized_devices, [])

    def test_solver_cpu_mode_is_rejected_even_when_model_is_on_cuda(self) -> None:
        runtime = FakeCudaRuntime()
        scene = SimpleNamespace(
            model=SimpleNamespace(device=runtime.device),
            solver=SimpleNamespace(use_mujoco_cpu=True),
        )
        with self.assertRaisesRegex(GpuSmokeSafetyError, "MJWarp CUDA"):
            require_scene_on_logical_gpu(scene)

    def test_missing_optional_gpu_metadata_is_explicitly_unknown(self) -> None:
        runtime = FakeCudaRuntime()
        runtime.device.name = None
        runtime.device.uuid = None
        runtime.driver_version = mock.Mock(side_effect=RuntimeError("unavailable"))
        runtime.toolkit_version = mock.Mock(return_value=None)
        runtime.nvidia_driver_version = mock.Mock(return_value=None)
        environment = dict(TRIAL_ENVIRONMENT)
        environment.pop("CUDA_VISIBLE_DEVICES")
        with mock.patch.dict(os.environ, environment, clear=True):
            audit = require_single_visible_cuda_gpu(runtime)
        self.assertEqual(audit.model, "unknown")
        self.assertEqual(audit.uuid, "unknown")
        self.assertEqual(audit.cuda_driver_api_version, "unknown")
        self.assertEqual(audit.cuda_toolkit_version, "unknown")
        self.assertEqual(audit.nvidia_driver_version, "unknown")
        self.assertEqual(audit.cuda_visible_devices, "unknown")


class GpuProfileAndCliTests(unittest.TestCase):
    def test_gpu_integration_profile_is_non_authoritative_and_separate_from_formal(self) -> None:
        from experiment_runner.profiles import describe_profiles

        name = "mujoco-warp-cuda-dt1ms-integration-smoke-v1"
        profile = get_profile(name)
        availability = describe_profiles()[name]["availability"]
        formal = describe_profiles()["mujoco-native-dt1ms-v1"]
        self.assertFalse(profile.authoritative)
        self.assertFalse(profile.use_mujoco_cpu)
        self.assertEqual(profile.execution_tier, "development_integration_smoke")
        self.assertEqual(profile.case_duration_seconds, 1.0)
        self.assertEqual(availability["entrypoints"], ["smoke-drop-gpu"])
        self.assertTrue(availability["runnable"])
        self.assertFalse(formal["availability"]["runnable"])
        self.assertEqual(formal["availability"]["entrypoints"], [])

    def test_profiles_cli_json_exposes_only_the_gpu_smoke_entrypoint(self) -> None:
        from experiment_runner.cli import main

        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            return_code = main(["profiles"])
        profile = json.loads(output.getvalue())[
            "mujoco-warp-cuda-dt1ms-integration-smoke-v1"
        ]
        self.assertEqual(return_code, 0)
        self.assertFalse(profile["authoritative"])
        self.assertEqual(profile["availability"]["entrypoints"], ["smoke-drop-gpu"])

    def test_gpu_cli_help_exposes_trial_gate_without_host_device_option(self) -> None:
        from experiment_runner.cli import build_parser

        parser = build_parser()
        subparser = parser._subparsers._group_actions[0].choices["smoke-drop-gpu"]
        help_text = subparser.format_help()
        normalized_help = help_text.replace("-\n", "-").replace("\n", " ")
        self.assertIn("Determined trial", normalized_help)
        self.assertIn("container-logical cuda:0", normalized_help)
        self.assertIn("forbids CPU fallback", normalized_help)
        self.assertIn("does not submit a Determined experiment", normalized_help)
        self.assertNotIn("--gpu", help_text)
        self.assertNotIn("--device", help_text)

    def test_gpu_cli_json_remains_non_authoritative(self) -> None:
        from experiment_runner.cli import main

        result = {
            "status": "succeeded",
            "authoritative": False,
            "profile": "mujoco-warp-cuda-dt1ms-integration-smoke-v1",
            "actual_compute_device": "cuda:0",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            with mock.patch(
                "experiment_runner.smoke.run_gpu_smoke_drop",
                return_value=result,
            ):
                with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                    return_code = main(
                        [
                            "smoke-drop-gpu",
                            "fixtures/box",
                            "a" * 64,
                            "--data-root",
                            str(root.path),
                        ]
                    )
        decoded, _end = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(return_code, 0)
        self.assertFalse(decoded["authoritative"])

    def test_local_menu_and_viewer_do_not_expose_gpu_smoke(self) -> None:
        from experiment_runner.config import ControllerConfig
        from experiment_runner.controller import _print_menu

        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            _print_menu(ControllerConfig())
        menu = output.getvalue()
        viewer_source = (
            Path(__file__).resolve().parent.parent / "asset_viewer" / "app.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("smoke-drop-gpu", menu)
        self.assertNotIn("cuda:0", menu)
        self.assertNotIn("smoke-drop-gpu", viewer_source)

    def test_determined_example_is_placeholdered_bounded_and_no_sync(self) -> None:
        repository = Path(__file__).resolve().parent.parent
        example = (
            repository / "deployment" / "determined-gpu-smoke-8min.yaml"
        ).read_text(encoding="utf-8")
        deployment_guide = (repository / "deployment" / "README.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "slots_per_trial: 1",
            "max_slots: 1",
            "max_restarts: 0",
            "checkpoint_policy: none",
            "container_path: /tmp/newton-test-checkpoints",
            "timeout --signal=INT --kill-after=30s 8m",
            "Python 3.12 required",
            "--git-commit \"$NEWTON_TEST_GIT_COMMIT\"",
            "\"$NEWTON_TEST_PYTHON\" -m experiment_runner.cli smoke-drop-gpu",
            "image: REPLACE_WITH_APPROVED_PYTHON312_GPU_IMAGE",
            "resource_pool: REPLACE_WITH_APPROVED_SINGLE_GPU_RESOURCE_POOL",
            "host_path: REPLACE_WITH_APPROVED_HOST_DATA_ROOT",
            "container_path: REPLACE_WITH_TRIAL_DATA_ROOT",
        ):
            self.assertIn(required, example)
        self.assertEqual(example.count("max_length:"), 1)
        self.assertEqual(example.count("batches: 1"), 1)
        self.assertNotIn("uv run --frozen", example)
        self.assertNotIn("uv sync", example)
        for forbidden in (
            "nvidia-smi",
            "sudo",
            "nohup",
            "tmux",
            "screen",
            "setsid",
            "systemd",
            "det experiment create",
            "/mount/",
            "default",
            "Cotsheep",
            "ccai",
            "password",
            "token",
            "ssh://",
            "https://",
        ):
            self.assertNotIn(forbidden, example)
        self.assertIn("REPLACE_WITH_", deployment_guide)
        self.assertIn("搜索结果为零", deployment_guide)
        self.assertIn("det experiment create --paused", deployment_guide)
        self.assertIn("不激活 trial，也不分配 GPU", deployment_guide)
        self.assertIn("不会自动运行该命令", deployment_guide)


class RunnerHarness:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = DataRoot(Path(self.temporary.name) / "data")
        self.root.initialize()
        package = self.root.location("assets") / "fixtures" / "box" / ("a" * 64)
        package.mkdir(parents=True)
        (package / ASSET_ENTRYPOINT).write_text("#usda 1.0\n", encoding="utf-8")
        self.snapshot = {
            "identity": "fixtures/box",
            "version": "a" * 64,
            "entrypoint": ASSET_ENTRYPOINT,
            "dependencies": [],
            "readiness": {
                "drop": {"status": "ready", "reason_codes": []},
                "slope_friction": {
                    "status": "not_ready",
                    "reason_codes": ["fixture"],
                },
            },
            "package_root": package,
        }
        self.bounds = mock.Mock(
            minimum=np.array((-0.1, -0.1, 0.0)),
            maximum=np.array((0.1, 0.1, 0.2)),
        )
        self.geometry = mock.Mock(
            clearance=0.2,
            characteristic_length=0.2,
            effective_length=0.2,
            initial_bounds=self.bounds,
        )
        self.runtime = FakeCudaRuntime()
        self.stack = ExitStack()
        self.scene = self._new_scene(self.runtime.device)

    class FakeArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def numpy(self) -> np.ndarray:
            return self.values.copy()

    class FakeState:
        def __init__(self) -> None:
            self.body_q = RunnerHarness.FakeArray(
                np.zeros((1, 7), dtype=np.float64)
            )
            self.body_qd = RunnerHarness.FakeArray(
                np.zeros((1, 6), dtype=np.float64)
            )

        def clear_forces(self) -> None:
            return None

    def _new_scene(self, device, *, use_mujoco_cpu: bool = False):
        from experiment_runner.experiments.drop import DropScene

        return DropScene(
            model=mock.Mock(device=device),
            state=self.FakeState(),
            state_next=self.FakeState(),
            control=object(),
            contacts=object(),
            solver=mock.Mock(use_mujoco_cpu=use_mujoco_cpu),
        )

    def __enter__(self):
        self.stack.enter_context(mock.patch("experiment_runner.release.require_gpu_source", return_value={"verification": "fake"}))
        self.stack.enter_context(mock.patch("experiment_runner.readiness.require_locked_runtime", return_value={}))
        self.synchronize = self.stack.enter_context(mock.patch(
            "experiment_runner.experiments.drop.wp.synchronize_device"
        ))
        self.stack.enter_context(
            mock.patch(
                "experiment_runner.smoke.resolve_git_commit",
                return_value="b" * 40,
            )
        )
        self.stack.enter_context(
            mock.patch(
                "experiment_runner.smoke.snapshot_asset_version",
                return_value=self.snapshot,
            )
        )
        self.stack.enter_context(
            mock.patch(
                "experiment_runner.experiments.drop.measure_drop_geometry",
                return_value=self.geometry,
            )
        )
        self.create_scene = self.stack.enter_context(
            mock.patch(
                "experiment_runner.experiments.drop.create_drop_scene",
                return_value=self.scene,
            )
        )
        self.stack.enter_context(
            mock.patch(
                "experiment_runner.experiments.drop.compute_asset_bounds",
                return_value=self.bounds,
            )
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stack.close()
        self.temporary.cleanup()

    def run(self, environment_overrides: dict[str, str] | None = None, *, record_video=False):
        from experiment_runner.smoke import run_gpu_smoke_drop

        environment = dict(
            TRIAL_ENVIRONMENT
            if environment_overrides is None
            else environment_overrides
        )
        for key in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "SystemRoot"):
            if key in os.environ:
                environment[key] = os.environ[key]
        with mock.patch.dict(os.environ, environment, clear=True):
            from experiment_runner.gpu_safety import issue_gpu_execution_permit
            # Fake scene substitutes the real constructor, which attaches this permit.
            if "DET_TASK_ID" in environment:
                self.scene.gpu_permit = issue_gpu_execution_permit()
            return run_gpu_smoke_drop(
                self.root,
                asset_identity="fixtures/box",
                asset_version="a" * 64,
                gpu_runtime=self.runtime,
                record_video=record_video,
            )

    def documents(self) -> tuple[dict, dict, dict, str]:
        runs = list(self.root.location("runs").iterdir())
        if len(runs) != 1:
            raise AssertionError(f"expected one run directory, found {len(runs)}")
        run_directory = runs[0]
        manifest = read_json(run_directory / "manifest.json")
        status = read_json(run_directory / "status.json")
        case = read_json(
            run_directory
            / "cases"
            / "001-medium-gpu-integration-smoke"
            / "case.json"
        )
        log = (run_directory / "run.log").read_text(encoding="utf-8")
        return manifest, status, case, log


class GpuSmokeRunnerTests(unittest.TestCase):
    def test_wall_limit_stops_physics_and_retains_real_completed_count(self) -> None:
        with RunnerHarness() as harness, mock.patch(
            "experiment_runner.experiments.drop.time.monotonic", side_effect=[0.0, 301.0]
        ):
            with self.assertRaisesRegex(TimeoutError, "300-second"):
                harness.run()
            manifest, status, case, _log = harness.documents()
            self.assertEqual(status["status"], "failed")
            self.assertEqual(harness.scene.solver.step.call_count, 1)
            self.assertEqual(case["execution"]["completed_physics_steps"], 1)
            self.assertFalse(manifest["environment"]["execution"]["gpu_physics_completed"])

    def test_source_or_dependency_failure_stops_before_cuda_discovery(self) -> None:
        for path in ("experiment_runner.release.require_gpu_source", "experiment_runner.readiness.require_locked_runtime"):
            with self.subTest(path=path), RunnerHarness() as harness, mock.patch(path, side_effect=ValueError("prerequisite rejected")):
                with self.assertRaisesRegex(ValueError, "prerequisite rejected"):
                    harness.run()
                self.assertEqual(harness.runtime.discovery_calls, 0)
                self.assertEqual(list(harness.root.location("runs").iterdir()), [])

    def test_first_step_or_async_cuda_failure_does_not_claim_gpu_success(self) -> None:
        for failure_point in ("step", "synchronize"):
            with self.subTest(failure_point=failure_point), RunnerHarness() as harness:
                target = harness.scene.solver.step if failure_point == "step" else harness.synchronize
                target.side_effect = RuntimeError("injected failure")
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    harness.run()
                manifest, status, case, _log = harness.documents()
                self.assertEqual(status["status"], "failed")
                for execution in (manifest["environment"]["execution"], case["execution"]):
                    self.assertTrue(execution["solver_step_started"])
                    self.assertFalse(execution["solver_step_completed"])
                    self.assertEqual(execution["attempted_physics_step"], 1)
                    self.assertEqual(execution["completed_physics_steps"], 0)
                    self.assertEqual(execution["actual_compute_device"], "unknown")
                    self.assertFalse(execution["cuda_used"])
                    self.assertFalse(execution["gpu_physics_started"])
                    self.assertFalse(execution["gpu_physics_completed"])

    def test_interrupt_retains_completed_steps_and_pending_attempt(self) -> None:
        with RunnerHarness() as harness:
            harness.scene.solver.step.side_effect = [None, None, KeyboardInterrupt()]
            with self.assertRaises(KeyboardInterrupt):
                harness.run()
            manifest, status, case, _log = harness.documents()
            self.assertEqual(status["status"], "interrupted")
            for execution in (manifest["environment"]["execution"], case["execution"]):
                self.assertEqual(execution["completed_physics_steps"], 2)
                self.assertEqual(execution["attempted_physics_step"], 3)
                self.assertFalse(execution["solver_step_completed"])
                self.assertTrue(execution["cuda_used"])
                self.assertFalse(execution["gpu_physics_completed"])

    def test_uuid_mismatch_and_invalid_runtime_values_fail_before_model(self) -> None:
        for uuid, status in (("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "mismatched"), ("invalid", "invalid")):
            with self.subTest(uuid=uuid), RunnerHarness() as harness:
                harness.runtime.device.uuid = uuid
                with self.assertRaises(GpuSmokeSafetyError):
                    harness.run()
                harness.create_scene.assert_not_called()
                manifest, _status, case, _log = harness.documents()
                for execution in (manifest["environment"]["execution"], case["execution"]):
                    self.assertTrue(execution["cuda_context_initialized"])
                    self.assertEqual(execution["allocated_gpu_uuid"], GPU_UUID)
                    self.assertEqual(execution["runtime_gpu_uuid"], uuid)
                    self.assertEqual(execution["uuid_verification_status"], status)
                    self.assertFalse(execution["cuda_used"])

    def test_missing_uuid_records_warning_and_case_matches_manifest(self) -> None:
        with RunnerHarness() as harness:
            harness.runtime.device.uuid = None
            harness.run()
            manifest, _status, case, _log = harness.documents()
            for execution in (manifest["environment"]["execution"], case["execution"]):
                self.assertEqual(execution["uuid_verification_status"], "unavailable")
                self.assertTrue(execution["uuid_warning"])
                self.assertEqual(execution["runtime_gpu_uuid"], "unknown")
                self.assertEqual(execution["allocated_gpu_uuid"], GPU_UUID)

    def test_uuid_case_and_optional_runtime_prefix_are_normalized(self) -> None:
        runtime = FakeCudaRuntime()
        runtime.device.uuid = GPU_UUID[4:].upper()
        audit = require_single_visible_cuda_gpu(runtime, TRIAL_ENVIRONMENT)
        self.assertEqual(audit.uuid, GPU_UUID)
        self.assertEqual(audit.uuid_verification_status, "matched")

    def test_missing_trial_gate_creates_no_run_or_cuda_context(self) -> None:
        with RunnerHarness() as harness:
            environment = dict(TRIAL_ENVIRONMENT)
            environment.pop("DET_TASK_ID")
            with self.assertRaisesRegex(GpuSmokeSafetyError, "DET_TASK_ID"):
                harness.run(environment)
            run_directories = list(harness.root.location("runs").iterdir())
            discovery_calls = harness.runtime.discovery_calls
            initialized_devices = list(harness.runtime.initialized_devices)
        self.assertEqual(run_directories, [])
        self.assertEqual(discovery_calls, 0)
        self.assertEqual(initialized_devices, [])

    def test_invalid_data_root_and_unready_asset_fail_before_cuda_runtime(self) -> None:
        from experiment_runner.smoke import run_gpu_smoke_drop

        runtime = FakeCudaRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            uninitialized = DataRoot(Path(temporary) / "missing")
            with mock.patch(
                "experiment_runner.smoke.resolve_git_commit", return_value="b" * 40
            ):
                with self.assertRaises(Exception):
                    run_gpu_smoke_drop(
                        uninitialized,
                        asset_identity="fixtures/box",
                        asset_version="a" * 64,
                        gpu_runtime=runtime,
                    )
        self.assertEqual(runtime.discovery_calls, 0)
        self.assertEqual(runtime.initialized_devices, [])

        runtime = FakeCudaRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            root = DataRoot(Path(temporary) / "data")
            root.initialize()
            snapshot = {
                "readiness": {
                    "drop": {
                        "status": "not_ready",
                        "reason_codes": ["missing_collision"],
                    }
                }
            }
            with mock.patch(
                "experiment_runner.smoke.resolve_git_commit", return_value="b" * 40
            ), mock.patch(
                "experiment_runner.smoke.snapshot_asset_version",
                return_value=snapshot,
            ):
                with self.assertRaisesRegex(ValueError, "not ready"):
                    run_gpu_smoke_drop(
                        root,
                        asset_identity="fixtures/box",
                        asset_version="a" * 64,
                        gpu_runtime=runtime,
                    )
        self.assertEqual(runtime.discovery_calls, 0)
        self.assertEqual(runtime.initialized_devices, [])

    def test_model_creation_failure_retains_context_but_not_gpu_physics_claim(self) -> None:
        with RunnerHarness() as harness:
            harness.create_scene.side_effect = RuntimeError("model creation failed")
            with self.assertRaisesRegex(RuntimeError, "model creation failed"):
                harness.run()
            manifest, _status, case, _log = harness.documents()
        for execution in (manifest["environment"]["execution"], case["execution"]):
            self.assertTrue(execution["scheduler_allocation_metadata_verified"])
            self.assertEqual(execution["process_visible_gpu_count"], 1)
            self.assertEqual(execution["selected_logical_device"], "cuda:0")
            self.assertTrue(execution["cuda_context_initialized"])
            self.assertFalse(execution["model_device_verified"])
            self.assertFalse(execution["gpu_physics_started"])
            self.assertFalse(execution["gpu_physics_completed"])
            self.assertEqual(execution["actual_compute_device"], "unknown")
            self.assertFalse(execution["cuda_used"])

    def test_cpu_model_is_rejected_without_gpu_physics_success_claim(self) -> None:
        with RunnerHarness() as harness:
            cpu = SimpleNamespace(alias="cpu", ordinal=None, is_cuda=False)
            harness.scene = harness._new_scene(cpu)
            harness.create_scene.return_value = harness.scene
            with self.assertRaisesRegex(GpuSmokeSafetyError, "not on"):
                harness.run()
            manifest, _status, case, _log = harness.documents()
        self.assertFalse(case["execution"]["model_device_verified"])
        self.assertFalse(case["execution"]["gpu_physics_started"])
        self.assertEqual(case["execution"]["actual_compute_device"], "unknown")
        self.assertFalse(manifest["environment"]["execution"]["cuda_used"])

    def test_cpu_solver_mode_is_rejected_without_gpu_physics_success_claim(self) -> None:
        with RunnerHarness() as harness:
            harness.scene = harness._new_scene(
                harness.runtime.device, use_mujoco_cpu=True
            )
            harness.create_scene.return_value = harness.scene
            with self.assertRaisesRegex(GpuSmokeSafetyError, "MJWarp CUDA"):
                harness.run()
            manifest, _status, case, _log = harness.documents()
        self.assertFalse(case["execution"]["model_device_verified"])
        self.assertFalse(case["execution"]["gpu_physics_started"])
        self.assertEqual(case["execution"]["actual_compute_device"], "unknown")
        self.assertFalse(manifest["environment"]["execution"]["cuda_used"])

    def test_actual_gpu_is_claimed_only_after_a_successful_solver_step(self) -> None:
        with RunnerHarness() as harness:
            harness.scene.solver.step.side_effect = [
                None,
                None,
                RuntimeError("third step failed"),
            ]
            with self.assertRaisesRegex(RuntimeError, "third step failed"):
                harness.run()
            manifest, status, case, _log = harness.documents()
        execution = manifest["environment"]["execution"]
        self.assertEqual(case["git_commit"], manifest["git_commit"])
        self.assertEqual(case["source"], manifest["environment"]["source"])
        self.assertEqual(case["execution"]["determined_trial_allocation"], execution["determined_trial_allocation"])
        self.assertEqual(status["status"], "failed")
        self.assertTrue(execution["model_device_verified"])
        self.assertTrue(execution["gpu_physics_started"])
        self.assertTrue(execution["gpu_physics_verified"])
        self.assertFalse(execution["gpu_physics_completed"])
        self.assertEqual(execution["completed_physics_steps"], 2)
        self.assertEqual(execution["actual_compute_device"], "cuda:0")
        self.assertTrue(execution["cuda_used"])
        self.assertEqual(case["execution"]["actual_compute_device"], "cuda:0")

    def test_1000_steps_produce_auditable_non_authoritative_completion(self) -> None:
        with RunnerHarness() as harness:
            result = harness.run()
            manifest, status, case, log = harness.documents()
            run_directory = harness.root.location("runs") / result["run_id"]
            checksums_exists = (run_directory / "checksums.sha256").is_file()
            preview_exists = (run_directory / "preview.jpg").exists()
            video_exists = (
                run_directory
                / "cases"
                / "001-medium-gpu-integration-smoke"
                / "video.mp4"
            ).exists()
            step_count = harness.scene.solver.step.call_count

        self.assertFalse(result["authoritative"])
        self.assertFalse(manifest["authoritative"])
        self.assertFalse(status["authoritative"])
        self.assertFalse(case["authoritative"])
        self.assertEqual(manifest["profile"]["name"], result["profile"])
        self.assertEqual(manifest["git_commit"], "b" * 40)
        self.assertTrue(manifest["environment"]["software"]["python"].startswith("3.12"))
        self.assertIn("warp-lang", manifest["environment"]["software"])
        execution = manifest["environment"]["execution"]
        self.assertTrue(execution["scheduler_allocation_metadata_verified"])
        self.assertFalse(execution["scheduler_gate_is_authentication"])
        self.assertEqual(execution["actual_compute_device"], "cuda:0")
        self.assertEqual(execution["process_visible_gpu_count"], 1)
        self.assertEqual(execution["gpu"]["model"], "Fake GPU")
        self.assertEqual(execution["gpu"]["allocated_uuid"], GPU_UUID)
        self.assertTrue(execution["cuda_context_initialized"])
        self.assertTrue(execution["model_device_verified"])
        self.assertTrue(execution["gpu_physics_started"])
        self.assertTrue(execution["gpu_physics_completed"])
        self.assertTrue(execution["gpu_physics_verified"])
        self.assertEqual(execution["completed_physics_steps"], 1000)
        self.assertTrue(execution["cuda_used"])
        self.assertFalse(execution["cpu_fallback"])
        self.assertEqual(
            execution["solver"], "newton.solvers.SolverMuJoCo"
        )
        self.assertEqual(execution["compute_backend"], "mujoco-warp-cuda")
        determined = manifest["environment"]["determined"]
        self.assertEqual(determined["DET_TRIAL_ID"], "202")
        self.assertEqual(determined["DET_RESOURCES_ID"], "resources-505")
        self.assertEqual(determined["DET_RESOURCES_TYPE"], "unknown")
        self.assertEqual(case["physics_steps"], 1000)
        self.assertEqual(case["recording"]["status"], "not_attempted")
        self.assertIn("misuse gate passed", log)
        self.assertIn("not authentication", log)
        self.assertTrue(checksums_exists)
        self.assertFalse(preview_exists)
        self.assertFalse(video_exists)
        self.assertEqual(step_count, 1000)


class GpuVideoSmokeTests(unittest.TestCase):
    def recording_mocks(self, harness):
        from asset_viewer.camera import AssetBounds
        stack = harness.stack
        viewer = mock.Mock()
        stack.enter_context(mock.patch(
            "experiment_runner.experiments.gpu_recording._open_gpu_viewer",
            return_value=(viewer, {"backend": "egl", "device": "cuda:0"}),
        ))
        frame = stack.enter_context(mock.patch(
            "experiment_runner.experiments.recording._render_frame",
            return_value=np.full((360, 640, 3), 80, dtype=np.uint8),
        ))
        stack.enter_context(mock.patch(
            "asset_viewer.camera.compute_asset_bounds",
            return_value=AssetBounds(harness.bounds.minimum, harness.bounds.maximum),
        ))
        stack.enter_context(mock.patch("asset_viewer.camera.frame_camera_on_bounds"))
        return viewer, frame

    def test_video_uses_same_1000_steps_encodes_75_frames_and_is_indexed(self):
        import hashlib
        import imageio_ffmpeg
        with RunnerHarness() as harness:
            viewer, frame = self.recording_mocks(harness)
            result = harness.run(record_video=True)
            manifest, status, case, _ = harness.documents()
            run = harness.root.location("runs") / result["run_id"]
            video = run / "cases/001-medium-gpu-integration-smoke/video.mp4"
            count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
            self.assertEqual(count, 75)
            self.assertAlmostEqual(seconds, 1.5)
            self.assertEqual(case["video_frames"], count)
            self.assertEqual(case["physics_steps"], 1000)
            self.assertEqual(harness.scene.solver.step.call_count, 1000)
            self.assertEqual(frame.call_count, 51)
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(manifest["recording"]["status"], "succeeded")
            self.assertEqual(result["profile"], "mujoco-warp-cuda-dt1ms-video-smoke-v1")
            self.assertFalse(case["authoritative"])
            viewer.close.assert_called_once()
            for line in (run / "checksums.sha256").read_text().splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((run / relative).read_bytes()).hexdigest(), digest)
            index = (harness.root.location("web") / "index.json").read_text(encoding="utf-8")
            self.assertIn("video.mp4", index)
            self.assertIn("poster.jpg", index)

    def test_initial_frame_failure_keeps_physics_unstarted(self):
        from experiment_runner.experiments.gpu_recording import GpuRecordingError
        with RunnerHarness() as harness:
            viewer, frame = self.recording_mocks(harness)
            frame.side_effect = RuntimeError("interop failed")
            with self.assertRaises(GpuRecordingError):
                harness.run(record_video=True)
            manifest, status, case, _ = harness.documents()
            self.assertEqual(case["execution"]["completed_physics_steps"], 0)
            self.assertFalse(case["execution"]["cuda_used"])
            self.assertEqual(case["recording"]["stage"], "frame_readback")
            self.assertEqual(manifest["recording"]["status"], "failed")
            self.assertEqual(status["status"], "failed")
            viewer.close.assert_called_once()

    def test_egl_mapping_failure_evidence_survives_recording_wrapper_and_persistence(self):
        from experiment_runner.experiments.gpu_recording import EglDeviceMappingError, GpuRecordingError
        evidence = {
            "reason_code": "no_logical_cuda_zero", "egl_device_count": 1,
            "devices": [{"egl_index": 0, "supports_cuda_mapping": True, "cuda_device": 3}],
        }
        with RunnerHarness() as harness:
            with mock.patch(
                "experiment_runner.experiments.gpu_recording._open_gpu_viewer",
                side_effect=EglDeviceMappingError(evidence),
            ):
                with self.assertRaises(GpuRecordingError) as caught:
                    harness.run(record_video=True)
            manifest, status, case, _ = harness.documents()
            self.assertIn("no_logical_cuda_zero", str(caught.exception))
            self.assertEqual(manifest["recording"]["diagnostics"], evidence)
            self.assertEqual(case["recording"]["diagnostics"], evidence)
            self.assertEqual(case["recording"]["stage"], "renderer_setup")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(case["execution"]["completed_physics_steps"], 0)
            self.assertFalse(case["execution"]["cuda_used"])
            harness.scene.solver.step.assert_not_called()

    def test_frame_failure_keeps_only_completed_physics_steps(self):
        from experiment_runner.experiments.gpu_recording import GpuRecordingError
        with RunnerHarness() as harness:
            viewer, frame = self.recording_mocks(harness)
            frame.side_effect = [frame.return_value, RuntimeError("frame failed")]
            with self.assertRaises(GpuRecordingError):
                harness.run(record_video=True)
            manifest, status, case, _ = harness.documents()
            self.assertEqual(case["execution"]["completed_physics_steps"], 20)
            self.assertTrue(case["execution"]["cuda_used"])
            self.assertFalse(case["execution"]["gpu_physics_completed"])
            self.assertEqual(status["status"], "failed")
            self.assertEqual(manifest["recording"]["status"], "failed")
            self.assertFalse(list(harness.root.location("runs").rglob("*.mp4")))
            viewer.close.assert_called_once()

    def test_encoding_failure_after_physics_does_not_report_video_success(self):
        from experiment_runner.experiments.gpu_recording import GpuRecordingError
        with RunnerHarness() as harness:
            self.recording_mocks(harness)
            with mock.patch("experiment_runner.video.H264VideoWriter") as writer:
                writer.return_value.__exit__.side_effect = RuntimeError("encoder finalize failed")
                with self.assertRaises(GpuRecordingError):
                    harness.run(record_video=True)
            manifest, status, case, _ = harness.documents()
            self.assertEqual(case["execution"]["completed_physics_steps"], 1000)
            self.assertTrue(case["execution"]["gpu_physics_completed"])
            self.assertEqual(status["status"], "failed")
            self.assertEqual(manifest["recording"]["status"], "failed")
            self.assertEqual(case["recording"]["stage"], "encoder_finalize")

    def test_interrupt_closes_viewer_and_retains_interrupted_status(self):
        with RunnerHarness() as harness:
            viewer, frame = self.recording_mocks(harness)
            frame.side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                harness.run(record_video=True)
            manifest, status, case, _ = harness.documents()
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(case["recording"]["status"], "interrupted")
            self.assertEqual(manifest["exit_code"], 130)
            viewer.close.assert_called_once()

    def test_video_option_does_not_bypass_allocation_gate(self):
        with RunnerHarness() as harness:
            environment = dict(TRIAL_ENVIRONMENT)
            environment.pop("DET_TASK_ID")
            with mock.patch("experiment_runner.experiments.gpu_recording._open_gpu_viewer") as opener:
                with self.assertRaises(GpuSmokeSafetyError):
                    harness.run(environment, record_video=True)
                opener.assert_not_called()
            self.assertEqual(harness.runtime.discovery_calls, 0)

    def test_cli_video_flag_is_explicit_and_default_is_off(self):
        from experiment_runner.cli import build_parser
        args = ["smoke-drop-gpu", "fixtures/box", "a" * 64]
        self.assertFalse(build_parser().parse_args(args).record_video)
        self.assertTrue(build_parser().parse_args(args + ["--record-video"]).record_video)

    def test_video_profile_is_permitted_but_modified_or_formal_profile_is_rejected(self):
        from dataclasses import replace
        from experiment_runner.gpu_safety import issue_gpu_execution_permit, require_gpu_execution_permit
        permit = issue_gpu_execution_permit(TRIAL_ENVIRONMENT)
        video = get_profile("mujoco-warp-cuda-dt1ms-video-smoke-v1")
        require_gpu_execution_permit(permit, video)
        for profile in (replace(video, case_duration_seconds=10), get_profile("mujoco-native-dt1ms-v1")):
            with self.assertRaises(GpuSmokeSafetyError):
                require_gpu_execution_permit(permit, profile)


if __name__ == "__main__":
    unittest.main()
