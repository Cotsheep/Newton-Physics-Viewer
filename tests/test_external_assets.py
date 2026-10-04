from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from experiment_runner.assets import (
    ASSET_ENTRYPOINT, AssetValidationError, accept_asset, list_asset_versions,
    snapshot_asset_version,
)
from experiment_runner.cli import main
from experiment_runner.controller import list_assets_for_menu
from experiment_runner.external_assets import register_external_asset, verify_external_asset_snapshot
from experiment_runner.storage import DataRoot, read_json
from tests.test_experiment_runner import READY_USDA


class ExternalAssetsTests(unittest.TestCase):
    def setUp(self):
        for target in ("experiment_runner.external_assets.resolve_git_commit", "experiment_runner.assets.resolve_git_commit"):
            patcher = mock.patch(target, return_value="a" * 40)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = DataRoot(self.directory / "outputs")
        self.root.initialize()
        self.source = self.directory / "shared"
        self.source.mkdir()
        self.asset = self.source / "model.usda"
        self.asset.write_text(READY_USDA, encoding="utf-8")

    def register(self, **kwargs):
        return register_external_asset(self.root, kwargs.pop("identity", "Shared/box"),
                                       source_root=self.source, source_name="Shared dataset",
                                       entrypoint=kwargs.pop("entrypoint", "model.usda"),
                                       git_commit="a" * 40, **kwargs)

    def snapshot(self, report):
        return snapshot_asset_version(self.root, report["asset_identity"], report["asset_version"])

    def source_contents(self):
        result = {}
        for path in self.source.rglob("*"):
            info = path.stat()
            result[path.relative_to(self.source).as_posix()] = (
                info.st_mode, info.st_mtime_ns, info.st_ctime_ns, info.st_ino,
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            )
        return result

    def test_register_snapshot_and_list_never_copy_move_chmod_or_write_source(self):
        self.asset.write_text(READY_USDA.replace('def Xform "World"\n{',
            'def Xform "World"\n{\n    custom asset texture = @texture.png@'), encoding="utf-8")
        (self.source / "texture.png").write_bytes(b"dependency bytes")
        original = self.source_contents()
        original_open = Path.open
        original_replace = os.replace

        def guarded_open(path, mode="r", *args, **kwargs):
            if path.resolve().is_relative_to(self.source.resolve()):
                self.assertFalse(any(flag in mode for flag in "wa+x"), (path, mode))
            return original_open(path, mode, *args, **kwargs)

        def guarded_replace(source, destination):
            self.assertFalse(Path(source).resolve().is_relative_to(self.source.resolve()))
            self.assertFalse(Path(destination).resolve().is_relative_to(self.source.resolve()))
            return original_replace(source, destination)

        with mock.patch.object(Path, "open", guarded_open), mock.patch("os.replace", guarded_replace), \
                mock.patch("os.chmod", side_effect=AssertionError("must not chmod source")), \
                mock.patch("shutil.copytree", side_effect=AssertionError("must not copy assets")):
            report = self.register()
            self.assertEqual(report["status"], "registered", report)
            snapshot = self.snapshot(report)
            verify_external_asset_snapshot(self.root, snapshot)
            rows = list_asset_versions(self.root)
        self.assertEqual(self.source_contents(), original)
        self.assertFalse(any(self.root.location("inbox").iterdir()))
        self.assertFalse(any(self.root.location("assets").iterdir()))
        self.assertEqual(snapshot["package_root"], self.source.resolve())
        self.assertEqual({item["path"] for item in snapshot["dependencies"]}, {"model.usda", "texture.png"})
        self.assertEqual(rows[0]["storage_mode"], "external_readonly")
        self.assertEqual(len(list(self.root.location("asset_references").rglob("registration.json"))), 1)

    def test_nested_entrypoint_can_reference_sibling_layer_inside_declared_root(self):
        nested = self.source / "models"
        nested.mkdir()
        (nested / "scene.usda").write_text('#usda 1.0\n(\nsubLayers = [@../model.usda@]\n)\n', encoding="utf-8")
        report = self.register(entrypoint="models/scene.usda")
        self.assertEqual(report["status"], "registered", report)
        snapshot = self.snapshot(report)
        self.assertEqual({item["path"] for item in snapshot["dependencies"]}, {"models/scene.usda", "model.usda"})
        self.assertEqual(snapshot["readiness"]["drop"]["status"], "ready")

    def test_missing_parameters_register_as_not_ready_without_filling_values(self):
        self.asset.write_text(READY_USDA.replace("float physics:mass = 1", "float physics:mass = 0"), encoding="utf-8")
        before = self.source_contents()
        report = self.register()
        self.assertEqual(report["status"], "registered")
        self.assertEqual(report["ready_templates"], [])
        self.assertEqual(list_assets_for_menu(self.root)[0]["readiness"], "未就绪")
        self.assertEqual(self.source_contents(), before)

    def test_changed_dependency_refuses_old_version_and_requires_explicit_registration(self):
        self.asset.write_text(READY_USDA.replace('def Xform "World"\n{',
            'def Xform "World"\n{\n    custom asset payload = @payload.json@'), encoding="utf-8")
        dependency = self.source / "payload.json"
        dependency.write_text("first", encoding="utf-8")
        first = self.register()
        dependency.write_text("second", encoding="utf-8")
        with self.assertRaisesRegex(AssetValidationError, "no longer matches"):
            self.snapshot(first)
        self.assertEqual(list_assets_for_menu(self.root)[0]["template_readiness"], {})
        self.assertEqual(list_assets_for_menu(self.root)[0]["readiness"], "源内容已变化，需重新登记")
        second = self.register()
        self.assertNotEqual(first["asset_version"], second["asset_version"])
        self.assertEqual(len(list_asset_versions(self.root)), 2)
        self.snapshot(second)

    def test_cache_does_not_hide_changed_authored_parameters(self):
        from pxr import Usd
        held_stage = Usd.Stage.Open(str(self.asset))
        first = self.register()
        self.asset.write_text(READY_USDA.replace("float physics:mass = 1", "float physics:mass = -2"), encoding="utf-8")
        second = self.register()
        self.assertNotEqual(first["asset_version"], second["asset_version"])
        self.assertEqual(second["ready_templates"], [])
        self.assertIsNotNone(held_stage)

    def test_run_verification_rejects_change_even_if_original_bytes_restored(self):
        snapshot = self.snapshot(self.register())
        self.asset.write_text("changed", encoding="utf-8")
        self.asset.write_text(READY_USDA, encoding="utf-8")
        with self.assertRaisesRegex(AssetValidationError, "changed during this run"):
            verify_external_asset_snapshot(self.root, snapshot)

    def test_deleted_source_is_reported_as_unavailable(self):
        snapshot = self.snapshot(self.register())
        self.asset.unlink()
        with self.assertRaises(AssetValidationError) as caught:
            verify_external_asset_snapshot(self.root, snapshot)
        self.assertEqual(caught.exception.code, "external_source_unavailable")

    def test_real_newton_scene_import_leaves_external_files_unchanged(self):
        from experiment_runner.cpu_safety import prepare_cpu_smoke_environment
        from experiment_runner.experiments.drop import create_drop_scene, measure_drop_geometry
        from experiment_runner.profiles import get_profile

        snapshot = self.snapshot(self.register())
        before = self.source_contents()
        original_open = Path.open

        def guarded_open(path, mode="r", *args, **kwargs):
            if path.resolve().is_relative_to(self.source.resolve()):
                self.assertFalse(any(flag in mode for flag in "wa+x"), (path, mode))
            return original_open(path, mode, *args, **kwargs)

        with mock.patch.dict(os.environ, dict(os.environ), clear=True), mock.patch.object(Path, "open", guarded_open):
            prepare_cpu_smoke_environment()
            profile = get_profile("mujoco-cpu-wsl-smoke-v1")
            geometry = measure_drop_geometry(self.asset, profile=profile)
            scene = create_drop_scene(self.asset, profile=profile,
                                      clearance=geometry.clearance, measured_bounds=geometry.initial_bounds)
            self.assertEqual(scene.model.body_count, 1)
            self.assertEqual(str(scene.model.device), "cpu")
            verify_external_asset_snapshot(self.root, snapshot)
        self.assertEqual(self.source_contents(), before)

    def test_source_junction_is_rejected_before_registration(self):
        original = Path.is_junction
        with mock.patch.object(Path, "is_junction", lambda path: path == self.source or original(path)):
            with self.assertRaisesRegex(AssetValidationError, "links or junctions"):
                self.register()

    def test_malformed_registry_is_not_silently_ignored(self):
        self.register()
        path = next(self.root.location("asset_references").rglob("registration.json"))
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(AssetValidationError) as caught:
            list_asset_versions(self.root)
        self.assertEqual(caught.exception.code, "external_registration_invalid")

    def test_duplicate_keeps_original_registration_metadata(self):
        first = self.register()
        path = next(self.root.location("asset_references").rglob("registration.json"))
        before = path.read_bytes()
        second = self.register()
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(first["asset_version"], second["asset_version"])
        self.assertEqual(path.read_bytes(), before)

    def test_source_binding_cannot_be_silently_changed(self):
        self.register()
        other = self.source / "other.usda"
        other.write_text(READY_USDA, encoding="utf-8")
        report = self.register(entrypoint="other.usda")
        self.assertEqual(report["error"]["code"], "external_source_binding_conflict")

    def test_missing_and_outside_dependencies_fail_without_registration(self):
        for target, code in [("missing.png", "unresolved_asset_dependency"), ("../outside.png", "asset_dependency_escape")]:
            with self.subTest(target=target):
                (self.directory / "outside.png").write_bytes(b"outside")
                self.asset.write_text(READY_USDA.replace('def Xform "World"\n{',
                    f'def Xform "World"\n{{\n    custom asset texture = @{target}@'), encoding="utf-8")
                before = self.source_contents()
                report = self.register()
                self.assertEqual(report["status"], "failed", report)
                self.assertEqual(report["error"]["code"], code)
                self.assertEqual(self.source_contents(), before)
                self.assertFalse(any(self.root.location("asset_references").iterdir()))

    def test_entrypoint_escape_is_rejected(self):
        report = self.register(entrypoint="../shared/model.usda")
        self.assertEqual(report["status"], "failed")

    def test_source_output_overlap_is_rejected_before_any_write(self):
        before = sorted(self.root.path.rglob("*"))
        for source in [self.root.path, self.root.location("assets"), self.directory]:
            with self.subTest(source=source), self.assertRaisesRegex(AssetValidationError, "must be separate"):
                register_external_asset(self.root, "Shared/box", source_root=source,
                                        entrypoint="model.usda", source_name="Shared", git_commit="a" * 40)
        self.assertEqual(sorted(self.root.path.rglob("*")), before)

    def test_missing_source_root_has_a_clear_error_without_output_writes(self):
        before = sorted(self.root.path.rglob("*"))
        with self.assertRaisesRegex(AssetValidationError, "does not exist or cannot be read"):
            register_external_asset(self.root, "Shared/missing", source_root=self.directory / "missing",
                                    entrypoint="model.usda", source_name="Shared", git_commit="a" * 40)
        self.assertEqual(sorted(self.root.path.rglob("*")), before)

    def test_changed_version_is_rejected_by_real_gpu_preflight_before_any_allocation(self):
        from experiment_runner.smoke import run_gpu_smoke_drop

        report = self.register()
        self.asset.write_text(READY_USDA.replace("float physics:mass = 1", "float physics:mass = 2"), encoding="utf-8")
        with mock.patch("experiment_runner.smoke.resolve_git_commit", return_value="a" * 40), \
                mock.patch("experiment_runner.gpu_safety.issue_gpu_execution_permit") as permit:
            with self.assertRaises(AssetValidationError) as caught:
                run_gpu_smoke_drop(self.root, asset_identity="Shared/box", asset_version=report["asset_version"])
        self.assertEqual(caught.exception.code, "external_asset_version_mismatch")
        permit.assert_not_called()
        self.assertFalse(any(self.root.location("runs").iterdir()))

    def test_external_identity_cannot_be_reused_for_managed_storage(self):
        self.register()
        package = self.root.location("inbox") / "Shared/box"
        package.mkdir(parents=True)
        (package / ASSET_ENTRYPOINT).write_text(READY_USDA, encoding="utf-8")
        report = accept_asset(self.root, "Shared/box", git_commit="a" * 40, enforce_readonly=False)
        self.assertEqual(report["error"]["code"], "asset_storage_mode_conflict")
        self.assertTrue(package.exists())

    def test_managed_identity_cannot_be_reused_for_external_storage(self):
        package = self.root.location("inbox") / "Shared/box"
        package.mkdir(parents=True)
        (package / ASSET_ENTRYPOINT).write_text(READY_USDA, encoding="utf-8")
        accepted = accept_asset(self.root, "Shared/box", git_commit="a" * 40, enforce_readonly=False)
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(self.register()["error"]["code"], "asset_storage_mode_conflict")

    def test_case_collision_includes_external_identities(self):
        self.register()
        self.assertEqual(self.register(identity="shared/box")["error"]["code"], "asset_identity_case_collision")

    def test_legacy_data_root_needs_no_reconfiguration(self):
        self.root.location("asset_references").rmdir()
        self.root.require_initialized()
        self.assertEqual(list_asset_versions(self.root), [])
        self.assertEqual(self.register()["status"], "registered")
        self.root.require_initialized()

    def test_cli_uses_same_registry_and_reports_readiness(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["register-external-asset", "Shared/box", "--source-root", str(self.source),
                           "--entrypoint", "model.usda", "--source-name", "Shared dataset",
                           "--data-root", str(self.root.path), "--git-commit", "a" * 40])
        self.assertEqual(status, 0, output.getvalue())
        self.assertIn('"registered"', output.getvalue())
        self.assertIn('"drop"', output.getvalue())

    def test_symlink_source_and_dependency_are_rejected(self):
        link = self.directory / "linked"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Symlink permission unavailable: {exc}")
        with self.assertRaisesRegex(AssetValidationError, "links or junctions"):
            register_external_asset(self.root, "Shared/link", source_root=link,
                                    entrypoint="model.usda", source_name="Shared", git_commit="a" * 40)
        (self.source / "alias.usda").symlink_to(self.asset)
        report = self.register(entrypoint="alias.usda")
        self.assertEqual(report["error"]["code"], "asset_dependency_symlink")


class ExternalGpuIntegrationTests(unittest.TestCase):
    def _register_for_harness(self, harness):
        source = Path(harness.temporary.name) / "shared"
        source.mkdir()
        (source / "model.usda").write_text(READY_USDA, encoding="utf-8")
        with mock.patch("experiment_runner.external_assets.resolve_git_commit", return_value="b" * 40):
            report = register_external_asset(harness.root, "Shared/box", source_root=source,
                                            source_name="Shared", entrypoint="model.usda", git_commit="b" * 40)
        snapshot = snapshot_asset_version(harness.root, "Shared/box", report["asset_version"])
        harness.snapshot.update(snapshot)
        return source

    def test_gpu_runner_reads_external_path_and_records_all_verifications(self):
        from tests.test_gpu_smoke import RunnerHarness
        with RunnerHarness() as harness:
            source = self._register_for_harness(harness)
            result = harness.run()
            self.assertEqual(result["status"], "succeeded")
            harness.create_scene.assert_called_once()
            self.assertEqual(harness.create_scene.call_args.args[0], source / "model.usda")
            manifest, status, case, _ = harness.documents()
            self.assertEqual(manifest["asset"]["storage_mode"], "external_readonly")
            self.assertEqual([item["stage"] for item in manifest["asset_input_verification"]["checks"]],
                             ["before_load", "after_load", "after_run"])
            self.assertEqual(status["status"], "succeeded")
            from experiment_runner.results import build_result_index

            public = build_result_index(harness.root)["assets"][0]["runs"][0]
            self.assertEqual(public["asset_storage_mode"], "external_readonly")
            self.assertEqual(public["source_name"], "Shared")
            self.assertNotIn("external_source", public)
            self.assertNotIn(str(source), str(public))
            self.assertTrue((source / "model.usda").exists())

    def test_change_before_cuda_initialization_is_a_failed_run(self):
        from tests.test_gpu_smoke import RunnerHarness
        with RunnerHarness() as harness:
            source = self._register_for_harness(harness)
            (source / "model.usda").write_text("changed", encoding="utf-8")
            with self.assertRaises(AssetValidationError):
                harness.run()
            self.assertEqual(harness.runtime.discovery_calls, 0)
            manifest, status, case, _ = harness.documents()
            self.assertEqual(status["status"], "failed")
            self.assertEqual(case["failure"]["code"], "external_asset_changed")
            self.assertEqual(manifest["asset_input_verification"]["stage"], "before_load")

    def test_change_during_simulation_never_reports_success(self):
        from tests.test_gpu_smoke import RunnerHarness
        with RunnerHarness() as harness:
            source = self._register_for_harness(harness)

            def change_source(*_args, **_kwargs):
                (source / "model.usda").write_text("changed during run", encoding="utf-8")
                return {"recording": {"status": "not_attempted", "files": []}, "physics_steps": 1000}

            with mock.patch("experiment_runner.experiments.drop.simulate_drop_case_without_recording", side_effect=change_source):
                with self.assertRaises(AssetValidationError):
                    harness.run()
            manifest, status, case, _ = harness.documents()
            self.assertEqual(status["status"], "failed")
            self.assertEqual(manifest["asset_input_verification"]["stage"], "after_run")
            self.assertEqual(case["failure"]["code"], "external_asset_changed")


if __name__ == "__main__":
    unittest.main()
