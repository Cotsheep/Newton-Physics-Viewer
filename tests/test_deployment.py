from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from experiment_runner.release import MANIFEST, export_source, require_gpu_source, source_files

ROOT = Path(__file__).resolve().parent.parent


def validate_template(document):
    """Check the bounded contract at its actual YAML paths and Python value types."""
    if not isinstance(document, dict):
        raise ValueError("template must be a mapping")
    expected = {
        ("resources", "slots_per_trial"): 1, ("resources", "max_slots"): 1,
        ("max_restarts",): 0, ("checkpoint_policy",): "none",
        ("checkpoint_storage", "type"): "directory",
        ("checkpoint_storage", "container_path"): "/tmp/newton-test-checkpoints",
        ("hyperparameters", "global_batch_size"): 1,
        ("searcher", "name"): "single", ("searcher", "max_length", "batches"): 1,
        ("searcher", "smaller_is_better"): True,
    }
    for keys, value in expected.items():
        actual = document
        try:
            for key in keys:
                actual = actual[key]
        except (KeyError, TypeError) as exc:
            raise ValueError("Missing " + ".".join(keys)) from exc
        if type(actual) is not type(value) or actual != value:
            raise ValueError("Wrong type or value for " + ".".join(keys))
    for section, key in (("environment", "image"), ("resources", "resource_pool")):
        if not document[section][key].startswith("REPLACE_WITH_"):
            raise ValueError("Template must use placeholders")
    mounts = document["bind_mounts"]
    if type(mounts) is not list or len(mounts) != 1:
        raise ValueError("Default image-contained environment needs only one data mount")
    if mounts[0]["read_only"] is not False:
        raise ValueError("Data mount must be writable")
    for key in ("host_path", "container_path"):
        if not mounts[0][key].startswith("REPLACE_WITH_"):
            raise ValueError("Mount must remain a placeholder")
    variables = document["environment"]["environment_variables"]
    if type(variables) is not list or any(type(v) is not str or "=" not in v for v in variables):
        raise ValueError("Environment variables must be a string list")
    env = dict(v.split("=", 1) for v in variables)
    for name in ("NEWTON_TEST_PYTHON", "NEWTON_TEST_GIT_COMMIT", "NEWTON_SMOKE_ASSET_VERSION", "NEWTON_SMOKE_ASSET_IDENTITY", "NEWTON_DATA_ROOT"):
        if not env[name].startswith("REPLACE_WITH_"):
            raise ValueError("Deployment identities must remain placeholders")
    if type(document["entrypoint"]) is not str:
        raise ValueError("Entrypoint must be a shell command string")


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.document = yaml.safe_load((ROOT / "deployment/determined-gpu-smoke-8min.yaml").read_text(encoding="utf-8"))

    def test_yaml_structure_and_types(self):
        validate_template(self.document)

    def test_video_template_retains_limits_and_requests_graphics_only_as_needed(self):
        document = yaml.safe_load((ROOT / "deployment/determined-gpu-video-smoke-8min.yaml").read_text(encoding="utf-8"))
        validate_template(document)
        self.assertIn("--record-video", document["entrypoint"])
        self.assertIn("timeout --signal=INT --kill-after=30s 8m", document["entrypoint"])
        env = dict(item.split("=", 1) for item in document["environment"]["environment_variables"])
        self.assertEqual(env["NVIDIA_DRIVER_CAPABILITIES"], "compute,utility,graphics")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "egl")
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", env)
        self.assertFalse(any(key.startswith("DET_") for key in env))
        for command in ("pip install", "uv sync", "sudo", "det experiment"):
            self.assertNotIn(command, document["entrypoint"])

    def test_wrong_nesting_and_string_or_boolean_slot_counts_fail(self):
        for value in ("1", True, 2):
            document = copy.deepcopy(self.document)
            document["resources"]["slots_per_trial"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_template(document)
        document = copy.deepcopy(self.document)
        document["global_batch_size"] = document.pop("hyperparameters")["global_batch_size"]
        with self.assertRaises(ValueError):
            validate_template(document)

    def test_trial_has_no_install_submit_or_background_commands(self):
        import re
        command = self.document["entrypoint"]
        self.assertIn("timeout --signal=INT --kill-after=30s 8m", command)
        self.assertNotIn("assert ", command)
        self.assertIn('test -x "$NEWTON_TEST_PYTHON"', command)
        self.assertIn("^[0-9a-f]{64}$", command)
        self.assertIn("^[0-9a-f]{40}$", command)
        self.assertIsNone(re.search(r"\b(uv|pip|sudo|nohup|tmux|screen|setsid|det|curl|wget)\b", command))
        self.assertIsNone(re.search(r"(?<![&>])&(?![&>])", command))


class ReleaseTests(unittest.TestCase):
    def make_source(self, root):
        (root / "experiment_runner").mkdir(parents=True)
        (root / "asset_viewer").mkdir()
        (root / "asset_viewer/__init__.py").write_text("")
        (root / "experiment_runner/worker.py").write_text("value = 1\n")
        (root / "pyproject.toml").write_text("[project]\n")
        (root / "uv.lock").write_text("version = 1\n")

    def test_export_manifest_requires_exact_executable_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            manifest = {"schema_version": 1, "git_commit": "a" * 40, "files": source_files(root)}
            (root / MANIFEST).write_text(json.dumps(manifest))
            with mock.patch("experiment_runner.release.git", side_effect=OSError("no git")):
                self.assertEqual(require_gpu_source("a" * 40, root)["verification"], "verified_source_export")
                (root / "experiment_runner/worker.py").write_text("value = 2\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    require_gpu_source("a" * 40, root)

    def test_dirty_checkout_and_missing_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            with mock.patch("experiment_runner.release.git", side_effect=["a" * 40, " M worker.py"]):
                with self.assertRaisesRegex(ValueError, "clean checkout"):
                    require_gpu_source("a" * 40, root)
            with mock.patch("experiment_runner.release.git", side_effect=OSError("no git")):
                with self.assertRaisesRegex(ValueError, "verified export"):
                    require_gpu_source("a" * 40, root)

    def test_export_reads_commit_blobs_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "source", Path(directory) / "export"
            self.make_source(root)
            files = {name: (root / name).read_bytes() for name in source_files(root)}
            listing = "\0".join(f"100644 blob {'b' * 40}\t{name}" for name in files)
            def fake_git(_root, *args):
                if args[0] == "rev-parse": return "a" * 40
                if args[0] == "status": return ""
                if args[0] == "ls-tree": return listing
                raise AssertionError(args)
            def blob(args, **kwargs):
                return subprocess.CompletedProcess(args, 0, files[args[-1].split(":", 1)[1]])
            with mock.patch("experiment_runner.release.git", side_effect=fake_git), mock.patch("experiment_runner.release.subprocess.run", side_effect=blob):
                export_source(output, root)
                self.assertEqual(source_files(output), source_files(root))
                with self.assertRaisesRegex(ValueError, "never overwritten"):
                    export_source(output, root)
