from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiment_runner import controller
from experiment_runner.readiness import check_readiness, require_locked_runtime
from experiment_runner.storage import DataRoot


def ready_report():
    return {
        "schema_version": 1, "results_ready": True,
        "checks": [
            {"id": key, "label": key, "status": "ready", "required_for_results": True}
            for key in ("python", "package", "command", "storage", "writable", "loopback")
        ] + [{"id": "gpu_policy", "label": "正式 GPU 未开放", "status": "blocked", "required_for_results": False}],
    }


class RemoteReadinessTests(unittest.TestCase):
    def call_with_report(self, report, code=0):
        calls = [subprocess.CompletedProcess([], 0, "command", ""),
                 subprocess.CompletedProcess([], code, json.dumps(report), "")]
        with mock.patch.object(controller.shutil, "which", return_value="ssh"), mock.patch.object(
            controller.subprocess, "run", side_effect=calls
        ) as run, mock.patch("sys.stdout", new_callable=io.StringIO):
            result = controller.check_remote_results_ready("lab-alias", remote_port=9876)
        self.assertEqual(run.call_count, 2)
        self.assertIn("check-readiness", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-2:], ["--port", "9876"])
        self.assertNotIn("shell", run.call_args.kwargs)
        return result

    def test_formal_gpu_block_does_not_block_readonly_results(self):
        self.assertTrue(self.call_with_report(ready_report())["results_ready"])

    def test_required_block_missing_duplicate_or_forged_success_is_rejected(self):
        original = ready_report()
        blocked = copy.deepcopy(original)
        blocked["checks"][3]["status"] = "blocked"
        missing = copy.deepcopy(original)
        missing["checks"].pop(4)
        duplicate = copy.deepcopy(original)
        duplicate["checks"].append(duplicate["checks"][0])
        for report in (blocked, missing, duplicate, {}, {"schema_version": 1, "checks": []}):
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                self.call_with_report(report)
        with self.assertRaises(RuntimeError):
            self.call_with_report(original, code=2)

    def test_timeout_or_invalid_json_never_opens_tunnel(self):
        for response in (subprocess.TimeoutExpired("ssh", 30), subprocess.CompletedProcess([], 0, "not-json", "")):
            with self.subTest(response=response), mock.patch.object(controller.shutil, "which", return_value="ssh"), mock.patch.object(
                controller.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0, "command", ""), response]
            ), mock.patch.object(controller.subprocess, "Popen") as start, mock.patch("sys.stdout", new_callable=io.StringIO):
                with self.assertRaises(RuntimeError):
                    controller.open_remote_results(host_alias="lab-alias", open_browser=False)
                start.assert_not_called()


class LocalReadinessTests(unittest.TestCase):
    def test_no_simulation_import_environment_change_or_file_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = DataRoot(Path(directory) / "data")
            root.initialize()
            config = Path(directory) / "missing-config.json"
            before = sorted(str(p) for p in Path(directory).rglob("*"))
            script = (
                "import json, os, sys; from pathlib import Path; "
                "from experiment_runner.readiness import check_readiness; "
                "before=dict(os.environ); result=check_readiness(Path(sys.argv[1]),config_path=Path(sys.argv[2])); "
                "assert dict(os.environ)==before; "
                "assert not any(n.split('.')[0] in {'warp','newton','mujoco','mujoco_warp'} for n in sys.modules); "
                "print(json.dumps(result))"
            )
            completed = subprocess.run([sys.executable, "-B", "-c", script, str(root.path), str(config)],
                                       check=True, capture_output=True, text=True, timeout=40,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"}, encoding="utf-8")
            report = json.loads(completed.stdout)
            self.assertEqual(before, sorted(str(p) for p in Path(directory).rglob("*")))
            self.assertFalse(report["gpu_probed"])
            self.assertFalse(report["formal_gpu_enabled"])
            self.assertEqual({c["id"] for c in report["checks"]}, {
                "python", "package", "command", "dependencies", "source", "storage", "writable", "disk", "ffmpeg", "ffprobe", "loopback", "gpu_policy"
            })

    def test_unconfigured_storage_has_complete_blocked_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            report = check_readiness(Path(directory) / "missing", config_path=Path(directory) / "config")
        checks = {c["id"]: c for c in report["checks"]}
        self.assertFalse(report["results_ready"])
        self.assertEqual(checks["storage"]["status"], "blocked")
        self.assertEqual(checks["writable"]["status"], "blocked")
        self.assertEqual(checks["disk"]["status"], "blocked")

    def test_dependency_mismatch_blocks_gpu_without_loading_runtime(self):
        with mock.patch("experiment_runner.readiness.runtime_version_check", return_value={"matches": False}):
            with self.assertRaisesRegex(ValueError, "exact locked"):
                require_locked_runtime()
