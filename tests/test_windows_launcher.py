from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = PROJECT_ROOT / "newton-test.cmd"


class WindowsLauncherTests(unittest.TestCase):
    def test_launcher_uses_crlf_line_endings(self) -> None:
        content = LAUNCHER.read_bytes()
        self.assertNotIn(
            b"\n",
            content.replace(b"\r\n", b""),
            "Windows cmd launchers must use CRLF line endings",
        )

    @unittest.skipUnless(os.name == "nt", "requires Windows cmd.exe")
    def test_launcher_invokes_uv_without_cmd_parse_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stub = Path(temporary_directory) / "uv.cmd"
            stub.write_text(
                "@echo off\r\n"
                'if "%~1"=="run" if "%~2"=="--frozen" '
                'if "%~3"=="newton-test" exit /b 0\r\n'
                "exit /b 64\r\n",
                encoding="ascii",
                newline="",
            )
            environment = os.environ.copy()
            environment["PATH"] = (
                f"{temporary_directory}{os.pathsep}{environment.get('PATH', '')}"
            )

            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", LAUNCHER.name],
                cwd=PROJECT_ROOT,
                env=environment,
                input="0\r\n" * 8,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=5,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )

    def test_result_launcher_uses_crlf(self) -> None:
        content = (PROJECT_ROOT / "open-server-results.cmd").read_bytes()
        self.assertNotIn(b"\n", content.replace(b"\r\n", b""))

    @unittest.skipUnless(os.name == "nt", "requires Windows cmd.exe")
    def test_result_launcher_locates_project_from_another_working_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="Newton launcher ") as temporary:
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(PROJECT_ROOT / "open-server-results.cmd"), "--help"],
                cwd=temporary, capture_output=True, encoding="utf-8", errors="replace", timeout=10,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("server-results", completed.stdout)
        self.assertIn("controller.toml", completed.stdout)

    @unittest.skipUnless(os.name == "nt", "requires Windows cmd.exe")
    def test_result_launcher_missing_environment_preserves_failure_exit_code(self) -> None:
        with tempfile.TemporaryDirectory(prefix="Newton missing env ") as temporary:
            launcher = Path(temporary) / "open-server-results.cmd"
            launcher.write_bytes((PROJECT_ROOT / launcher.name).read_bytes())
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(launcher)], cwd=PROJECT_ROOT,
                input="\n", capture_output=True, encoding="utf-8", errors="replace", timeout=10,
            )
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("找不到", completed.stdout)


if __name__ == "__main__":
    unittest.main()
