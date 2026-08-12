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


if __name__ == "__main__":
    unittest.main()
