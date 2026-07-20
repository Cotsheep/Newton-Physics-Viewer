from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from third_party.artiverse_official import run_view_model


class OfficialViewerAdapterTests(unittest.TestCase):
    def test_resolve_model_inputs_accepts_prerelease_articulation_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model-123"
            model_dir.mkdir()
            (model_dir / "model-123.segmented.glb").write_bytes(b"glb")
            uncorrected = model_dir / "model-123.articulations.json"
            uncorrected.write_text('{"articulations": []}', encoding="utf-8")

            inputs = run_view_model.resolve_model_inputs(model_dir)

            self.assertEqual(inputs.model_id, "model-123")
            self.assertEqual(inputs.articulations_json, uncorrected)
            self.assertTrue(inputs.uses_uncorrected_fallback)

            with run_view_model.staged_official_input(inputs) as (staged_model, staged_output):
                self.assertTrue((staged_model / "model-123.segmented.glb").is_file())
                self.assertTrue(
                    (staged_model / "model-123.corrected.articulations.json").is_file()
                )
                self.assertFalse(staged_output.exists())

    def test_official_viewer_files_match_recorded_snapshot(self) -> None:
        expected = {
            "view_model.py": "2393a89ff20fbec1ce5a9cf31591431716af3ea195dae0f59272b3f5bd009d14",
            "utils/render_part_articulation.py": (
                "5ba4b3d1aabab3d0c700f0ea72a80ef225705b922c01f667b8f9a87a718a5c20"
            ),
            "utils/brown_photostudio_01_4k.exr": (
                "618c919371207cfac00e2256e9cb87d8f6886182167aba82521854178ff1da03"
            ),
        }
        for relative_path, expected_hash in expected.items():
            path = run_view_model.UPSTREAM_ROOT / relative_path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_hash, relative_path)


if __name__ == "__main__":
    unittest.main()
