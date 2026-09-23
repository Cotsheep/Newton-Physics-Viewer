"""Real USD import checks without simulation, rendering, or GPU execution."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import newton
import numpy as np
import warp as wp

from asset_viewer.app import build_model, parse_args
from experiment_runner.experiments.drop import mujoco_usd_schema_resolvers
from experiment_runner.experiments.slope import create_slope_scene, measure_slope_geometry
from experiment_runner.profiles import get_profile


FIXTURE = Path(__file__).parent / "fixtures/smoke_asset/newton-mujoco.usda"


class UsdMuJoCoImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.asset = Path(self.temporary.name) / "newton-mujoco.usda"
        content = FIXTURE.read_text(encoding="utf-8")
        content = content.replace("[0.02, 1.0]", "[0.04, 0.8]")
        content = content.replace("[0.9, 0.95, 0.001, 0.5, 2.0]", "[0.7, 0.85, 0.003, 0.4, 3.0]")
        self.asset.write_text(content, encoding="utf-8")

    def assert_authored_contact_parameters(self, model: newton.Model) -> None:
        np.testing.assert_allclose(model.mujoco.solref.numpy()[0], [0.04, 0.8], rtol=1e-6)
        np.testing.assert_allclose(model.mujoco.geom_solimp.numpy()[0], [0.7, 0.85, 0.003, 0.4, 3.0], rtol=1e-6)

    def test_shared_drop_viewer_import_preserves_authored_contact_parameters(self) -> None:
        args = parse_args([str(self.asset), "--no-ground", "--z", "0", "--solver", "mujoco"])
        with wp.ScopedDevice("cpu"):
            model, _state, _panel = build_model(
                args, self.asset, usd_schema_resolvers=mujoco_usd_schema_resolvers()
            )
        self.assertTrue(model.device.is_cpu)
        self.assert_authored_contact_parameters(model)

    def test_direct_slope_import_preserves_authored_contact_parameters(self) -> None:
        profile = get_profile("mujoco-cpu-wsl-smoke-v1")
        register = newton.solvers.SolverMuJoCo.register_custom_attributes
        with wp.ScopedDevice("cpu"):
            geometry = measure_slope_geometry(self.asset, profile=profile, angle_degrees=25.)
            # Use the real importer/model, but do not construct or run a solver.
            with mock.patch("experiment_runner.experiments.slope.newton.solvers.SolverMuJoCo") as solver:
                solver.register_custom_attributes.side_effect = register
                scene = create_slope_scene(self.asset, profile=profile, geometry=geometry)
        self.assertTrue(scene.model.device.is_cpu)
        self.assert_authored_contact_parameters(scene.model)
