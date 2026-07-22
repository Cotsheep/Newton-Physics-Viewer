"""Physics-solver selection and construction helpers."""

from __future__ import annotations

from typing import Any

import newton


SOLVER_NAMES = ("xpbd", "semi-implicit", "featherstone", "vbd", "mujoco")
SOLVER_LABELS = {
    "xpbd": "XPBD",
    "semi-implicit": "Semi-Implicit",
    "featherstone": "Featherstone",
    "vbd": "VBD (experimental)",
    "mujoco": "MuJoCo",
}


def prepare_builder_for_solver(builder: newton.ModelBuilder, solver_name: str) -> None:
    """Prepare builder data required by a selected solver before finalization."""
    if solver_name == "vbd":
        builder.color()


def create_solver(model: newton.Model, solver_name: str, iterations: int) -> Any:
    """Construct a supported rigid-body solver for an already finalized model."""
    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=iterations)
    if solver_name == "semi-implicit":
        return newton.solvers.SolverSemiImplicit(model)
    if solver_name == "featherstone":
        return newton.solvers.SolverFeatherstone(model)
    if solver_name == "vbd":
        return newton.solvers.SolverVBD(model, iterations=iterations)
    if solver_name == "mujoco":
        # Use MuJoCo Warp with Newton's contact pipeline so collision display,
        # dragging, and cross-solver comparisons all see the same contacts.
        return newton.solvers.SolverMuJoCo(
            model,
            iterations=iterations,
            use_mujoco_cpu=False,
            use_mujoco_contacts=False,
        )

    choices = ", ".join(SOLVER_NAMES)
    raise ValueError(f"Unknown solver {solver_name!r}; expected one of: {choices}")
