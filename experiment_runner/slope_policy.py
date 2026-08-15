from __future__ import annotations


def single_body_slope_smoke_error(rigid_body_count: object) -> str:
    """Describe the temporary local-smoke boundary without narrowing readiness."""

    detected = str(rigid_body_count) if type(rigid_body_count) is int else "unknown"
    return (
        "Current fixed 25-degree local CPU slope smoke supports exactly one dynamic "
        "rigid body; the asset may still pass generic slope_friction readiness; this "
        "is not a permanent limit on future formal slope experiments; "
        f"detected {detected}."
    )
