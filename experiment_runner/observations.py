"""Small, versioned observations; these never grade an asset's physics."""
from __future__ import annotations

import math
from typing import Any


SLOPE_DEFINITION = "single-body-origin-downhill-displacement-v1"


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def slope_displacement_observation(
    initial_position: Any, final_position: Any, down_slope: Any,
    *, body_index: int, body_count: int, end_time_seconds: float,
) -> dict[str, Any]:
    """Signed endpoint displacement of one body origin, not COM or path length."""
    if type(body_count) is not int or type(body_index) is not int or body_count != 1 or body_index != 0:
        raise ValueError("This measurement definition requires exactly one body")
    vectors = [list(map(float, vector)) for vector in (initial_position, final_position, down_slope)]
    if any(len(vector) != 3 or not all(map(finite_number, vector)) for vector in vectors):
        raise ValueError("Slope measurement requires finite 3D vectors")
    initial, final, direction = vectors
    if not math.isclose(sum(x * x for x in direction), 1.0, abs_tol=1e-10):
        raise ValueError("Downhill direction must be a unit vector")
    if not finite_number(end_time_seconds) or end_time_seconds <= 0:
        raise ValueError("Observation time must be finite and positive")
    value = sum((b - a) * d for a, b, d in zip(initial, final, direction))
    if not math.isfinite(value):
        raise ValueError("Slope displacement overflowed")
    return {
        "status": "measured", "definition": SLOPE_DEFINITION,
        "value": value, "unit": "m", "body_index": body_index,
        "body_count": body_count, "reference_point": "body_frame_origin",
        "direction_world": direction, "initial_position_m": initial,
        "final_position_m": final, "start_time_seconds": 0.0,
        "end_time_seconds": end_time_seconds, "sampling": "initial_and_final_state",
    }


def case_observations(document: dict[str, Any], template: str) -> dict[str, Any]:
    """Publish supported observations without filling gaps in historical results."""
    if template == "drop":
        from .drop_metrics import public_drop_observations

        return public_drop_observations(document)
    keys = ("along_slope_displacement",)
    status = document.get("status")
    unavailable = (
        "invalid" if document.get("finite") is False else
        "not_started" if status == "created" else
        "incomplete" if status != "succeeded" else "not_measured"
    )
    metrics = {key: {"status": unavailable, "value": None, "unit": "m"} for key in keys}
    if template != "slope_friction" or unavailable != "not_measured":
        return metrics
    if document.get("finite") is not True:
        metrics["along_slope_displacement"]["status"] = "unverified"
        return metrics

    recorded = document.get("observations")
    measurement = recorded.get("along_slope_displacement") if isinstance(recorded, dict) else None
    if measurement is not None:
        try:
            if not isinstance(measurement, dict) or measurement.get("definition") != SLOPE_DEFINITION:
                raise ValueError("Unknown measurement definition")
            validated = slope_displacement_observation(
                measurement["initial_position_m"], measurement["final_position_m"],
                measurement["direction_world"], body_index=measurement["body_index"],
                body_count=measurement["body_count"], end_time_seconds=measurement["end_time_seconds"],
            )
            if any(measurement.get(key) != validated[key] for key in (
                "status", "unit", "reference_point", "sampling", "start_time_seconds",
            )) or not finite_number(measurement.get("value")) or not math.isclose(
                measurement["value"], validated["value"], rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError("Inconsistent measurement")
        except (KeyError, TypeError, ValueError, OverflowError):
            metrics["along_slope_displacement"]["status"] = "invalid"
        else:
            # Preserve the recorded number; validation does not rewrite the run.
            validated["value"] = measurement["value"]
            metrics["along_slope_displacement"] = validated
        return metrics

    value = document.get("displacement_along_slope")
    duration = document.get("duration_seconds")
    if finite_number(value) and finite_number(duration) and duration > 0:
        metrics["along_slope_displacement"] = {
            "status": "legacy_record", "value": value, "unit": "m",
            "duration_seconds": duration,
        }
    return metrics
