"""Streaming, sampled geometric observations; no physical quality thresholds."""
from __future__ import annotations

import math
from typing import Any


DROP_DEFINITION = "whole-asset-plane-clearance-first-flight-v1"
DROP_KEYS = ("first_rebound_height", "first_rebound_ratio", "maximum_ground_penetration")
CONTACT_TOLERANCE_M = 1e-5


class DropAccumulator:
    def __init__(self, physics_dt: float):
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("Invalid sampling interval")
        self.dt = physics_dt
        self.count = 0
        self.invalid = False
        self.initial = None
        self.end = 0.0
        self.contact = None
        self.liftoff = None
        self.landing = None
        self.peak = 0.0
        self.peak_time = None
        self.penetration = 0.0
        self.penetration_time = 0.0

    def add(self, time_seconds: float, clearance: float) -> None:
        if (not math.isfinite(time_seconds) or not math.isfinite(clearance)
                or not math.isclose(time_seconds, self.count * self.dt, rel_tol=0, abs_tol=1e-8)):
            self.invalid = True
            raise ValueError("Missing, misaligned or invalid sample")
        if self.count == 0:
            self.initial = clearance
            if clearance <= CONTACT_TOLERANCE_M:
                self.invalid = True
        self.count += 1
        self.end = time_seconds
        depth = max(0.0, -clearance)
        if depth > self.penetration:
            self.penetration, self.penetration_time = depth, time_seconds
        if self.contact is None:
            if clearance <= CONTACT_TOLERANCE_M:
                self.contact = time_seconds
        elif self.landing is None:
            if self.liftoff is None and clearance > CONTACT_TOLERANCE_M:
                self.liftoff = time_seconds
            if self.liftoff is not None:
                if clearance <= CONTACT_TOLERANCE_M:
                    self.landing = time_seconds
                elif clearance > self.peak:
                    self.peak, self.peak_time = clearance, time_seconds

    def result(self, duration: float) -> dict[str, Any]:
        common = {
            "definition": DROP_DEFINITION, "sampling": "initial_and_every_completed_physics_step",
            "sample_interval_seconds": self.dt, "sample_count": self.count,
            "start_time_seconds": 0.0, "end_time_seconds": self.end,
            "initial_clearance_m": self.initial, "contact_tolerance_m": CONTACT_TOLERANCE_M,
            "first_contact_time_seconds": self.contact, "liftoff_time_seconds": self.liftoff,
            "landing_time_seconds": self.landing,
        }
        complete = math.isclose(self.end, duration, rel_tol=0, abs_tol=1e-8) and self.count > 1
        status = "invalid" if self.invalid else "incomplete" if not complete else None
        rebound_status = status or (
            "not_observed" if self.contact is None else
            "no_liftoff" if self.liftoff is None else
            "incomplete" if self.landing is None else "measured"
        )
        height = self.peak if rebound_status == "measured" else None
        return {
            "first_rebound_height": {**common, "status": rebound_status, "value": height,
                                     "unit": "m", "peak_time_seconds": self.peak_time},
            "first_rebound_ratio": {**common, "status": rebound_status,
                                    "value": height / self.initial if height is not None else None, "unit": "1"},
            "maximum_ground_penetration": {**common, "status": status or "measured",
                                           "value": None if status else self.penetration, "unit": "m",
                                           "peak_time_seconds": self.penetration_time},
        }


def public_drop_observations(document: dict[str, Any]) -> dict[str, Any]:
    """Validate the versioned summary; never synthesize observations from bounds."""
    def blank(status):
        return {key: {"status": status, "value": None, "unit": "1" if key.endswith("ratio") else "m"}
                for key in DROP_KEYS}

    if document.get("finite") is False:
        return blank("invalid")
    if document.get("status") != "succeeded":
        return blank("not_started" if document.get("status") == "created" else "incomplete")
    observations = document.get("observations")
    if not isinstance(observations, dict) or not any(key in observations for key in DROP_KEYS):
        return blank("not_measured")
    if document.get("finite") is not True:
        return blank("unverified")

    def number(value):
        try:
            return type(value) in (float, int) and math.isfinite(value)
        except OverflowError:
            return False

    try:
        items = [observations[key] for key in DROP_KEYS]
        if not all(isinstance(item, dict) for item in items):
            raise ValueError("Invalid observations")
        if all(item.get("status") == "unsupported" and item.get("value") is None for item in items):
            return blank("unsupported")
        height, ratio, depth = items
        if any(item.get("status") == "invalid" for item in items):
            return blank("invalid")
        shared = ("definition", "sampling", "sample_interval_seconds", "sample_count",
                  "start_time_seconds", "end_time_seconds", "initial_clearance_m", "contact_tolerance_m",
                  "first_contact_time_seconds", "liftoff_time_seconds", "landing_time_seconds")
        if any(any(item.get(key) != height.get(key) for key in shared) for item in items):
            raise ValueError("Inconsistent observation windows")
        if height["definition"] != DROP_DEFINITION or height["sampling"] != "initial_and_every_completed_physics_step":
            raise ValueError("Unknown measurement definition")
        dt, end, initial = (height[key] for key in ("sample_interval_seconds", "end_time_seconds", "initial_clearance_m"))
        count, tolerance = height["sample_count"], height["contact_tolerance_m"]
        if (not all(number(value) for value in (dt, end, initial, tolerance)) or dt <= 0
                or tolerance != CONTACT_TOLERANCE_M or initial <= tolerance
                or type(count) is not int or count < 2 or height["start_time_seconds"] != 0
                or not math.isclose(end, (count - 1) * dt, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Invalid sampling window")
        duration = document.get("duration_seconds")
        if not number(duration) or not math.isclose(end, duration, rel_tol=0, abs_tol=1e-8):
            return blank("incomplete")
        if "physics_steps" in document and document["physics_steps"] != count - 1:
            raise ValueError("Step count disagrees with sampling")
        if (height.get("unit") != "m" or ratio.get("unit") != "1" or depth.get("unit") != "m"
                or ratio.get("status") != height.get("status") or depth.get("status") != "measured"
                or not number(depth.get("value")) or depth["value"] < 0):
            raise ValueError("Invalid metric")
        contact, liftoff, landing = (height[key] for key in (
            "first_contact_time_seconds", "liftoff_time_seconds", "landing_time_seconds"))
        previous = -1.0
        for event in (contact, liftoff, landing):
            if event is not None:
                if not number(event) or not previous < event <= end:
                    raise ValueError("Invalid event order")
                previous = event
        for item in (height, depth):
            peak_time = item.get("peak_time_seconds")
            if peak_time is not None and (not number(peak_time) or not 0 <= peak_time <= end):
                raise ValueError("Invalid peak time")
        if not number(depth.get("peak_time_seconds")):
            raise ValueError("Missing penetration sample time")
        status = height.get("status")
        if status == "measured":
            if (None in (contact, liftoff, landing) or not number(height.get("value"))
                    or height["value"] <= tolerance or not number(ratio.get("value"))
                    or not number(height.get("peak_time_seconds"))
                    or not liftoff <= height["peak_time_seconds"] < landing
                    or not math.isclose(ratio["value"], height["value"] / initial, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError("Invalid first flight")
        elif status == "not_observed":
            if any(event is not None for event in (contact, liftoff, landing)):
                raise ValueError("Unexpected contact")
        elif status == "no_liftoff":
            if contact is None or liftoff is not None or landing is not None:
                raise ValueError("Unexpected liftoff")
        elif status == "incomplete":
            if contact is None or liftoff is None or landing is not None:
                raise ValueError("No incomplete flight")
        else:
            raise ValueError("Unknown metric status")
        if status != "measured" and (height.get("value") is not None or ratio.get("value") is not None):
            raise ValueError("Incomplete observation must not have a final value")
        result = {}
        for key, item in zip(DROP_KEYS, items):
            fields = (*shared, "status", "value", "unit", "peak_time_seconds")
            result[key] = {field: item[field] for field in fields if field in item}
        return result
    except (ValueError, KeyError, TypeError, OverflowError):
        return blank("invalid")
