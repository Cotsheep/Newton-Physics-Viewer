"""Read input-package parameters without guessing defaults or their provenance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .observations import finite_number


CONTACT_FIELDS = ("mjc:solref", "mjc:solimp", "mjc:condim", "mjc:priority")
MATERIAL_FIELDS = ("physics:dynamicFriction", "mjc:rollingfriction")


def _parameter(prim: Any, field: str, collision_path: str) -> dict[str, Any]:
    result = {"field": field, "collision_prim": collision_path,
              "status": "not_authored", "value": None, "source": "unknown"}
    if prim is None:
        result["status"] = "material_unbound"
        return result
    result["value_prim"] = str(prim.GetPath())
    attribute = prim.GetAttribute(field)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return result
    # Animated parameters are not a single constant; do not pick an arbitrary time.
    if attribute.ValueMightBeTimeVarying():
        result["status"] = "time_varying"
        return result
    value = attribute.Get()
    if finite_number(value):
        numeric = value
    else:
        try:
            numeric = list(value)
            if not numeric or not all(map(finite_number, numeric)):
                raise ValueError("Non-finite parameter")
        except (ValueError, TypeError):
            result["status"] = "invalid"
            return result
    result.update(status="recorded", value=numeric, source="authored_in_package")
    return result


def snapshot_physics_parameters(entrypoint: Path) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(entrypoint))
    if stage is None:
        raise ValueError("Cannot open asset for parameter snapshot")
    records = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute("physics:collisionEnabled").Get() is False:
            continue
        collision_path = str(prim.GetPath())
        records.extend(_parameter(prim, field, collision_path) for field in CONTACT_FIELDS)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        material_prim = material.GetPrim() if material else None
        if material_prim is not None and not material_prim.HasAPI(UsdPhysics.MaterialAPI):
            material_prim = None
        records.extend(_parameter(material_prim, field, collision_path) for field in MATERIAL_FIELDS)
    return {
        "schema_version": 1, "records": records,
        "provenance": "package_authorship_only",
        "solver_effective_values": "not_recorded",
    }
