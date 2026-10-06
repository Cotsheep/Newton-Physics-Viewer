"""Closed development cases for relative-height and fixed one-metre drop tests."""
from dataclasses import dataclass

DROP_VALIDATION_PROFILE = "mujoco-warp-cuda-drop-validation-10s-v1"


@dataclass(frozen=True)
class DropValidationCase:
    height: str
    label: str
    clearance_scale: float | None
    fixed_clearance_m: float | None = None

    @property
    def case_id(self) -> str:
        return f"drop-{self.height}-validation"

    @property
    def case_scope(self) -> str:
        return ("fixed_1m_drop_validation_v1" if self.fixed_clearance_m is not None
                else "three_height_drop_validation_v1")


def drop_validation_case(height: str) -> DropValidationCase:
    cases = {
        "low": DropValidationCase("low", "低档摔落（开发验证）", 0.5),
        "medium": DropValidationCase("medium", "中档摔落（开发验证）", 1.0),
        "high": DropValidationCase("high", "高档摔落（开发验证）", 2.0),
        "fixed-1m": DropValidationCase("fixed-1m", "固定 1 米摔落（开发验证）", None, 1.0),
    }
    try:
        return cases[height]
    except (KeyError, TypeError) as exc:
        raise ValueError("Drop height must be low, medium, high, or fixed-1m") from exc
