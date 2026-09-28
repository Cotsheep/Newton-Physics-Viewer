"""Closed development cases for validating the planned three-height drop test."""
from dataclasses import dataclass

DROP_VALIDATION_PROFILE = "mujoco-warp-cuda-drop-validation-10s-v1"


@dataclass(frozen=True)
class DropValidationCase:
    height: str
    label: str
    clearance_scale: float

    @property
    def case_id(self) -> str:
        return f"drop-{self.height}-validation"


def drop_validation_case(height: str) -> DropValidationCase:
    cases = {
        "low": DropValidationCase("low", "低档摔落（开发验证）", 0.5),
        "medium": DropValidationCase("medium", "中档摔落（开发验证）", 1.0),
        "high": DropValidationCase("high", "高档摔落（开发验证）", 2.0),
    }
    try:
        return cases[height]
    except (KeyError, TypeError) as exc:
        raise ValueError("Drop height must be low, medium, or high") from exc
