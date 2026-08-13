from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ExperimentProfile:
    """A versioned, closed set of simulation and recording parameters."""

    name: str
    authoritative: bool
    use_mujoco_cpu: bool
    use_mujoco_contacts: bool
    iterations: int
    physics_dt: float
    video_fps: int
    video_width: int
    video_height: int
    preview_width: int
    preview_height: int
    preview_interval_seconds: float
    case_duration_seconds: float | None = None

    @property
    def physics_steps_per_video_frame(self) -> int:
        steps = round((1.0 / self.video_fps) / self.physics_dt)
        if abs(steps * self.physics_dt - 1.0 / self.video_fps) > 1.0e-12:
            raise ValueError(
                f"Profile {self.name!r} does not have an integral number of physics steps per video frame"
            )
        return steps

    def expanded(self) -> dict[str, Any]:
        result = asdict(self)
        result["physics_steps_per_video_frame"] = self.physics_steps_per_video_frame
        result["video_codec"] = "h264"
        result["video_pixel_format"] = "yuv420p"
        result["video_crf"] = 18
        return result


_PROFILE_VALUES = {
    "mujoco-native-dt1ms-v1": ExperimentProfile(
        name="mujoco-native-dt1ms-v1",
        authoritative=True,
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        iterations=10,
        physics_dt=0.001,
        video_fps=50,
        video_width=1280,
        video_height=720,
        preview_width=640,
        preview_height=360,
        preview_interval_seconds=1.0,
    ),
    "mujoco-cpu-wsl-smoke-v1": ExperimentProfile(
        name="mujoco-cpu-wsl-smoke-v1",
        authoritative=False,
        use_mujoco_cpu=True,
        use_mujoco_contacts=True,
        iterations=10,
        physics_dt=0.001,
        video_fps=50,
        video_width=640,
        video_height=360,
        preview_width=320,
        preview_height=180,
        preview_interval_seconds=1.0,
        case_duration_seconds=2.0,
    ),
}

PROFILES = MappingProxyType(_PROFILE_VALUES)


_PROFILE_AVAILABILITY = MappingProxyType(
    {
        "mujoco-native-dt1ms-v1": {
            "status": "reserved_not_runnable",
            "runnable": False,
            "entrypoint": None,
            "message": (
                "Registered for future formal GPU validation; no formal experiment "
                "execution entry point is currently available."
            ),
        },
        "mujoco-cpu-wsl-smoke-v1": {
            "status": "development_smoke_only",
            "runnable": True,
            "entrypoint": "smoke-drop",
            "message": (
                "Available only for one non-authoritative MuJoCo CPU medium-height "
                "drop smoke case."
            ),
        },
    }
)


def describe_profiles() -> dict[str, dict[str, Any]]:
    """Describe registered profiles without implying every profile is runnable."""

    return {
        name: {
            **profile.expanded(),
            "availability": dict(_PROFILE_AVAILABILITY[name]),
        }
        for name, profile in PROFILES.items()
    }


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown experiment profile {name!r}; expected one of: {choices}") from exc
