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
    execution_tier: str
    recording_mode: str
    case_duration_seconds: float | None = None
    wall_time_limit_seconds: float | None = None

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
        result["solver"] = "newton.solvers.SolverMuJoCo"
        result["compute_backend"] = (
            "mujoco-cpu" if self.use_mujoco_cpu else "mujoco-warp-cuda"
        )
        result["cpu_fallback_allowed"] = False
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
        execution_tier="formal_reserved",
        recording_mode="required",
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
        execution_tier="development_smoke",
        recording_mode="required",
        case_duration_seconds=2.0,
    ),
    "mujoco-warp-cuda-dt1ms-integration-smoke-v1": ExperimentProfile(
        name="mujoco-warp-cuda-dt1ms-integration-smoke-v1",
        authoritative=False,
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        iterations=10,
        physics_dt=0.001,
        video_fps=50,
        video_width=640,
        video_height=360,
        preview_width=320,
        preview_height=180,
        preview_interval_seconds=1.0,
        execution_tier="development_integration_smoke",
        recording_mode="disabled_until_gpu_headless_rendering_is_validated",
        case_duration_seconds=1.0,
        wall_time_limit_seconds=300.0,
    ),
}

PROFILES = MappingProxyType(_PROFILE_VALUES)


_PROFILE_AVAILABILITY = MappingProxyType(
    {
        "mujoco-native-dt1ms-v1": {
            "status": "reserved_not_runnable",
            "runnable": False,
            "entrypoint": None,
            "entrypoints": (),
            "message": (
                "Registered for future formal GPU validation; no formal experiment "
                "execution entry point is currently available."
            ),
        },
        "mujoco-cpu-wsl-smoke-v1": {
            "status": "development_smoke_only",
            "runnable": True,
            # Compatibility field for existing consumers.  New consumers should
            # use ``entrypoints`` so every runnable smoke command is visible.
            "entrypoint": "smoke-drop",
            "entrypoints": ("smoke-drop", "smoke-slope"),
            "message": (
                "Available only for one non-authoritative MuJoCo CPU medium-height "
                "drop smoke case and one fixed 25-degree slope smoke case."
            ),
        },
        "mujoco-warp-cuda-dt1ms-integration-smoke-v1": {
            "status": "development_integration_smoke_only",
            "runnable": True,
            "entrypoint": "smoke-drop-gpu",
            "entrypoints": ("smoke-drop-gpu",),
            "message": (
                "Available only after the fail-closed Determined single-trial GPU gate for "
                "one bounded, non-authoritative medium-height drop integration smoke. "
                "GPU headless recording is not yet enabled."
            ),
        },
    }
)


def describe_profiles() -> dict[str, dict[str, Any]]:
    """Describe registered profiles without implying every profile is runnable."""

    descriptions: dict[str, dict[str, Any]] = {}
    for name, profile in PROFILES.items():
        availability = dict(_PROFILE_AVAILABILITY[name])
        availability["entrypoints"] = list(availability["entrypoints"])
        descriptions[name] = {
            **profile.expanded(),
            "availability": availability,
        }
    return descriptions


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown experiment profile {name!r}; expected one of: {choices}") from exc
