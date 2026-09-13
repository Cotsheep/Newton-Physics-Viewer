from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any, Callable, Mapping, Protocol


LOGICAL_CUDA_DEVICE = "cuda:0"
UNKNOWN = "unknown"
_GPU_UUID_PATTERN = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_DETERMINED_REQUIRED_IDS = (
    "DET_EXPERIMENT_ID",
    "DET_TRIAL_ID",
    "DET_TASK_ID",
    "DET_ALLOCATION_ID",
)


class GpuSmokeSafetyError(RuntimeError):
    """The externally allocated GPU boundary is absent or unusable."""


class CudaRuntime(Protocol):
    def visible_device_count(self) -> int: ...

    def initialize_logical_device(self, logical_device: str) -> Any: ...

    def driver_version(self) -> Any: ...

    def toolkit_version(self) -> Any: ...

    def nvidia_driver_version(self) -> Any: ...


class WarpCudaRuntime:
    """Small adapter kept lazy so unit tests never initialize a real CUDA runtime."""

    def __init__(self, permit: GpuExecutionPermit) -> None:
        self.permit = permit

    def _warp(self) -> Any:
        require_gpu_execution_permit(self.permit)
        import warp as wp

        return wp

    def visible_device_count(self) -> int:
        return int(self._warp().get_cuda_device_count())

    def initialize_logical_device(self, logical_device: str) -> Any:
        if logical_device != LOGICAL_CUDA_DEVICE:
            raise GpuSmokeSafetyError("Only container-logical cuda:0 is permitted")
        wp = self._warp()
        wp.set_device(logical_device)
        device = wp.get_device()
        # A tiny allocation forces CUDA context creation.  Any driver or context
        # failure must escape; this path never tries a CPU device.
        probe = wp.empty(shape=1, dtype=wp.uint8, device=device)
        wp.synchronize_device(device)
        del probe
        return device

    def driver_version(self) -> Any:
        return self._warp().get_cuda_driver_version()

    def toolkit_version(self) -> Any:
        return self._warp().get_cuda_toolkit_version()

    def nvidia_driver_version(self) -> Any:
        """Read the mounted driver metadata without spawning nvidia-smi."""

        try:
            with open(
                "/proc/driver/nvidia/version",
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                content = handle.read(4096)
        except OSError:
            return None
        match = re.search(r"Kernel Module\s+([0-9][0-9.]*)", content)
        return match.group(1) if match else None


def _known(value: Any) -> Any:
    if value is None or value == "":
        return UNKNOWN
    return value


def _version(value: Any) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        return f"{value[0]}.{value[1]}"
    return str(_known(value))


def _warp_version() -> str:
    try:
        return metadata.version("warp-lang")
    except metadata.PackageNotFoundError:
        return UNKNOWN


@dataclass(frozen=True)
class GpuAudit:
    logical_device: str
    visible_gpu_count: int
    model: str
    uuid: str
    nvidia_driver_version: str
    cuda_driver_api_version: str
    cuda_toolkit_version: str
    warp_version: str
    nvidia_visible_devices: str
    cuda_visible_devices: str
    uuid_verification_status: str
    uuid_warning: str | None

    def expanded(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeterminedTrialAllocation:
    """Operational evidence that this process is inside one Determined trial slot.

    These environment values can be forged by a process and are therefore a misuse
    guard, not authentication.  Real isolation remains the responsibility of the
    Determined scheduler and NVIDIA container runtime.
    """

    experiment_id: str
    trial_id: str
    task_id: str
    allocation_id: str
    slot_id: int
    nvidia_visible_device_uuid: str
    task_type: str

    def expanded(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CudaVisibility:
    runtime: CudaRuntime
    visible_gpu_count: int
    nvidia_visible_devices: str
    cuda_visible_devices: str
    permit: GpuExecutionPermit


_PERMIT_SEAL = object()


@dataclass(frozen=True, init=False)
class GpuExecutionPermit:
    """Process-local misuse capability; not authentication or resource isolation."""

    allocation: DeterminedTrialAllocation
    process_id: int
    _seal: object = field(repr=False)


def issue_gpu_execution_permit(
    environment: Mapping[str, str] | None = None,
) -> GpuExecutionPermit:
    allocation = require_determined_trial_allocation(environment)
    permit = object.__new__(GpuExecutionPermit)
    object.__setattr__(permit, "allocation", allocation)
    object.__setattr__(permit, "process_id", os.getpid())
    object.__setattr__(permit, "_seal", _PERMIT_SEAL)
    return permit


def require_gpu_execution_permit(permit: GpuExecutionPermit | None, profile: Any = None) -> None:
    if (
        type(permit) is not GpuExecutionPermit
        or getattr(permit, "_seal", None) is not _PERMIT_SEAL
        or getattr(permit, "process_id", None) != os.getpid()
    ):
        raise GpuSmokeSafetyError("A gate-issued GPU execution permit is required before CUDA")
    if profile is not None:
        from .profiles import get_profile

        if profile != get_profile("mujoco-warp-cuda-dt1ms-integration-smoke-v1"):
            raise GpuSmokeSafetyError("Only the fixed GPU integration smoke profile is permitted")


def require_cpu_smoke_profile(profile: Any) -> None:
    if profile.authoritative or not profile.use_mujoco_cpu:
        raise GpuSmokeSafetyError("This function accepts only non-authoritative CPU smoke profiles")


def normalize_gpu_uuid(value: str) -> str:
    text = value.strip()
    if text.lower().startswith("gpu-"):
        text = "GPU-" + text[4:]
    else:
        text = "GPU-" + text
    if not _GPU_UUID_PATTERN.fullmatch(text):
        raise GpuSmokeSafetyError("Runtime GPU UUID has an invalid format")
    return "GPU-" + text[4:].lower()


def _required_environment_value(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key, "").strip()
    if not value or value.casefold() == UNKNOWN:
        raise GpuSmokeSafetyError(
            f"GPU integration smoke requires non-empty Determined trial metadata: {key}"
        )
    return value


def require_determined_trial_allocation(
    environment: Mapping[str, str] | None = None,
) -> DeterminedTrialAllocation:
    """Fail closed unless Determined and NVIDIA expose exactly one trial GPU slot."""

    selected_environment = os.environ if environment is None else environment
    values = {
        key: _required_environment_value(selected_environment, key)
        for key in _DETERMINED_REQUIRED_IDS
    }
    task_type = _required_environment_value(selected_environment, "DET_TASK_TYPE")
    if task_type != "TRIAL":
        raise GpuSmokeSafetyError(
            "GPU integration smoke requires DET_TASK_TYPE=TRIAL"
        )

    raw_slot_ids = _required_environment_value(selected_environment, "DET_SLOT_IDS")
    try:
        slot_ids = json.loads(raw_slot_ids)
    except (TypeError, ValueError) as exc:
        raise GpuSmokeSafetyError(
            "DET_SLOT_IDS must be a JSON list containing exactly one allocated slot"
        ) from exc
    if not isinstance(slot_ids, list) or len(slot_ids) != 1:
        raise GpuSmokeSafetyError(
            "DET_SLOT_IDS must describe exactly one allocated Determined slot"
        )
    slot_id = slot_ids[0]
    if type(slot_id) is not int or slot_id < 0:
        raise GpuSmokeSafetyError("DET_SLOT_IDS must contain one non-negative integer")

    nvidia_visible = _required_environment_value(
        selected_environment, "NVIDIA_VISIBLE_DEVICES"
    )
    visible_tokens = [token.strip() for token in nvidia_visible.split(",")]
    if len(visible_tokens) != 1 or not visible_tokens[0]:
        raise GpuSmokeSafetyError(
            "NVIDIA_VISIBLE_DEVICES must contain exactly one allocated GPU UUID"
        )
    visible_uuid = visible_tokens[0]
    if visible_uuid.casefold() in {"all", "none", "void"}:
        raise GpuSmokeSafetyError(
            "NVIDIA_VISIBLE_DEVICES must name one allocated GPU UUID, not a visibility sentinel"
        )
    if visible_uuid.isdecimal() or not _GPU_UUID_PATTERN.fullmatch(visible_uuid):
        raise GpuSmokeSafetyError(
            "NVIDIA_VISIBLE_DEVICES must be one canonical GPU UUID; host GPU indices are forbidden"
        )

    return DeterminedTrialAllocation(
        experiment_id=values["DET_EXPERIMENT_ID"],
        trial_id=values["DET_TRIAL_ID"],
        task_id=values["DET_TASK_ID"],
        allocation_id=values["DET_ALLOCATION_ID"],
        slot_id=slot_id,
        nvidia_visible_device_uuid=normalize_gpu_uuid(visible_uuid),
        task_type=task_type.upper(),
    )


def require_single_warp_visible_gpu(
    runtime: CudaRuntime | None = None,
    environment: Mapping[str, str] | None = None,
    *,
    permit: GpuExecutionPermit | None = None,
) -> CudaVisibility:
    """Discover one process-visible Warp GPU without creating a CUDA context."""

    require_gpu_execution_permit(permit)
    selected_environment = os.environ if environment is None else environment
    if require_determined_trial_allocation(selected_environment) != permit.allocation:
        raise GpuSmokeSafetyError("Determined allocation changed after the GPU permit was issued")
    cuda_visible = selected_environment.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() in {"", "-1"}:
        raise GpuSmokeSafetyError(
            "GPU integration smoke cannot run with CUDA hidden; the CPU smoke safety "
            "environment must not be reused"
        )

    selected_runtime = runtime or WarpCudaRuntime(permit)
    try:
        count = selected_runtime.visible_device_count()
    except Exception as exc:
        raise GpuSmokeSafetyError(
            f"CUDA discovery failed ({type(exc).__name__}); CPU fallback is forbidden"
        ) from exc
    if type(count) is not int or count != 1:
        raise GpuSmokeSafetyError(
            "GPU integration smoke requires exactly one process-visible CUDA GPU; "
            f"found {count}. Device selection must be provided by the container scheduler."
        )

    return CudaVisibility(
        runtime=selected_runtime,
        visible_gpu_count=count,
        nvidia_visible_devices=str(
            _known(selected_environment.get("NVIDIA_VISIBLE_DEVICES"))
        ),
        cuda_visible_devices=str(_known(cuda_visible)),
        permit=permit,
    )


def initialize_logical_cuda_gpu(
    visibility: CudaVisibility, *, on_audit: Callable[..., None] | None = None,
) -> GpuAudit:
    """Initialize only container-logical cuda:0; never try a CPU or host index."""

    require_gpu_execution_permit(visibility.permit)
    try:
        device = visibility.runtime.initialize_logical_device(LOGICAL_CUDA_DEVICE)
    except Exception as exc:
        raise GpuSmokeSafetyError(
            f"CUDA initialization of logical device {LOGICAL_CUDA_DEVICE} failed "
            f"({type(exc).__name__}); CPU fallback is forbidden"
        ) from exc

    if on_audit is not None:
        on_audit(cuda_context_initialized=True)
    alias = str(getattr(device, "alias", ""))
    ordinal = getattr(device, "ordinal", None)
    is_cuda = bool(getattr(device, "is_cuda", False))
    if alias != LOGICAL_CUDA_DEVICE or ordinal != 0 or not is_cuda:
        raise GpuSmokeSafetyError(
            "CUDA initialized an unexpected device; expected the container-logical cuda:0 "
            "and refusing CPU or host-index fallback"
        )

    try:
        driver = visibility.runtime.driver_version()
    except Exception:
        driver = None
    try:
        toolkit = visibility.runtime.toolkit_version()
    except Exception:
        toolkit = None
    try:
        nvidia_driver = visibility.runtime.nvidia_driver_version()
    except Exception:
        nvidia_driver = None
    runtime_uuid = getattr(device, "uuid", None)
    missing_uuid = runtime_uuid is None or runtime_uuid == ""
    if not missing_uuid:
        try:
            if not isinstance(runtime_uuid, str):
                raise GpuSmokeSafetyError("Runtime GPU UUID must be text")
            runtime_uuid = normalize_gpu_uuid(runtime_uuid)
        except GpuSmokeSafetyError:
            if on_audit is not None:
                on_audit(uuid_verification_status="invalid", runtime_gpu_uuid=str(runtime_uuid))
            raise
        if runtime_uuid != visibility.permit.allocation.nvidia_visible_device_uuid:
            if on_audit is not None:
                on_audit(uuid_verification_status="mismatched", runtime_gpu_uuid=runtime_uuid)
            raise GpuSmokeSafetyError("Runtime GPU UUID does not match the allocated GPU UUID")
    return GpuAudit(
        logical_device=LOGICAL_CUDA_DEVICE,
        visible_gpu_count=visibility.visible_gpu_count,
        model=str(_known(getattr(device, "name", None))),
        uuid=UNKNOWN if missing_uuid else runtime_uuid,
        nvidia_driver_version=str(_known(nvidia_driver)),
        cuda_driver_api_version=_version(driver),
        cuda_toolkit_version=_version(toolkit),
        warp_version=_warp_version(),
        nvidia_visible_devices=visibility.nvidia_visible_devices,
        cuda_visible_devices=visibility.cuda_visible_devices,
        uuid_verification_status="unavailable" if missing_uuid else "matched",
        uuid_warning="Warp did not expose a GPU UUID" if missing_uuid else None,
    )


def require_single_visible_cuda_gpu(
    runtime: CudaRuntime | None = None,
    environment: Mapping[str, str] | None = None,
) -> GpuAudit:
    """Compatibility wrapper for the full fail-closed allocation and CUDA gate."""

    permit = issue_gpu_execution_permit(environment)
    visibility = require_single_warp_visible_gpu(runtime, environment, permit=permit)
    return initialize_logical_cuda_gpu(visibility)


def require_scene_on_logical_gpu(scene: Any) -> str:
    """Prove model arrays were built on the selected logical CUDA device."""

    device = getattr(getattr(scene, "model", None), "device", None)
    alias = str(getattr(device, "alias", device or ""))
    ordinal = getattr(device, "ordinal", None)
    is_cuda = bool(getattr(device, "is_cuda", False))
    if alias != LOGICAL_CUDA_DEVICE or ordinal != 0 or not is_cuda:
        raise GpuSmokeSafetyError(
            "The Newton model is not on container-logical cuda:0; CPU fallback is forbidden"
        )
    solver = getattr(scene, "solver", None)
    if getattr(solver, "use_mujoco_cpu", None) is not False:
        raise GpuSmokeSafetyError(
            "SolverMuJoCo is not explicitly configured for MJWarp CUDA; "
            "CPU fallback is forbidden"
        )
    return alias
