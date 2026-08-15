from __future__ import annotations

import json


UNAVAILABLE_EXIT = 77


def main() -> int:
    from experiment_runner.cpu_safety import (
        CpuSmokeEnvironmentUnavailable,
        prepare_cpu_smoke_environment,
    )

    prepare_cpu_smoke_environment()

    from experiment_runner.experiments.recording import _headless_viewer
    from experiment_runner.profiles import get_profile

    try:
        viewer, rendering = _headless_viewer(get_profile("mujoco-cpu-wsl-smoke-v1"))
    except CpuSmokeEnvironmentUnavailable as exc:
        print(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason_code": exc.reason_code,
                    "renderer": exc.renderer,
                    "vendor": exc.vendor,
                }
            )
        )
        return UNAVAILABLE_EXIT
    viewer.close()
    print(
        json.dumps(
            {
                "status": "available",
                "reason_code": None,
                "renderer": rendering["renderer"],
                "vendor": rendering["vendor"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
