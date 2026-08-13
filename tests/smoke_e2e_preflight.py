from __future__ import annotations

import json


def main() -> int:
    from experiment_runner.cpu_safety import prepare_cpu_smoke_environment

    prepare_cpu_smoke_environment()

    from experiment_runner.experiments.recording import _headless_viewer
    from experiment_runner.profiles import get_profile

    viewer, rendering = _headless_viewer(get_profile("mujoco-cpu-wsl-smoke-v1"))
    viewer.close()
    print(json.dumps(rendering))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
