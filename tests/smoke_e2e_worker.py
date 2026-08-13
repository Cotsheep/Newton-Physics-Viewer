from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", choices=("drop", "slope_friction"))
    parser.add_argument("data_root", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("summary_path", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # This must run before importing any module that imports Newton, Warp, or
    # OpenGL.  The parent test also sets these variables defensively.
    from experiment_runner.cpu_safety import prepare_cpu_smoke_environment

    prepare_cpu_smoke_environment()

    from experiment_runner.assets import accept_asset
    from experiment_runner.smoke import run_cpu_smoke_drop, run_cpu_smoke_slope
    from experiment_runner.storage import DataRoot

    data_root = DataRoot(args.data_root)
    data_root.initialize()
    package = data_root.location("inbox") / "fixtures" / "blue-box"
    package.mkdir(parents=True)
    shutil.copy2(args.fixture, package / "newton-mujoco.usda")
    acceptance = accept_asset(data_root, "fixtures/blue-box", enforce_readonly=False)
    if acceptance["status"] not in {"accepted", "partially_ready"}:
        raise RuntimeError(f"Fixture acceptance failed: {acceptance['status']}")

    runner = run_cpu_smoke_drop if args.template == "drop" else run_cpu_smoke_slope
    result = runner(
        data_root,
        asset_identity="fixtures/blue-box",
        asset_version=acceptance["asset_version"],
    )
    args.summary_path.write_text(
        json.dumps(
            {
                "acceptance_status": acceptance["status"],
                "asset_version": acceptance["asset_version"],
                "result": result,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
