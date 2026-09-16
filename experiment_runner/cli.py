from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from .assets import accept_asset, list_asset_versions
from .config import (
    DEFAULT_SERVER_CONFIG,
    ServerConfig,
    load_server_config,
    resolve_data_root,
    save_server_config,
)
from .profiles import describe_profiles
from .results import build_result_index
from .storage import DataRoot
from .web import install_static_site, serve_results


def _data_root(args: argparse.Namespace) -> DataRoot:
    data_root = resolve_data_root(args.data_root, config_path=args.config)
    data_root.require_existing_owned_directory()
    return data_root


def _command_configure_storage(args: argparse.Namespace) -> int:
    data_root = DataRoot(args.path)
    confirmed = DataRoot(args.confirm_admin_approved_path)
    if data_root.path != confirmed.path:
        raise ValueError(
            "The repeated administrator-approved path must exactly match the resolved data root"
        )
    data_root.require_existing_owned_directory()
    data_root.initialize(source_root=args.source_root)
    save_server_config(
        ServerConfig(data_root=data_root.path, web_port=args.web_port),
        args.config,
    )
    install_static_site(data_root)
    build_result_index(data_root)
    print(f"服务器数据根目录已配置：{data_root.path}")
    print(f"个人配置已保存：{args.config}")
    return 0


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _command_show_storage(args: argparse.Namespace) -> int:
    data_root = _data_root(args)
    data_root.require_initialized()
    usage = shutil.disk_usage(data_root.path)
    print(f"数据根目录：{data_root.path}")
    print(f"文件系统剩余：{_format_bytes(usage.free)} / {_format_bytes(usage.total)}")
    for name in (
        "inbox",
        "assets",
        "asset_trash",
        "import_reports",
        "batches",
        "runs",
        "trash",
        "web",
    ):
        location = data_root.location(name)
        print(f"- {name}: {_format_bytes(_directory_size(location))}")
    return 0


def _command_accept_asset(args: argparse.Namespace) -> int:
    report = accept_asset(
        _data_root(args),
        args.identity,
        git_commit=args.git_commit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"accepted", "partially_ready", "duplicate"} else 2


def _command_list_assets(args: argparse.Namespace) -> int:
    versions = list_asset_versions(_data_root(args))
    print(json.dumps(versions, ensure_ascii=False, indent=2))
    return 0


def _command_rebuild_index(args: argparse.Namespace) -> int:
    data_root = _data_root(args)
    install_static_site(data_root)
    index = build_result_index(data_root)
    print(
        f"结果索引已更新：{len(index['assets'])} 个资产，"
        f"{sum(len(asset['runs']) for asset in index['assets'])} 次运行"
    )
    if index["ignored_runs"]:
        print(f"忽略了 {len(index['ignored_runs'])} 个不完整或损坏的运行目录")
    return 0


def _command_serve_results(args: argparse.Namespace) -> int:
    data_root = _data_root(args)
    install_static_site(data_root)
    build_result_index(data_root)
    serve_results(
        data_root,
        host=args.host,
        port=args.port,
        exit_when_stdin_closes=args.exit_when_stdin_closes,
    )
    return 0


def _command_profiles(_args: argparse.Namespace) -> int:
    print(
        json.dumps(
            describe_profiles(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _command_check_readiness(args: argparse.Namespace) -> int:
    from .readiness import check_readiness
    report = check_readiness(args.data_root, config_path=args.config, port=args.port)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check in report["checks"]:
            print(f"{check['label']}: {check['status']} — {check['detail']}")
    return 0 if report["results_ready"] else 2


def _command_smoke_drop(args: argparse.Namespace) -> int:
    from .smoke import run_cpu_smoke_drop

    result = run_cpu_smoke_drop(
        _data_root(args),
        asset_identity=args.identity,
        asset_version=args.version,
        git_commit=args.git_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("物理解算使用 MuJoCo/Warp CPU，不使用 CUDA。")
    print("Windows 录像可能使用本机图形 GPU；Linux 录像必须验证为 Mesa 软件 OpenGL。")
    print("注意：这是 CPU 冒烟结果，不是正式 GPU 物理试验结论。")
    return 0


def _command_smoke_slope(args: argparse.Namespace) -> int:
    from .smoke import run_cpu_smoke_slope

    result = run_cpu_smoke_slope(
        _data_root(args),
        asset_identity=args.identity,
        asset_version=args.version,
        git_commit=args.git_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("物理解算使用 MuJoCo/Warp CPU，不使用 CUDA。")
    print("Windows 录像可能使用本机图形 GPU；Linux 录像必须验证为 Mesa 软件 OpenGL。")
    print("注意：这是固定 25°、CPU、单案例的非正式冒烟观察，不是正式摩擦结论。")
    return 0


def _command_smoke_drop_gpu(args: argparse.Namespace) -> int:
    from .gpu_safety import GpuSmokeSafetyError
    from .smoke import run_gpu_smoke_drop

    try:
        result = run_gpu_smoke_drop(
            _data_root(args),
            asset_identity=args.identity,
            asset_version=args.version,
            git_commit=args.git_commit,
            record_video=args.record_video,
        )
    except GpuSmokeSafetyError as exc:
        raise ValueError(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("物理解算固定使用容器内唯一可见的逻辑设备 cuda:0；禁止 CPU fallback。")
    if args.record_video:
        print("本次运行要求保存无头渲染录像；请在结果页检查视频内容。")
    else:
        print("本次未请求录像，仅保存结构化物理结果。")
    print("注意：这是 development/integration smoke，不是正式 GPU 物理试验结论。")
    return 0


def _add_data_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override NEWTON_DATA_ROOT and the server config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SERVER_CONFIG,
        help="Personal server config file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Newton-Test server-side storage, asset, and result commands."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    readiness = subparsers.add_parser("check-readiness", help="Read-only checks without CUDA or simulation.")
    _add_data_root_arguments(readiness)
    readiness.add_argument("--json", action="store_true")
    readiness.add_argument("--port", type=int, default=8765)
    readiness.set_defaults(handler=_command_check_readiness)

    configure = subparsers.add_parser(
        "configure-storage",
        help="Initialize the fixed data layout and save the personal server config.",
    )
    configure.add_argument("path", type=Path)
    configure.add_argument(
        "--confirm-admin-approved-path",
        type=Path,
        required=True,
        help=(
            "Repeat the exact administrator-approved persistent data-root path; "
            "this is a human safety gate, not a permission grant."
        ),
    )
    configure.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Deployed source root that must remain separate from the data root.",
    )
    configure.add_argument("--web-port", type=int, default=8765)
    configure.add_argument("--config", type=Path, default=DEFAULT_SERVER_CONFIG)
    configure.set_defaults(handler=_command_configure_storage)

    show = subparsers.add_parser("show-storage", help="Show storage location and usage.")
    _add_data_root_arguments(show)
    show.set_defaults(handler=_command_show_storage)

    accept = subparsers.add_parser(
        "accept-asset",
        help="Validate and atomically accept one inbox package.",
    )
    accept.add_argument("identity", help="Inbox-relative asset package directory.")
    accept.add_argument("--git-commit", default=None)
    _add_data_root_arguments(accept)
    accept.set_defaults(handler=_command_accept_asset)

    list_assets = subparsers.add_parser("list-assets", help="List accepted immutable versions.")
    _add_data_root_arguments(list_assets)
    list_assets.set_defaults(handler=_command_list_assets)

    rebuild = subparsers.add_parser("rebuild-index", help="Rebuild web/index.json.")
    _add_data_root_arguments(rebuild)
    rebuild.set_defaults(handler=_command_rebuild_index)

    serve = subparsers.add_parser(
        "serve-results",
        help="Run the loopback-only, read-only result server.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--exit-when-stdin-closes",
        action="store_true",
        help="Stop when the SSH control channel reaches EOF.",
    )
    _add_data_root_arguments(serve)
    serve.set_defaults(handler=_command_serve_results)

    profiles = subparsers.add_parser(
        "profiles",
        help="Show registered profiles and execution availability.",
    )
    profiles.set_defaults(handler=_command_profiles)

    smoke = subparsers.add_parser(
        "smoke-drop",
        help="Run one non-authoritative MuJoCo CPU drop case.",
    )
    smoke.add_argument("identity", help="Accepted asset identity.")
    smoke.add_argument("version", help="Complete immutable asset SHA-256 version.")
    smoke.add_argument("--git-commit", default=None)
    _add_data_root_arguments(smoke)
    smoke.set_defaults(handler=_command_smoke_drop)

    slope_smoke = subparsers.add_parser(
        "smoke-slope",
        help=(
            "Run one fixed 25-degree MuJoCo CPU slope smoke case "
            "(non-authoritative, single-case development smoke)."
        ),
        description=(
            "Run fixed 25-degree CPU slope smoke. This non-authoritative "
            "single-case development check does not produce a formal friction "
            "conclusion."
        ),
    )
    slope_smoke.add_argument("identity", help="Accepted slope-ready asset identity.")
    slope_smoke.add_argument("version", help="Complete immutable asset SHA-256 version.")
    slope_smoke.add_argument("--git-commit", default=None)
    _add_data_root_arguments(slope_smoke)
    slope_smoke.set_defaults(handler=_command_smoke_slope)

    gpu_smoke = subparsers.add_parser(
        "smoke-drop-gpu",
        help=(
            "Run one bounded non-authoritative MJWarp CUDA drop integration smoke "
            "inside an externally scheduled single-GPU container."
        ),
        description=(
            "Run exactly one medium-height Newton SolverMuJoCo/MJWarp CUDA integration "
            "smoke. Requires a complete Determined trial/allocation gate and exactly one "
            "process-visible GPU, selects only container-logical cuda:0, forbids CPU "
            "fallback, and does not submit a Determined experiment."
        ),
    )
    gpu_smoke.add_argument("identity", help="Accepted drop-ready asset identity.")
    gpu_smoke.add_argument("version", help="Complete immutable asset SHA-256 version.")
    gpu_smoke.add_argument("--git-commit", default=None)
    gpu_smoke.add_argument(
        "--record-video", action="store_true",
        help="Require EGL headless rendering and H.264 recording for this bounded case.",
    )
    _add_data_root_arguments(gpu_smoke)
    gpu_smoke.set_defaults(handler=_command_smoke_drop_gpu)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
