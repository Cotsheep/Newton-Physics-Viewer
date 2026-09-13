from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .assets import (
    ASSET_ENTRYPOINT,
    accept_asset,
    inspect_template_readiness,
    list_asset_versions,
)
from .config import (
    DEFAULT_CONTROLLER_CONFIG,
    ControllerConfig,
    load_controller_config,
    save_controller_config,
)
from .results import build_result_index
from .storage import DataRoot
from .paths import require_viewer_source
from .web import create_result_server, install_static_site


def _validated_port(value: int) -> int:
    if not 1024 <= value <= 65535:
        raise ValueError("端口必须在 1024 到 65535 之间")
    return value


def _validated_alias(value: str) -> str:
    if (
        not value
        or value.startswith("-")
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise ValueError("SSH 别名只能包含字母、数字、点、下划线和短横线")
    return value


def build_remote_result_command(
    *,
    ssh_executable: str,
    host_alias: str,
    local_port: int,
    remote_port: int,
) -> list[str]:
    """Build a fixed SSH command without addresses, usernames, keys, or shell input."""

    alias = _validated_alias(host_alias)
    local = _validated_port(local_port)
    remote = _validated_port(remote_port)
    return [
        ssh_executable,
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-L",
        f"127.0.0.1:{local}:127.0.0.1:{remote}",
        "--",
        alias,
        "newton-test-remote",
        "serve-results",
        "--host",
        "127.0.0.1",
        "--port",
        str(remote),
        "--exit-when-stdin-closes",
    ]


def build_remote_readiness_command(
    *,
    ssh_executable: str,
    host_alias: str,
) -> list[str]:
    """Build the read-only check used before opening a result tunnel."""

    alias = _validated_alias(host_alias)
    return [
        ssh_executable,
        "-T",
        "-o",
        "ConnectTimeout=10",
        "--",
        alias,
        "command",
        "-v",
        "newton-test-remote",
    ]


def check_remote_results_ready(host_alias: str, *, remote_port: int = 8765) -> dict:
    try:
        return _check_remote_results_ready(host_alias, remote_port=remote_port)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("服务器只读就绪检查超时；不会启动 SSH 隧道") from exc


def _check_remote_results_ready(host_alias: str, *, remote_port: int = 8765) -> dict:
    ssh_executable = shutil.which("ssh.exe") or shutil.which("ssh")
    if ssh_executable is None:
        raise RuntimeError("找不到 OpenSSH 客户端 ssh.exe")
    print("正在执行只读服务器就绪检查……")
    completed = subprocess.run(
        build_remote_readiness_command(
            ssh_executable=ssh_executable,
            host_alias=host_alias,
        ),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f"（{detail[-1]}）" if detail else ""
        raise RuntimeError(
            "服务器尚未提供 newton-test-remote；本地程序不会自动部署或修改服务器"
            + suffix
        )
    command = [
        ssh_executable, "-T", "-o", "ConnectTimeout=10", "--", _validated_alias(host_alias),
        "newton-test-remote", "check-readiness", "--json", "--port", str(_validated_port(remote_port)),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True,
                               encoding="utf-8", errors="replace", timeout=60)
    try:
        report = json.loads(completed.stdout)
        if type(report) is not dict or report.get("schema_version") != 1:
            raise ValueError("unsupported readiness schema")
        checks = report["checks"]
        if not isinstance(checks, list) or not checks:
            raise ValueError("missing readiness checks")
        required = {"python", "package", "command", "storage", "writable", "loopback"}
        seen = set()
        blocked = []
        for check in checks:
            key = check["id"]
            if not isinstance(key, str) or key in seen or check["status"] not in {"ready", "blocked"}:
                raise ValueError("invalid readiness check")
            seen.add(key)
            print(f"{'通过' if check['status'] == 'ready' else '未就绪'}：{check['label']}")
            if (key in required or check.get("required_for_results") is True) and check["status"] == "blocked":
                blocked.append(key)
        if not required <= seen:
            raise ValueError("missing required result checks")
        if blocked or report.get("results_ready") is not True or completed.returncode != 0:
            raise RuntimeError("只读结果浏览尚未就绪；不会启动 SSH 隧道：" + ", ".join(blocked))
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("服务器返回了无效的结构化就绪报告；不会启动 SSH 隧道") from exc
    return report


def _wait_for_local_port(
    process: subprocess.Popen[bytes],
    *,
    port: int,
    timeout_seconds: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SSH 浏览会话提前结束，退出码 {return_code}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("等待本地结果端口就绪超时")


def _stop_foreground_process(process: subprocess.Popen[bytes]) -> None:
    """Stop a foreground child on every exceptional or interrupted exit path."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    except OSError:
        # The process can exit between poll() and terminate().
        return


def open_remote_results(
    *,
    host_alias: str,
    local_port: int = 8765,
    remote_port: int = 8765,
    open_browser: bool = True,
) -> int:
    check_remote_results_ready(host_alias, remote_port=remote_port)
    ssh_executable = shutil.which("ssh.exe") or shutil.which("ssh")
    if ssh_executable is None:
        raise RuntimeError("找不到 Windows OpenSSH 客户端 ssh.exe")
    command = build_remote_result_command(
        ssh_executable=ssh_executable,
        host_alias=host_alias,
        local_port=local_port,
        remote_port=remote_port,
    )
    print("正在建立只读结果浏览会话……")
    print("只读结果服务不会启动物理仿真；本地浏览器播放时可能使用本机图形加速。")
    print("关闭窗口或按 Ctrl+C 只会结束结果访问。")
    process = subprocess.Popen(command)
    try:
        _wait_for_local_port(process, port=local_port)
        address = f"http://127.0.0.1:{local_port}/"
        print(f"结果页：{address}")
        if open_browser:
            webbrowser.open(address)
        return process.wait()
    except KeyboardInterrupt:
        print("\n正在关闭结果浏览会话……")
        return 130
    finally:
        _stop_foreground_process(process)


def open_local_results(
    data_root: Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    root = DataRoot(data_root)
    root.require_initialized()
    install_static_site(root)
    build_result_index(root)
    server = create_result_server(root, port=_validated_port(port))
    address = f"http://127.0.0.1:{server.server_port}/"
    print(f"本地结果页：{address}")
    print("只读结果服务不会启动物理仿真；浏览器播放时可能使用本机图形加速。")
    print("按 Ctrl+C 关闭。")
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n本地结果页已关闭。")
    finally:
        server.server_close()
    return 0


def run_local_smoke(
    data_root: Path,
    *,
    asset_identity: str,
    asset_version: str,
    template: str = "drop",
) -> int:
    commands = {
        "drop": ("smoke-drop", "摔落"),
        "slope_friction": ("smoke-slope", "坡度"),
    }
    try:
        smoke_command, label = commands[template]
    except KeyError as exc:
        raise ValueError(f"不支持的本地 CPU 冒烟模板：{template}") from exc
    command = [
        sys.executable,
        "-m",
        "experiment_runner.cli",
        smoke_command,
        asset_identity,
        asset_version,
        "--data-root",
        str(data_root),
    ]
    print(f"正在运行一个非正式 MuJoCo CPU {label}工况……")
    print("物理解算和 Warp 固定为 CPU，不使用 CUDA。")
    print("Windows 录像使用系统 OpenGL，可能使用本机图形 GPU。")
    print("Linux 录像仅允许已验证的 Mesa 软件 OpenGL。")
    return subprocess.run(command, check=False).returncode


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_separate_roots(config: ControllerConfig) -> None:
    if config.viewer_source is not None:
        require_viewer_source(config.viewer_source, config.data_root)
    if config.viewer_source is None or config.data_root is None:
        return
    viewer = config.viewer_source.resolve(strict=False)
    data = config.data_root.resolve(strict=False)
    if viewer == data or _is_relative_to(viewer, data) or _is_relative_to(data, viewer):
        raise ValueError("Viewer 资产源目录和试验数据目录必须相互独立")


def _strip_path_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def _prompt_path(label: str, current: Path | None) -> Path | None:
    shown = str(current) if current is not None else "未设置"
    value = _strip_path_quotes(input(f"{label} [{shown}]（Enter 保持，- 清除）："))
    if not value:
        return current
    if value == "-":
        return None
    return Path(value).expanduser().resolve(strict=False)


def _prompt_text(label: str, current: str | None) -> str | None:
    shown = current or "未设置"
    value = input(f"{label} [{shown}]（Enter 保持，- 清除）：").strip()
    if not value:
        return current
    if value == "-":
        return None
    return value


def _prompt_port(label: str, current: int) -> int:
    value = input(f"{label} [{current}]（Enter 保持）：").strip()
    return current if not value else _validated_port(int(value))


def _confirm(message: str) -> bool:
    return input(f"{message} [y/N]：").strip().casefold() in {"y", "yes", "是"}


def _configure_interactive(
    config: ControllerConfig,
    *,
    config_path: Path,
) -> ControllerConfig:
    print("\n设置只保存在当前用户的个人配置中，不保存密码或 SSH 私钥。")
    viewer_source = _prompt_path("Viewer 资产源目录", config.viewer_source)
    if viewer_source is not None and (not viewer_source.exists() or not viewer_source.is_dir()):
        raise ValueError("Viewer 资产源目录必须是已经存在的目录")

    data_root_path = _prompt_path("试验数据目录", config.data_root)
    ssh_alias = _prompt_text("SSH 服务器别名", config.ssh_alias)
    local_port = _prompt_port("本地结果页端口", config.local_port)
    remote_port = _prompt_port("服务器结果页端口", config.remote_port)
    browser_default = "Y" if config.open_browser else "N"
    browser_answer = input(
        f"打开结果页时自动启动浏览器？[Y/N，当前 {browser_default}]："
    ).strip().casefold()
    open_browser = config.open_browser
    if browser_answer in {"y", "yes", "是"}:
        open_browser = True
    elif browser_answer in {"n", "no", "否"}:
        open_browser = False
    elif browser_answer:
        raise ValueError("请输入 Y、N 或直接按 Enter")

    updated = replace(
        config,
        viewer_source=viewer_source,
        data_root=data_root_path,
        ssh_alias=ssh_alias,
        local_port=local_port,
        remote_port=remote_port,
        open_browser=open_browser,
    )
    _require_separate_roots(updated)
    if updated.ssh_alias is not None:
        _validated_alias(updated.ssh_alias)

    if data_root_path is not None:
        data_root = DataRoot(data_root_path)
        separate_from = [viewer_source] if viewer_source is not None else []
        data_root.validate_location(
            source_root=_project_root(),
            separate_from=separate_from,
        )
        try:
            data_root.require_initialized(
                source_root=_project_root(),
                separate_from=separate_from,
            )
        except ValueError:
            print(f"试验数据目录尚未初始化：{data_root.path}")
            if _confirm("现在创建受管理的 inbox、assets、runtime 等子目录吗？"):
                data_root.initialize(source_root=_project_root())
                install_static_site(data_root)
                build_result_index(data_root)
                print(f"已初始化试验数据目录：{data_root.path}")
            else:
                print("未初始化，因此不会保存这个试验数据目录。")
                updated = replace(updated, data_root=config.data_root)

    save_controller_config(updated, config_path)
    print(f"个人配置已保存：{config_path}")
    return updated


def _require_data_root(config: ControllerConfig) -> DataRoot:
    if config.data_root is None:
        raise ValueError("尚未设置试验数据目录；请先进入“设置”")
    root = DataRoot(config.data_root)
    _require_separate_roots(config)
    separate_from = [config.viewer_source] if config.viewer_source is not None else []
    root.require_initialized(
        source_root=_project_root(),
        separate_from=separate_from,
    )
    return root


def discover_inbox_packages(data_root: DataRoot) -> list[str]:
    inbox = data_root.location("inbox")
    packages: list[str] = []
    for entrypoint in inbox.rglob(ASSET_ENTRYPOINT):
        if not entrypoint.is_file():
            continue
        packages.append(entrypoint.parent.relative_to(inbox).as_posix())
    return sorted(set(packages), key=str.casefold)


def _menu_readiness(
    data_root: DataRoot,
    identity: str,
    version: str,
) -> dict[str, Any] | None:
    entrypoint = data_root.resolve_managed(
        "assets",
        *identity.split("/"),
        version,
        ASSET_ENTRYPOINT,
    )
    try:
        return inspect_template_readiness(entrypoint)
    except Exception:
        return None


def _readiness_label(readiness: dict[str, Any] | None) -> str:
    if readiness is None:
        return "就绪状态检查失败"
    drop_ready = readiness["drop"]["status"] == "ready"
    slope_ready = readiness["slope_friction"]["status"] == "ready"
    if drop_ready and slope_ready:
        return "摔落、坡度就绪"
    if drop_ready:
        return "部分就绪（仅摔落）"
    if slope_ready:
        return "部分就绪（仅坡度）"
    return "未就绪"


def list_assets_for_menu(data_root: DataRoot) -> list[dict[str, Any]]:
    rows = list_asset_versions(data_root)
    menu_rows: list[dict[str, Any]] = []
    for row in rows:
        readiness = _menu_readiness(data_root, row["identity"], row["version"])
        menu_rows.append(
            {
                **row,
                "readiness": _readiness_label(readiness),
                "template_readiness": readiness or {},
            }
        )
    return menu_rows


def _template_ready(row: dict[str, Any], template: str) -> bool:
    readiness = row.get("template_readiness")
    if not isinstance(readiness, dict):
        return False
    result = readiness.get(template)
    return isinstance(result, dict) and result.get("status") == "ready"


def _single_body_slope_smoke_ready(row: dict[str, Any]) -> bool:
    if not _template_ready(row, "slope_friction"):
        return False
    result = row["template_readiness"]["slope_friction"]
    checks = result.get("checks")
    return (
        isinstance(checks, dict)
        and not isinstance(checks.get("rigid_body_count"), bool)
        and checks.get("rigid_body_count") == 1
    )


def _choose_row(rows: list[dict[str, Any]], *, heading: str) -> dict[str, Any] | None:
    if not rows:
        print("没有可选择的项目。")
        return None
    print(f"\n{heading}")
    for index, row in enumerate(rows, start=1):
        suffix = f"    {row['readiness']}" if "readiness" in row else ""
        print(f"{index}. {row['identity']}    {row.get('version', '')[:8]}{suffix}")
    print("0. 返回")
    value = input("请选择编号：").strip()
    if value == "0" or not value:
        return None
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError("请输入列表中的数字编号") from exc
    if not 1 <= index <= len(rows):
        raise ValueError("选择的编号超出列表范围")
    return rows[index - 1]


def launch_viewer(source: Path) -> int:
    require_viewer_source(source, load_controller_config().data_root)
    if not source.exists():
        raise ValueError(f"Viewer 资产路径不存在：{source}")
    print("即将启动本地桌面 Viewer；根据所选求解器，它可能使用本机 GPU。")
    command = [sys.executable, str(_project_root() / "view.py"), str(source)]
    return subprocess.run(command, check=False).returncode


def _open_viewer_interactive(config: ControllerConfig) -> None:
    shown = str(config.viewer_source) if config.viewer_source is not None else "未设置"
    value = _strip_path_quotes(
        input(f"资产目录或单个 URDF/USD/GLB 路径 [{shown}]（Enter 使用默认）：")
    )
    source = Path(value).expanduser().resolve(strict=False) if value else config.viewer_source
    if source is None:
        raise ValueError("尚未设置 Viewer 资产源目录，也没有输入临时资产路径")
    require_viewer_source(source, config.data_root)
    return_code = launch_viewer(source)
    if return_code:
        print(f"Viewer 已退出，退出码：{return_code}")


def _accept_asset_interactive(config: ControllerConfig) -> None:
    data_root = _require_data_root(config)
    identities = discover_inbox_packages(data_root)
    rows = [{"identity": identity} for identity in identities]
    selected = _choose_row(rows, heading=f"待验收资产包（{data_root.location('inbox')}）")
    if selected is None:
        if not rows:
            print(f"请先把包含 {ASSET_ENTRYPOINT} 的完整资产包放入上述 inbox 目录。")
        return
    report = accept_asset(data_root, selected["identity"])
    status_labels = {
        "accepted": "已验收入库",
        "partially_ready": "已入库，但仅部分试验模板就绪",
        "duplicate": "相同资产版本已存在；inbox 副本保持不变",
        "failed": "验收失败",
    }
    print(f"结果：{status_labels.get(report['status'], report['status'])}")
    if report.get("asset_version"):
        print(f"资产版本：{report['asset_version']}")
    if report.get("error"):
        print(f"原因：{report['error']['code']} — {report['error']['message']}")
    print(f"验收报告目录：{data_root.location('import_reports')}")


def _show_assets_interactive(config: ControllerConfig) -> None:
    data_root = _require_data_root(config)
    rows = list_assets_for_menu(data_root)
    if not rows:
        print("尚无已验收入库的资产。")
        return
    print("\n已验收资产")
    for index, row in enumerate(rows, start=1):
        print(
            f"{index}. {row['identity']}    {row['version'][:8]}    {row['readiness']}"
        )
    print("完整版本哈希由程序内部使用，日常选择不需要手动输入。")


def _run_smoke_interactive(config: ControllerConfig) -> None:
    data_root = _require_data_root(config)
    rows = [
        row
        for row in list_assets_for_menu(data_root)
        if _template_ready(row, "drop")
    ]
    selected = _choose_row(rows, heading="可运行本地摔落冒烟的资产")
    if selected is None:
        return
    print("\n运行摘要")
    print(f"资产：{selected['identity']}")
    print(f"版本：{selected['version'][:8]}")
    print("工况：本地 CPU 中等高度摔落冒烟")
    print("物理解算：MuJoCo/Warp CPU，不使用 CUDA")
    print("录像：Windows 使用系统 OpenGL，可能使用本机图形 GPU")
    print("录像：Linux 必须验证为 Mesa 软件 OpenGL")
    print("正式物理结论：否")
    if not _confirm("确认运行？"):
        print("已取消。")
        return
    return_code = run_local_smoke(
        data_root.path,
        asset_identity=selected["identity"],
        asset_version=selected["version"],
    )
    print("冒烟运行完成。" if return_code == 0 else f"冒烟运行失败，退出码：{return_code}")


def _run_slope_smoke_interactive(config: ControllerConfig) -> None:
    data_root = _require_data_root(config)
    rows = [
        row
        for row in list_assets_for_menu(data_root)
        if _single_body_slope_smoke_ready(row)
    ]
    selected = _choose_row(rows, heading="可运行本地坡度冒烟的资产")
    if selected is None:
        return
    print("\n运行摘要")
    print(f"资产：{selected['identity']}")
    print(f"版本：{selected['version'][:8]}")
    print("工况：固定 25°、2 秒的本地 CPU 单案例坡度冒烟")
    print("物理解算：MuJoCo/Warp CPU，不使用 CUDA")
    print("录像：Windows 使用系统 OpenGL，可能使用本机图形 GPU")
    print("录像：Linux 必须验证为 Mesa 软件 OpenGL")
    print("结论边界：仅报告 moved/stayed_near_start/inconclusive，不是正式摩擦结论")
    print("运行方式：前台执行；按 Ctrl+C 可安全取消并返回主菜单")
    if not _confirm("确认运行？"):
        print("已取消。")
        return
    return_code = run_local_smoke(
        data_root.path,
        asset_identity=selected["identity"],
        asset_version=selected["version"],
        template="slope_friction",
    )
    print(
        "坡度冒烟运行完成。"
        if return_code == 0
        else f"坡度冒烟运行失败，退出码：{return_code}"
    )


def _open_remote_interactive(config: ControllerConfig) -> None:
    if config.ssh_alias is None:
        raise ValueError("尚未设置 SSH 服务器别名；请先进入“设置”")
    open_remote_results(
        host_alias=config.ssh_alias,
        local_port=config.local_port,
        remote_port=config.remote_port,
        open_browser=config.open_browser,
    )


def _show_capabilities() -> None:
    print("\n当前已可用")
    print("- 桌面资产 Viewer（URDF、USD、GLB）")
    print("- inbox 资产验收入库与按模板就绪检查")
    print("- 非正式 MuJoCo CPU 中等高度摔落冒烟")
    print("- 非正式 MuJoCo CPU 固定 25° 单案例坡度冒烟")
    print("- 本地只读结果页")
    print("- 经 Tailscale/SSH 隧道访问服务器只读结果页")
    print("\n尚未开放")
    print("- 正式 GPU 试验批次")
    print("- 正式多高度摔落与多角度坡度试验执行")
    print("- Web UI 提交任务、回收或删除结果")
    print("- 自动部署或更新服务器")


def _print_menu(config: ControllerConfig) -> None:
    data_root = str(config.data_root) if config.data_root is not None else "未设置"
    ssh_alias = config.ssh_alias or "未设置"
    print("\nNewton-Test 统一控制入口")
    print(f"试验数据目录：{data_root}")
    print(f"远程服务器：{ssh_alias}")
    print("1. 打开桌面资产 Viewer")
    print("2. 验收 inbox 中的资产")
    print("3. 运行本地 CPU 摔落冒烟")
    print("4. 运行本地 CPU 坡度冒烟")
    print("5. 打开本地结果页")
    print("6. 打开远程结果页")
    print("7. 查看已验收资产与就绪状态")
    print("8. 查看项目能力状态")
    print("9. 设置")
    print("0. 退出")


def _interactive(*, config_path: Path = DEFAULT_CONTROLLER_CONFIG) -> int:
    first_run = not config_path.is_file()
    config = load_controller_config(config_path)
    if first_run:
        print("首次运行：先设置常用目录和 SSH 别名；不需要的项目可以按 Enter 跳过。")
        try:
            config = _configure_interactive(config, config_path=config_path)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"首次设置未完成：{exc}")
            print("你仍可进入主菜单，并稍后选择“设置”。")

    while True:
        _print_menu(config)
        try:
            choice = input("请选择：").strip()
            if choice == "0":
                return 0
            if choice == "1":
                _open_viewer_interactive(config)
            elif choice == "2":
                _accept_asset_interactive(config)
            elif choice == "3":
                _run_smoke_interactive(config)
            elif choice == "4":
                _run_slope_smoke_interactive(config)
            elif choice == "5":
                root = _require_data_root(config)
                open_local_results(
                    root.path,
                    port=config.local_port,
                    open_browser=config.open_browser,
                )
            elif choice == "6":
                _open_remote_interactive(config)
            elif choice == "7":
                _show_assets_interactive(config)
            elif choice == "8":
                _show_capabilities()
            elif choice == "9":
                config = _configure_interactive(config, config_path=config_path)
            else:
                print("无法识别的菜单选项。")
        except KeyboardInterrupt:
            print("\n当前操作已取消，返回主菜单。")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"操作未完成：{exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Newton-Test local controller (does not import Newton until an action needs it)."
    )
    subparsers = parser.add_subparsers(dest="command")

    local = subparsers.add_parser("local-results", help="Open a local read-only result page.")
    local.add_argument("data_root", type=Path)
    local.add_argument("--port", type=int, default=8765)
    local.add_argument("--no-browser", action="store_true")

    remote = subparsers.add_parser(
        "remote-results",
        help="Open an SSH-tunneled remote read-only result page.",
    )
    remote.add_argument("host_alias")
    remote.add_argument("--local-port", type=int, default=8765)
    remote.add_argument("--remote-port", type=int, default=8765)
    remote.add_argument("--no-browser", action="store_true")

    smoke = subparsers.add_parser(
        "local-smoke",
        help="Run one non-authoritative local MuJoCo CPU drop.",
    )
    smoke.add_argument("data_root", type=Path)
    smoke.add_argument("asset_identity")
    smoke.add_argument("asset_version")

    slope_smoke = subparsers.add_parser(
        "local-slope-smoke",
        help="Run one fixed 25-degree non-authoritative local MuJoCo CPU slope case.",
    )
    slope_smoke.add_argument("data_root", type=Path)
    slope_smoke.add_argument("asset_identity")
    slope_smoke.add_argument("asset_version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            return _interactive()
        if args.command == "local-results":
            return open_local_results(
                args.data_root,
                port=args.port,
                open_browser=not args.no_browser,
            )
        if args.command == "remote-results":
            return open_remote_results(
                host_alias=args.host_alias,
                local_port=args.local_port,
                remote_port=args.remote_port,
                open_browser=not args.no_browser,
            )
        if args.command == "local-smoke":
            return run_local_smoke(
                args.data_root,
                asset_identity=args.asset_identity,
                asset_version=args.asset_version,
            )
        if args.command == "local-slope-smoke":
            return run_local_smoke(
                args.data_root,
                asset_identity=args.asset_identity,
                asset_version=args.asset_version,
                template="slope_friction",
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2
