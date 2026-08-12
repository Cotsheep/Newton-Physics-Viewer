@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo 未找到 uv，Newton-Test 尚未启动。
    echo 请先按照 uv 官方安装说明完成安装：
    echo https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

uv run --frozen newton-test
set "newton_test_exit_code=%errorlevel%"
if not "%newton_test_exit_code%"=="0" (
    echo.
    echo Newton-Test 异常退出，退出码：%newton_test_exit_code%
    pause
)
exit /b %newton_test_exit_code%
