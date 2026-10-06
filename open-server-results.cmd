@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if errorlevel 1 (
  echo 无法打开项目目录。
  pause
  exit /b 2
)
if not exist ".venv\Scripts\python.exe" (
  echo 找不到本项目的本地 Python 环境：.venv\Scripts\python.exe
  echo 请按 README.md 检查本地环境后重试。
  popd
  pause
  exit /b 2
)
".venv\Scripts\python.exe" -X utf8 -B "experimentctl.py" server-results %*
set "newton_result_exit=%errorlevel%"
popd
if not "%newton_result_exit%"=="0" if not "%newton_result_exit%"=="130" pause
exit /b %newton_result_exit%
