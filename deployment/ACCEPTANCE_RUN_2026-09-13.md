# 2026-09-13 验收执行记录

这是用户要求“本地可运行验收自动完成、服务器操作逐步指导”后的新一轮验收。它补充 [修复轮记录](LOCAL_VALIDATION.md)，不覆盖或改写上轮未执行真实CPU E2E的事实。可复用执行要求见 [分步验收提示词](ACCEPTANCE_PROMPT.md)。

## 当前结论

本地默认测试和WSL真实CPU E2E均通过。源码仍有未提交修改，发布版本尚未冻结；服务器就绪和真实GPU验收尚未执行。没有commit、push、服务器连接或Determined提交/激活，也没有修改业务代码或安装依赖。

| 阶段 | 结果 | 证据 |
| --- | --- | --- |
| Windows默认测试 | 通过，2项显式opt-in E2E跳过 | 192项：190通过、2跳过、0失败，17.992秒 |
| WSL真实CPU E2E模块 | 通过，无跳过 | 14项全部通过，30.573秒；包含真实drop和slope各1项 |
| 锁文件/依赖/语法/CLI/YAML/链接 | 通过 | 见下述实际检查 |
| 干净源码发布 | 阻塞 | 当前工作树dirty，GPU来源检查按设计拒绝 |
| 服务器只读检查 | 待用户操作 | 下一步只检查已有命令和Python版本 |
| Determined真实GPU运行 | 未执行 | 没有trial、GPU运行产物或资源释放证据 |

两组测试存在重叠，不能把192与14简单相加当作独立测试总数。Windows默认套件中的两个跳过项，已在后续WSL opt-in执行中分别实际通过。

## 验证对象

- 仓库：`D:\code\PythonProjects\Newton-Test`。
- HEAD：`f41f9d2105145abde9e4baaea22627dacb2fe100`，工作树dirty；HEAD不是当前未提交修复的发布版本。
- 运行源码指纹：`0ae60ac5c22f8026675d242aecb880a03ef75c5ca23efab6249747b5f4138c1e`，与修复轮相同；提示词和验收文档不在运行源码指纹覆盖范围内。
- Windows uv：0.12.2；WSL现成项目Python：3.12.13。
- WSL系统默认Python没有项目依赖；实际测试使用已有项目venv，未安装或升级任何包。

## Windows自动检查

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NEWTON_TEST_RUN_CPU_SMOKE_E2E='0'
$env:PYTHONIOENCODING='utf-8'
& 'D:\app\uv\uv.exe' run --no-sync --offline python -B -m unittest discover -s tests -q
```

结果：`Ran 192 tests in 17.992s`，`OK (skipped=2)`。

默认跳过项是 `RealCpuSmokeE2ETests.test_real_drop_smoke_end_to_end` 与 `test_real_slope_smoke_end_to_end`；原因是该次命令明确关闭opt-in。不是缺依赖或异常被伪装成skip。

其他实际执行项：

- `uv lock --check --offline`通过，解析21个锁定包。
- `git -c core.safecrlf=false diff --check`通过。
- Windows与WSL的`require_locked_runtime()`均通过，16个传递运行依赖的版本完整匹配。
- CLI主help、check-readiness help、smoke-drop-gpu help均正常退出；profiles中正式profile仍不可执行。
- 全仓库Python文件AST语法检查通过。
- 添加本轮提示词前的21个Markdown相对链接目标全部存在；本轮新增文档链接另作收尾检查。
- 解析YAML后对内层entrypoint执行现有Git Bash的`bash -n -c`，退出0；未实际执行GPU入口。
- 对当前源码直接调用GPU来源校验，得到预期拒绝：`GPU execution requires a clean checkout at the exact deployed commit`。因此没有生成伪装成已提交版本的导出。

测试日志中的Warp CUDA driver unavailable来自CPU测试导入。Trimesh缺少可选SciPy时转入其内置计算路径的日志未构成测试失败；未为此更改依赖。

## WSL真实CPU验收

从Windows实际执行以下命令，使用Ubuntu内已有环境、当前挂载源码和仓库外临时目录：

```powershell
wsl.exe -d Ubuntu --cd /mnt/d/code/PythonProjects/Newton-Test --exec env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 LIBGL_ALWAYS_SOFTWARE=true MESA_LOADER_DRIVER_OVERRIDE=swrast NEWTON_TEST_REQUIRE_SOFTWARE_OPENGL=1 NEWTON_TEST_RUN_CPU_SMOKE_E2E=1 NEWTON_TEST_E2E_TEMP_ROOT=/home/cotsheep/newton-test-e2e-temp NEWTON_TEST_FFPROBE=/usr/bin/ffprobe /home/cotsheep/.local/share/newton-test/venv/bin/python -B -m unittest tests.test_cpu_smoke_e2e -v
```

关键结果：

```text
test_real_drop_smoke_end_to_end ... ok
test_real_slope_smoke_end_to_end ... ok
Ran 14 tests in 30.573s
OK
```

现有测试先验证Mesa软件OpenGL，再用`tests/fixtures/smoke_asset`验收入库、调用真实CPU runner、验证manifest/status/case、图片、MP4和checksums。视频通过ffprobe核对H.264、yuv420p、640×360、50 FPS；结果验证非正式、CPU与finite状态。两项真实测试均未跳过。临时产物按现有测试设计在结束后清理，本轮没有保存可长期浏览的视频副本。

这证明本地CPU/Mesa开发链路，不证明服务器镜像或GPU执行链路。

## 下一步

由用户在已经登录的目标服务器Linux终端检查`python3`、`newton-test-remote`、`det`是否存在，并返回Python版本。助手收到输出后仅给下一步；先完成只读准备核验，不直接启动GPU或安装环境。

源码提交与冻结仍须遵守用户尚未解除的Git操作限制。服务器信息收集可以继续，但实际部署导出和GPU激活前必须先形成经审核的干净commit、批准的环境参数及已校验配置。
