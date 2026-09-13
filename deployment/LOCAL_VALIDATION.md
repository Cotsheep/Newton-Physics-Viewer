# GPU 交付候选本地验证记录

日期：2026-09-13。范围：本地代码修复与非 GPU 验证。

后续用户授权自动验收后，已完成WSL真实CPU E2E；详见[后续验收记录](ACCEPTANCE_RUN_2026-09-13.md)。下文保留修复轮当时的执行范围。

结论：已修复本次审查发现的代码和部署契约问题，可提交代码候选。当前工作树尚未提交，未生成带干净 commit 的发布导出，也没有目标 trial 验收产物；不能宣称“已在真实 GPU 跑通”。实际部署和首轮执行条件见 [交付说明](README.md)。

## Git 与来源

- 分支：`main`。
- HEAD：`f41f9d2105145abde9e4baaea22627dacb2fe100`。
- 相对本地缓存的 `origin/main`：领先2、落后0；未 fetch 或连接远程 Git。
- 工作树包含用户原有修改及本轮修复，仍为 dirty；没有 commit、push 或破坏性 Git 操作。
- 验证时运行源码指纹：`0ae60ac5c22f8026675d242aecb880a03ef75c5ca23efab6249747b5f4138c1e`。该指纹由 `experiment_runner.release.source_state()` 计算，覆盖源码、静态 UI、顶层 Python 文件和依赖声明，不覆盖测试与文档；不是发布 commit 或签名。

## 修复覆盖

| 原问题 | 修复及验证 |
| --- | --- |
| 共享 GPU 函数可绕过入口、正式 profile 可执行 | gate 签发强类型进程内许可；几何测量、建模、设备选择、步进校验许可；正式 profile 拒绝，录像仅允许 CPU |
| slot 输入类型宽松 | JSON 必须是单个非负整数列表；拒绝字符串、bool、float、负数和多 slot |
| 未核对分配 UUID 与 runtime UUID | 规范化匹配；不匹配/非法值在模型前失败，缺失明确记 unavailable/warning |
| 首步/完成审计不充分 | solver 开始与完成分离，CUDA 同步成功后计数，异常和中断保留完成数，1000步后才声明完成 |
| dirty 代码仍只记 HEAD | CPU/GPU 记录 source 指纹和 dirty；GPU 要求干净匹配 checkout 或校验通过的源码导出 |
| 临时 Viewer 路径与数据/源码重叠 | 统一路径边界，运行入口再次验证；拒绝数据目录中的符号链接/junction 逃逸 |
| 单个完整材质掩盖其他材质摩擦缺失 | 所有绑定碰撞材质分别检查；检查器升级为 asset-readiness-v2 |
| 缺少服务器 readiness 及远程浏览前置检查 | 结构化只读 readiness；必要项缺失、blocked、损坏或超时均不建隧道；正式GPU关闭不阻止结果浏览 |
| YAML 缺少 global_batch_size、测试只查字符串 | 补齐正确层级；PyYAML 实际解析并校验类型、值和挂载结构；Bash 语法检查通过 |
| Python 来源和部署步骤不明确 | 默认镜像内锁定环境；文档区分独立只读环境挂载，禁止 trial 内安装；提供干净源码导出步骤 |

结果服务器额外收紧为只接受字面量 `127.0.0.1`，避免名称解析改变监听地址。未开放正式GPU、Web提交/删除、自动部署、数据集上传或后台托管。

## 实际验证

Windows 使用 `D:\app\uv\uv.exe`，Python 3.12.13。测试进程显式设置：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NEWTON_TEST_RUN_CPU_SMOKE_E2E='0'
$env:PYTHONIOENCODING='utf-8'
& 'D:\app\uv\uv.exe' run --no-sync --offline python -B -m unittest discover -s tests -q
```

最后一次完整执行：`Ran 192 tests in 20.463s`，`OK (skipped=2)`，即190项通过、2项跳过、0失败。包含 GPU fake/mock、底层绕过回归、源码导出、readiness、本地CPU普通测试、YAML结构与结果协议。

跳过的两项准确为 `RealCpuSmokeE2ETests.test_real_drop_smoke_end_to_end` 和 `RealCpuSmokeE2ETests.test_real_slope_smoke_end_to_end`。两项均因未显式设置 `NEWTON_TEST_RUN_CPU_SMOKE_E2E=1` 而跳过，没有启动其真实录像E2E，也没有把环境前置检测当作已运行通过。

其他检查：

- `uv lock --check --offline`：通过，锁定21个包。
- `git -c core.safecrlf=false diff --check`：通过。
- `python -B -m experiment_runner.cli --help`、`check-readiness --help`、`profiles`：通过，正式profile仍不可执行。
- `require_locked_runtime()`：本地Python与16个运行依赖（含传递依赖）的版本全部匹配。PyYAML仅为开发依赖。
- 全仓库Markdown文件的20个相对链接目标检查：全部存在（不包含本文新增链接）。
- 使用PyYAML读取entrypoint，拆出内层脚本后以Git Bash `bash -n -c`检查：退出0，未执行脚本或启动GPU。
- readiness测试在全新Python进程验证：未导入Warp/Newton/MuJoCo，未改环境变量、未写测试数据或配置；报告保留各项状态。

测试中的“CUDA driver unavailable”来自CPU测试导入Warp，不代表运行了GPU。Trimesh缺少可选SciPy时使用其内置非稀疏路径的日志也未构成测试失败；未为此升级或安装物理依赖。

## 未验证事项与最小下一步

本轮没有初始化真实CUDA、运行GPU smoke、连接真实服务器或Determined、提交或激活experiment。只读查阅了公开的Determined 0.38.1官方源码以核对配置与退出语义。

下一步须形成经审核的干净commit，导出源码；由操作员填写批准的镜像/环境路径、资源池、数据挂载和已入库资产identity/version，人工建立paused experiment并复核最终配置。随后在获准的一次trial中完成1000步，核对退出码、UUID、同步步数、产物校验和资源释放，才能升级为“目标GPU已验收”。本地测试不能代替这一步。
