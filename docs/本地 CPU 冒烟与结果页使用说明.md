# 本地 CPU 冒烟与结果页使用说明

状态：本地摔落与固定坡度两条 CPU 冒烟链路可运行；独立 GPU integration smoke 已实现但未在目标集群运行
适用范围：开发验证，不是正式 GPU 资产物理结论

这条链路已经能够完成：

1. 创建固定的数据目录；
2. 验收一个包含 `newton-mujoco.usda` 的资产包；
3. 分别检查 `drop` 与 `slope_friction` 是否就绪，允许“部分就绪”；
4. 使用 `mujoco-cpu-wsl-smoke-v1` 运行一个中等高度摔落工况，或一个固定 25° 坡度工况；
5. 生成 H.264 MP4、开始画面、结束画面、预览、JSON 清单和 SHA-256 校验清单；
6. 在资产封面式 Web UI 中直接播放工况录像；
7. 通过只绑定回环地址的只读服务访问结果，MP4 支持拖动进度条所需的字节范围读取。

正式 GPU 配置 `mujoco-native-dt1ms-v1` 已登记，但正式 GPU 批次仍保持关闭。只有管理员提供
不可由个人账号修改的 GPU 授权策略，并完成远程集成验收后，才会开放正式运行。

独立的 `mujoco-warp-cuda-dt1ms-integration-smoke-v1` 不属于正式配置。它只允许在
Determined 分配单 GPU 的 trial 中执行一个 1 秒中等高度摔落，始终
`authoritative=false`，不能用于开放或替代 `mujoco-native-dt1ms-v1`。

## 1. 当前 Windows 环境快速验证

先在仓库根目录同步锁定环境并打开统一菜单：

```bat
uv sync --frozen
uv run newton-test
```

也可以双击仓库根目录的 `newton-test.cmd`。首次运行时，把“试验数据目录”设置为仓库之外的
专用目录，例如 `%USERPROFILE%\newton-test-data`；程序在确认后创建固定子目录。

要使用仓库自带的蓝色方块冒烟资产，可以在另一个 CMD 中执行：

```bat
mkdir "%USERPROFILE%\newton-test-data\inbox\fixtures\blue-box"
copy "tests\fixtures\smoke_asset\newton-mujoco.usda" "%USERPROFILE%\newton-test-data\inbox\fixtures\blue-box\newton-mujoco.usda"
```

回到统一菜单后依次选择：

1. “验收 inbox 中的资产”，再选择 `fixtures/blue-box`；
2. “运行本地 CPU 摔落冒烟”或“运行本地 CPU 坡度冒烟”，从对应就绪列表选择同一资产并确认摘要；
3. “打开本地结果页”。

菜单在内部保留完整的 64 位 `asset_version`，不要求手动复制哈希。冒烟把 MuJoCo 和 Warp
固定到 CPU，物理解算不使用 CUDA，推进 2 秒物理时间；录像开头另含 0.5 秒静止展示，因此
640×360、50 FPS 视频总长约 2.5 秒。Windows 录像使用系统 OpenGL，可能使用本机图形 GPU；
这不等于正式服务器 GPU 物理解算。Linux 在导入 Newton/Warp 前隐藏 CUDA，要求
Mesa 软件 OpenGL，并在推进物理时间前验证实际渲染器，验证失败时拒绝回退到硬件渲染。
结果清单中的 `authoritative` 为 `false`，不能与以后服务器生成的正式 GPU 结果混为一谈。
坡度入口固定为 25° 单案例，资产以零线速度和零角速度放在坡面上方的小安全间隙处，使用
MuJoCo 原生接触推进。结果记录初末位置、沿坡位移、末速度和
`moved`/`stayed_near_start`/`inconclusive` 开发观察；这些字段不构成正式摩擦结论。
当前固定 25° 本地坡度冒烟只接受就绪信息中 `rigid_body_count == 1` 的资产。多刚体资产仍可
通过通用 `slope_friction` 就绪检查，但不会出现在本地坡度冒烟菜单中，直接调用底层命令也会
在创建批次或运行目录前拒绝。这只是当前开发冒烟工况的实现限制，不是正式坡度试验的永久限制。
`newton-test-remote profiles` 的 `availability.entrypoints` 是列表：CPU profile 当前列出
`smoke-drop` 与 `smoke-slope`；正式 profile 保持 `reserved_not_runnable`、`runnable=false`
且入口列表为空。旧的单值 `availability.entrypoint` 作为兼容字段继续输出：CPU profile 为
`"smoke-drop"`，正式预留 profile 为 `null`；新代码推荐读取 `availability.entrypoints`。
本次兼容修复不删除旧字段，也不开放任何正式 GPU 命令。

当前验收层级需要分开理解：实现和自动化测试覆盖 Windows 与 WSL；真实求解、录像、图片、
清单和结果页端到端链路需要在 WSL 的 CPU 与 Mesa 软件 OpenGL 环境用下文的显式命令验证。Windows 的真实
端到端用例会因为无法证明软件 OpenGL 而明确跳过；远程服务器与正式数据目录未连接、未检查，
也没有运行任何服务器 GPU 仿真。人工结果页检查仍应按下文步骤进行，不能由自动化测试替代。

浏览器首页先显示资产封面。点击封面后进入资产试验页；最新一次试验默认展开，点击视频画面
即可播放，也可以进入独立播放页。按 `Ctrl+C` 关闭结果页后会返回统一菜单，不会删除结果。

## 2. WSL 目标用法

WSL 将使用仓库提交的 `pyproject.toml` 与 `uv.lock`，而不是复制 Windows Conda 环境：

```bash
uv sync --frozen
uv run newton-test
```

菜单行为与 Windows 相同，路径改为 Linux 路径。试验数据目录应位于仓库之外的 WSL ext4
目录。Windows 双击启动文件只用于 Windows，WSL 直接运行上面的命令。

默认全套测试不会自动执行真实 CPU 冒烟 E2E；这两项用例需要实际启动 MuJoCo、创建录像并
验证 MP4，耗时且依赖 Linux 软件渲染，因此保持显式 opt-in。默认命令中看到这两项 `skipped`
只表示没有运行，跳过不等于通过。

WSL 中运行真实摔落与坡度 E2E 的通用命令如下；临时根目录必须位于源码仓库之外：

```bash
NEWTON_TEST_RUN_CPU_SMOKE_E2E=1 \
NEWTON_TEST_E2E_TEMP_ROOT=/path/outside/Newton-Test/e2e-temp \
NEWTON_TEST_FFPROBE="$(command -v ffprobe)" \
uv run --frozen python -m unittest tests.test_cpu_smoke_e2e -v
```

运行前必须确认：`ffprobe` 可执行；`imageio-ffmpeg` 自带的 FFmpeg 提供 `libx264`；OpenGL
renderer 能验证为 Mesa 的 `llvmpipe`、`softpipe` 或 `swrast`。只有系统 `ffprobe` 缺失、
FFmpeg 成功查询编码器后明确缺少 `libx264`，或结构化 OpenGL 预检返回认可的环境缺失时，
真实 E2E 才允许跳过。`imageio-ffmpeg` 导入或 API 错误、FFmpeg 启动失败、超时、非零退出、
产品 traceback、无效 JSON 或未知非零退出码必须报失败。Windows 会明确跳过真实 E2E，
因为当前链路无法证明其系统 OpenGL 是软件渲染；这不否定 Windows 的单元测试，也不能被
记录成 Windows 真实 E2E 通过。

2026-08-15 对当前工作树的本地 WSL 验收实际运行了当前 `tests.test_cpu_smoke_e2e` 模块：
14 项全部通过、0 项跳过，其中真实摔落与真实坡度 E2E 各 1 项并均实际执行。结构化预检记录
renderer 为 `llvmpipe (LLVM 20.1.2, 256 bits)`、vendor 为 `Mesa`；Warp 使用 CPU，CUDA
被隐藏且结果记录 `cuda_used=false`。两个视频均通过 H.264、`yuv420p`、640×360、50 FPS
及工件/校验和检查。精确数量对应这次最终工作树；后续新增协议测试时，应以“当前模块全部
通过且两项真实 E2E 均未跳过”为验收条件，而不能沿用旧数量。这只证明本地 CPU/Mesa 开发
链路，不代表远程服务器或正式 GPU 物理试验已经验收。

本次实现没有修改 `.wslconfig`，也没有执行 `wsl --shutdown`。在 WSL 安装依赖前，仍需按
路线图单独维护并验证 WSL NAT、DNS 和 HTTPS 下载；这个操作会停止 WSL 中的现有进程，不能
和普通代码测试混在一起静默执行。

## 3. 远程结果浏览入口

部署完成后，在统一菜单的设置中保存 Windows SSH 别名，再选择“打开远程结果页”。底层接口
仍可用于自动化：

```bat
uv run newton-test remote-results <ssh-alias>
```

程序只把别名交给 Windows `ssh.exe`，不会在仓库或结果中保存服务器地址、用户名和私钥
路径。它建立：

```text
Windows 127.0.0.1
    → SSH 本地端口转发
    → 服务器 127.0.0.1
    → 只读结果服务
```

远程部署需要先让锁定环境中的 `newton-test-remote` 固定入口对非交互 SSH 会话可见。只读
结果服务不会启动 GPU 物理仿真；本地浏览器播放视频时可能使用本机图形加速。关闭这个窗口
只会关闭结果访问，不会停止另一个窗口中正在执行的试验。

## 4. 底层命令参考

统一菜单是日常入口。开发或自动化仍可直接使用：

```bat
uv run python run_physics_experiment.py configure-storage <管理员已创建并批准的数据目录> --confirm-admin-approved-path <重复同一绝对路径> --source-root "%CD%"
uv run python run_physics_experiment.py accept-asset <inbox-relative-identity> --data-root <data-root>
uv run newton-test local-smoke <data-root> <asset-identity> <full-asset-version>
uv run newton-test local-slope-smoke <data-root> <asset-identity> <full-asset-version>
uv run newton-test-remote smoke-drop <asset-identity> <full-asset-version> --data-root <data-root>
uv run newton-test-remote smoke-slope <asset-identity> <full-asset-version> --data-root <data-root>
uv run newton-test local-results <data-root>
```

GPU 命令不属于上述本地用法。它只能由 Determined trial 使用预制 Python 直接调用，不能
在 GPU 分配后通过 `uv run` 安装环境；具体命令和源码导出见[部署说明](../deployment/README.md)。

## 5. GPU integration smoke 的当前边界

实现和验证状态必须分开：

- 已实现：独立非正式 profile、`smoke-drop-gpu` CLI、恰好一块可见 GPU 的 fail-closed
  校验、容器逻辑 `cuda:0` 选择、1 秒/单工况/300 秒内部上限、审计元数据和结构化结果；
- 已在 Windows 本地用 mock/CPU 验证：零卡、多卡、单卡选择、CUDA 初始化失败不回退、CPU
  仍隐藏 CUDA、profile/CLI、metadata unknown、1000 步 worker 与结果协议；
- 历史文档记载过 Determined 非仿真环境探针，但本交付没有可复核的目标运行产物；
  仓库无法证明当前版本在服务器上已验收，旧探针中的系统 Python 3.10.12 也不满足项目要求；
- 尚未验证：在目标 trial 中用 Newton/MJWarp 推进 GPU 物理、GPU 无头 OpenGL/EGL 录像、
  正式 GPU 物理试验、多高度正式摔落和正式坡度实验。

GPU 冒烟不调用 CPU safety 路径，不隐藏 CUDA，也不修改全局设备可见性。反过来，现有 CPU
入口仍在导入仿真栈前设置 `CUDA_VISIBLE_DEVICES=-1`，Linux 继续要求 Mesa 软件 OpenGL。

当前 GPU 容器渲染能力未知，所以 GPU 冒烟不尝试录像。成功结果复用 manifest、status、
case、run.log 和 checksums 结构，明确写入 `gpu_headless_recording_not_validated`；没有真实
preview、poster、final 或 MP4 时不会创建这些文件。Determined 人工示例见
[`../deployment/README.md`](../deployment/README.md)。

## 6. 当前尚未开放

- 正式 GPU 批次；
- 多高度正式摔落；
- 正式多角度斜坡摩擦试验、预测临界坡度邻域和水平滑行段；
- 工况之间的 60 秒本地续行确认；
- 服务器 GPU 独占、磁盘余量和进程竞争策略；
- Web UI 中的回收和恢复操作；
- 通过浏览器创建或取消试验。

这些入口不会用临时缺省值提前开放，以免误占服务器 GPU 或把冒烟结果误认为正式结论。

服务器 `configure-storage` 不会替管理员授予目录权限。目标必须预先存在、属于当前个人账号且
可写；重复输入解析后的批准路径只是防止选错目录的人为确认关口。程序在后续每条服务器命令
中仍会复查目录所有权。
