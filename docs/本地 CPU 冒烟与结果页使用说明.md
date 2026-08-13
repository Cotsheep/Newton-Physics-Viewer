# 本地 CPU 冒烟与结果页使用说明

状态：本地摔落与固定坡度两条 CPU 冒烟链路可运行
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
固定到 CPU，推进 2 秒物理时间；录像开头另含 0.5 秒静止展示，因此 640×360、50 FPS 视频
总长约 2.5 秒。Windows 录像使用本机 OpenGL；Linux 在导入 Newton/Warp 前隐藏 CUDA，要求
Mesa 软件 OpenGL，并在推进物理时间前验证实际渲染器，验证失败时拒绝回退到硬件渲染。
结果清单中的 `authoritative` 为 `false`，不能与以后服务器生成的正式 GPU 结果混为一谈。
坡度入口固定为 25° 单案例，资产以零线速度和零角速度放在坡面上方的小安全间隙处，使用
MuJoCo 原生接触推进。结果记录初末位置、沿坡位移、末速度和
`moved`/`stayed_near_start`/`inconclusive` 开发观察；这些字段不构成正式摩擦结论。
`newton-test-remote profiles` 的 `availability.entrypoints` 是列表：CPU profile 当前列出
`smoke-drop` 与 `smoke-slope`；正式 profile 保持 `reserved_not_runnable`、`runnable=false`
且入口列表为空。旧的单值 `availability.entrypoint` 字段不再输出。

当前验收层级需要分开理解：实现和自动化测试覆盖 Windows 与 WSL；真实求解、录像、图片、
清单和结果页端到端链路已在 WSL 的 CPU 与 Mesa 软件 OpenGL 环境运行验证。Windows 的真实
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

远程部署需要先让锁定环境中的 `newton-test-remote` 固定入口对非交互 SSH 会话可见。浏览
窗口不使用 GPU；关闭这个窗口只会关闭结果访问，不会停止另一个窗口中正在执行的试验。

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

## 5. 当前尚未开放

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
