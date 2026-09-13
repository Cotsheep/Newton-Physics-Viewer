# Newton-Test

Newton-Test 用于查看 URDF、USD、GLB 资产，并逐步建立基于 Newton–MuJoCo 的 3D 资产物理合理性检查流程。项目默认物理求解器为 MuJoCo。

当前版本已经打通桌面资产查看、本地 CPU 摔落与固定坡度冒烟、资产验收入库和浏览器结果查看，并实现了一个等待目标集群验证的独立 GPU integration smoke 入口；正式服务器 GPU 批次仍未开放。README 会把“已经实现”“已经验证”和“正式能力”分开说明。

## 当前能力

| 功能 | 当前状态 | 说明 |
| --- | --- | --- |
| URDF、USD、GLB 桌面 Viewer | 可用 | 支持关节控制、碰撞显示、拖拽、相机控制和多种求解器 |
| 资产验收入库 | 可用 | 解析 USD 依赖、计算内容版本并分别检查摔落与坡度就绪状态 |
| 本地 CPU 摔落冒烟 | 可用 | 只运行一个中等高度工况，用于验证代码和结果链路 |
| 本地 CPU 坡度冒烟 | 可用 | 只运行固定 25°、2 秒的单案例开发工况，不产生正式摩擦结论 |
| 本地结果 Web UI | 可用 | 按资产封面组织结果，在页面中直接播放工况录像 |
| 单 GPU 摔落 integration smoke | 已实现，目标集群未验证 | 只允许 Determined trial 内一个中等高度、1 秒、非正式结构化工况，不开放录像或参数扫描 |
| SSH 远程结果浏览 | 可用，但服务器需预先部署 | 经 Tailscale 网络和个人 SSH 身份建立按需只读隧道 |
| 服务器就绪检查 | 已实现，本地验证 | `check-readiness --json` 检查环境、存储、编码和回环端口；不导入仿真栈或探测 GPU |
| 正式 GPU 批次 | 未开放 | 等待管理员控制的 GPU 授权策略和远程集成验收 |
| 正式多高度摔落、多角度坡度试验 | 未开放 | 当前菜单只提供两个非正式 CPU 单案例冒烟入口 |
| Web UI 提交或删除结果 | 未开放 | 当前 Web UI 只读，不提供任务提交、回收或永久删除 |

本地两类数据必须分开：

```text
Viewer 资产源目录 ──> 桌面 Viewer（读取原始数据集）

试验数据目录
├── inbox/          ──> 待验收资产包
├── assets/         ──> 已验收的不可变资产版本
├── runtime/runs/   ──> 运行结果和视频
└── web/            ──> 只读结果页
```

不要把 Viewer 原始数据集目录直接设置为试验数据目录。

## 快速开始

项目使用 Python 3.12、`pyproject.toml` 和 `uv.lock` 固定依赖。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在仓库根目录执行：

```powershell
uv sync --frozen
uv run newton-test
```

Windows 也可以直接双击仓库根目录的 `newton-test.cmd`。该文件只检查并调用已经安装的 `uv`，不会自行下载软件或修改系统环境。

首次启动会引导保存以下非敏感设置：

- Viewer 资产源目录；
- 试验数据目录；
- Windows `~/.ssh/config` 中的服务器别名；
- 本地与远程结果页端口；
- 是否自动打开浏览器。

不需要的设置可以暂时跳过。个人配置默认保存在：

```text
~/.config/newton-test/controller.toml
```

配置中不保存密码、SSH 私钥、真实服务器地址或服务器用户名。

## 统一菜单

`uv run newton-test` 不会直接启动物理引擎，而是打开持续运行的中文控制菜单：

```text
1. 打开桌面资产 Viewer
2. 验收 inbox 中的资产
3. 运行本地 CPU 摔落冒烟
4. 运行本地 CPU 坡度冒烟
5. 打开本地结果页
6. 打开远程结果页
7. 查看已验收资产与就绪状态
8. 查看项目能力状态
9. 设置
0. 退出
```

普通操作完成后会返回主菜单。结果页占用当前终端；按 `Ctrl+C` 关闭只读服务后返回菜单，不会删除结果。菜单不会创建后台服务，也不会自动部署或更新服务器。

### 打开桌面 Viewer

选择菜单第 1 项后，可以使用已保存的 Viewer 资产源目录，也可以临时输入某个 URDF、USD 或 GLB 路径。Viewer 是本地交互式程序，根据所选求解器可能使用本机 GPU。

完整操作和参数参见 [VIEW_USAGE.md](VIEW_USAGE.md)。

### 验收资产

把一个完整资产包放到：

```text
<试验数据目录>/inbox/<资产身份>/
└── newton-mujoco.usda
```

入口 USD 引用的 mesh、纹理和其他 USD 文件也必须位于同一个资产包中。选择菜单第 2 项后，程序会自动扫描可验收资产，不要求手写内部路径。

验收会生成带完整 Git commit 的报告，并把通过至少一个试验模板检查的内容作为不可变资产版本放入 `assets`。资产可能处于“部分就绪”状态，例如能够运行摔落检查，但缺少坡度试验所需的摩擦属性。

### 运行本地 CPU 冒烟

选择菜单第 3 项会列出 `drop` 就绪的资产；第 4 项只列出 `slope_friction` 就绪的资产。只需选择编号，不需要复制完整 SHA-256 版本号。启动前会显示资产、版本、工况和计算资源摘要，并默认要求明确确认。

当前实现只执行：

- MuJoCo 与 Warp 固定为 CPU 物理解算，不使用 CUDA；
- 一个中等高度摔落工况，或固定 25° 的单案例坡度工况；
- 2 秒物理推进，录像开头另有 0.5 秒静止展示；
- `640×360`、50 FPS、总长约 2.5 秒的录像；
- 非权威结果，`authoritative` 为 `false`。

Windows 的录像使用系统 OpenGL，可能使用本机图形 GPU 完成图形渲染；这不等于正式服务器
GPU 物理解算。Linux 冒烟会在导入仿真栈前隐藏 CUDA、强制 Mesa 软件
OpenGL，并在推进物理时间前验证渲染器为软件实现；验证失败就停止，不回退到 NVIDIA GPU。
结果 `manifest.json` 记录 Git commit、锁定软件版本、CPU/平台信息和实际渲染器。这个结果只能
证明代码、资产导入、录制和浏览链路可以运行，不能作为正式 GPU 资产物理结论。坡度冒烟只报告 `moved`、`stayed_near_start` 或 `inconclusive` 开发观察，不测量临界坡度，也不产生正式摩擦结论。

### 查看结果

本地结果页只绑定 `127.0.0.1`。首页使用不含地面或释放工况的独立资产封面组织内容，点击
封面进入资产试验页，再点击视频画面直接播放对应工况。

结果页支持视频进度条和下载，但不提供删除按钮。当前如需处理结果，只能通过以后实现的完整运行回收流程；不要单独删除某个 `video.mp4`，否则会破坏运行结果的一致性。

### GPU integration smoke（服务器侧开发入口）

服务器侧新增了独立命令 `newton-test-remote smoke-drop-gpu`，但它不会出现在普通本地菜单或 Viewer 中，也不负责提交 Determined experiment。它只应由操作者在 `slots_per_trial: 1` 的 Determined trial 内前台调用。命令会在任何 Warp/CUDA 初始化前 fail-closed 检查 trial、allocation、单 slot 元数据和唯一的 `NVIDIA_VISIBLE_DEVICES=GPU-...` UUID；这只是防误用安全门，不是不可伪造的认证，真正隔离由 Determined/NVIDIA runtime 提供。随后 Warp 仍必须恰好发现一块设备，并且只选择容器逻辑 `cuda:0`。命令不接收宿主机 GPU 编号，发现元数据缺失、零块、多块、初始化失败、模型设备错误或 CPU solver 时立即失败，绝不回退 CPU。

该命令使用独立 profile `mujoco-warp-cuda-dt1ms-integration-smoke-v1`：Newton `SolverMuJoCo`、MuJoCo native contacts、MJWarp/CUDA、一个中等高度摔落、1 秒物理时间、300 秒内部墙钟上限，且始终 `authoritative=false`。它不会调用正式 profile `mujoco-native-dt1ms-v1`；后者继续保持 `reserved_not_runnable`。

审计状态会分别记录调度元数据门、进程可见 GPU 数、选定逻辑设备、CUDA context、Newton 模型设备、solver step 的开始/完成、实际计算设备和 CPU fallback。首个 solver step 成功前不会写入 `actual_compute_device=cuda:0` 或 `cuda_used=true`，完整 1000 步前不会写入 `gpu_physics_completed=true`；失败结果保留最后成功阶段。

底层设备选择、几何测量和场景构造必须持有安全门签发的强类型 GPU 许可，正式 profile 即使持有许可仍拒绝。`DET_SLOT_IDS` 严格要求一个非负整数。分配 UUID 与 Warp UUID 规范化核对：不匹配或非法值在模型创建前失败，缺失则明确记录 unavailable/warning。每个 GPU step 在 CUDA 同步成功后才计为完成，避免把异步提交当作执行成功。

GPU 入口还要求锁定依赖，以及干净且匹配的 Git checkout 或经 SHA-256 校验的源码导出；脏工作树不作为可部署版本。导出方法、镜像内环境与独立只读环境挂载的区别，以及首次实际运行验收条件均见[部署说明](deployment/README.md)。

目标容器的 GPU 无头 OpenGL/EGL 录像链路尚未验证。当前入口只保存真实的 manifest、status、case、日志和 checksums，并明确记录 `recording.status=not_attempted`；不会套用 CPU 的 Mesa 软件渲染边界，也不会伪造 preview、poster、final 或 MP4。Determined 示例使用显式镜像/资源池/挂载占位符、完整 commit 和预先准备好的 Python 3.12；trial 内不会同步依赖。完整部署边界见 [deployment/README.md](deployment/README.md)。

## 远程结果浏览

远程结果浏览需要满足：

1. Windows 已能通过 Tailscale 地址和个人 SSH 配置登录服务器；
2. 服务器已由操作者按部署文档手动准备好锁定环境；
3. `newton-test-remote` 在非交互 SSH 会话的 `PATH` 中可见；
4. 服务器个人配置已指定试验数据根目录。

选择菜单第 6 项后，本地程序先执行只读就绪检查，再建立：

```text
本地浏览器
  -> Windows 127.0.0.1
  -> SSH 本地端口转发（连接经过 Tailscale）
  -> 服务器 127.0.0.1
  -> 只读结果服务
```

两端都不监听公网或局域网地址，不配置 Tailscale Serve。按 `Ctrl+C` 或关闭控制窗口会结束本次结果访问；只读服务不会启动物理仿真，但本地浏览器播放视频时可能使用本机图形加速。统一入口不会自动上传代码、安装依赖、修改服务器 `PATH` 或更新仓库。

建立隧道前，先检查远端命令，再调用 `check-readiness --json`，用中文显示每项结果。只读必要项 blocked、报告缺失或超时都会阻止隧道；正式 GPU 未开放不阻止浏览既有结果。

当前远程部署和正式试验仍以设计路线为准，详见[远程物理试验运行与结果查看路线](docs/远程物理试验运行与结果查看路线.md)。

## 底层命令

`experimentctl.py`、`run_physics_experiment.py`、`newton-test-control` 和 `newton-test-remote` 继续作为开发、自动化和服务器侧底层入口保留。日常使用优先选择 `uv run newton-test`，无需记忆这些命令的参数。

GPU integration smoke 的底层命令为：

```text
newton-test-remote smoke-drop-gpu <asset-identity> <full-asset-version> --data-root <data-root> --git-commit <40-hex-commit>
```

它只执行 trial 内工作，不登录或提交 Determined，也不代表正式 GPU 验证。

当前可运行链路的底层命令和产物格式参见[本地 CPU 冒烟与结果页使用说明](<docs/本地 CPU 冒烟与结果页使用说明.md>)。

## 文档

- [view.py 桌面 Viewer 使用手册](VIEW_USAGE.md)
- [本地 CPU 冒烟与结果页使用说明](<docs/本地 CPU 冒烟与结果页使用说明.md>)
- [远程物理试验运行与结果查看路线](docs/远程物理试验运行与结果查看路线.md)
- [Newton–MuJoCo 3D 资产物理合理性仿真检查方案](<docs/Newton–MuJoCo 3D 资产物理合理性仿真检查方案.md>)
- [项目领域语言与已确认边界](CONTEXT.md)
- [Determined 单 GPU integration smoke 示例](deployment/README.md)

## 开发验证

在锁定环境中运行测试：

```powershell
uv run --frozen python -m unittest discover -s tests -v
```

真实资产、试验数据、视频、个人配置和 SSH 凭据不应提交到仓库。
