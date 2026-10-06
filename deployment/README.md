# Determined 单 GPU 集成冒烟交付说明

日常查看已部署服务器的结果，双击仓库根目录 `open-server-results.cmd`。它使用个人
controller 配置中的 SSH 别名和端口，无需 GPU 或重新部署。配置缺失时从
`newton-test.cmd` 的“9. 设置”填写；结果服务读取服务器已有个人 `server.toml`。

部署目录已按用途整理：`guides/` 保存步骤，`templates/` 保存通用任务模板，
`history/` 保存带原日期的验收记录，`local/README.md` 索引忽略的个人资料。
代码整理通过 Git 更新个人 `repo/`，不修改旧源码导出、环境或运行数据。
完整目录与迁移对照见[项目目录与迁移说明](../docs/项目目录与迁移说明.md)。

2026-10-04 接入方式补充：[外部只读资产登记](../docs/外部只读资产.md)已部署到个人开发仓库，并完成一个公共 USD 包的登记及预检阻断核对，尚无新的 GPU trial。采用它时必须为完整公共源目录增加单独只读挂载，且容器内源路径与登记记录一致；入口单位和朝向必须明确，现有 GPU 和干净源码交付限制全部保持，下方入库说明仍适用于自有副本方式。

支持默认无录像及独立的 [GPU 带录像入口](guides/GPU_VIDEO_SMOKE.md)：`smoke-drop-gpu --record-video`。两条路径均有指定节点上的实际运行证据，最新范围见 [当前状态](CURRENT_STATUS.md)。默认不录像的行为保留。

当前交付范围是单 trial、单 GPU、单个中等高度摔落、1000 步的非正式集成入口，可显式要求 GPU 录像。测试资产已完成目标节点的执行、文件校验和播放确认；真实资产和正式批次仍待验证。正式 profile `mujoco-native-dt1ms-v1` 继续为 `reserved_not_runnable`，本地 fake/mock 测试仍不能替代目标运行。

仓库不连接、提交、激活或终止 Determined experiment，不安装系统服务，不修改 SSH、Tailscale 或全局环境。操作员使用 [8 分钟配置模板](templates/determined-gpu-smoke-8min.yaml) 人工提交。

## 1. 分配 GPU 前准备环境

默认模板采用 **镜像内部预装环境**。批准的 GPU 镜像内部必须已有 Python 3.12 解释器以及与本次 `pyproject.toml`、`uv.lock` 完全一致的运行依赖；`NEWTON_TEST_PYTHON` 填写该解释器在镜像内的绝对路径。只有一个“带 Python 的镜像”不满足条件。镜像还须兼容 Determined 0.38.1 的容器启动环境，并提供 Bash 和 GNU `timeout`；不能删除集群需要的启动组件。

环境制作发生在 GPU experiment 激活前。在管理员批准的环境制作位置，可用同一份锁文件预先执行 `uv sync --frozen --no-dev --python 3.12`。这只是环境准备方法，不是 trial entrypoint。开发测试使用默认 dev 组，新增的 `PyYAML==6.0.3` 仅用于真正解析部署 YAML；锁文件因此新增该开发依赖及其发行文件，物理运行依赖版本未升级。

另一种选择是 **独立只读挂载预制环境**。必须经管理员认可，在审核后的配置副本中增加第二个挂载，例如：

```yaml
bind_mounts:
  - host_path: REPLACE_WITH_APPROVED_HOST_DATA_ROOT
    container_path: REPLACE_WITH_TRIAL_DATA_ROOT
    read_only: false
  - host_path: REPLACE_WITH_APPROVED_PREBUILT_ENV_HOST_PATH
    container_path: REPLACE_WITH_COMPATIBLE_ENV_CONTAINER_PATH
    read_only: true
```

这时 `NEWTON_TEST_PYTHON` 指向第二个挂载内的解释器。数据挂载与环境挂载必须分开；数据、环境和源码目录不能互相覆盖。SSH shell 中激活的 venv 不会自动出现在 trial。venv 常含绝对路径、解释器符号链接及 native extension，不能直接把 Windows venv 搬到 Linux，也不能任意改变 Linux venv 的绝对路径。管理员须确认 Python 基础解释器路径、系统、glibc、CPU 架构、CUDA 用户态库和宿主驱动兼容性；环境只读，但运行用户仍须有可写的临时目录和 Warp 编译缓存目录。

trial 只检查并直接使用已经存在的 `NEWTON_TEST_PYTHON`。执行前检查 Python 3.12 与锁定运行依赖的完整传递版本集合，失败则在 Warp GPU 枚举前退出。项目入口禁止在 slot 内执行 `uv sync`、普通 `uv run`、`pip install` 或下载依赖。平台自己的启动依赖处理另行看待，不应误记为项目环境安装。

## 2. 导出确定的源码

GPU runner 拒绝脏工作树、错误 commit、缺失来源信息或被修改的导出包。当前修改必须经过审核、由操作员提交并形成干净 commit 后，才能产生部署包；不要给脏代码随便填写 HEAD 充当版本。

从干净 checkout 使用已准备好的 Python 导出到一个全新的、位于源码和数据目录之外的目录：

```text
python -B -m experiment_runner.release <new-source-export-directory>
```

导出器只读取 Git，不提交或部署。它读取指定 commit 的已跟踪文件内容，拒绝覆盖已有目录，并生成 `.newton-source.json`，记录完整 commit 及运行源码、静态页面、依赖声明的 SHA-256。所有 CPU/GPU manifest 另外记录工作树状态与源码指纹。GPU trial 在没有 `.git` 时校验这份导出清单；不要移除 `.newton-source.json`，也不要用 `.detignore` 排除清单或运行文件。该清单保证传输和内容一致性，不是签名或不可伪造认证。

上传目录只使用上述源码导出；不要包含本机 venv、私钥、配置或真实数据集。资产仅从管理员批准的数据挂载读取，使用已验收入库且 `drop` 就绪的完整 identity 和 64 位版本。模板内只保存占位符，不保存真实服务器地址、账号、路径或资产版本。具体环境笔记和配置副本保存在 `.gitignore` 已覆盖的 `deployment/local/`；该目录不随 Git 部署。

同一代码版本可复用同一源码导出，只改资产或任务配置时无需重新导出。每次创建 experiment 仍上传该源码上下文，Python 环境、图形库和资产通过挂载复用。源码变更则需新的干净提交和新的导出目录；不覆盖旧导出。

## 3. 人工复核配置与 paused 提交

复制模板到源码导出之外，替换并核对以下项目：

| 配置 | 要求 |
| --- | --- |
| `environment.image` | 已获准的 GPU 镜像，优先固定版本或 digest；用户明确接受浮动标签时记录限制，不为此索取管理凭据 |
| `resources.resource_pool` | 管理员批准的单 GPU 资源池 |
| 数据挂载的 host/container path | 已批准、运行用户拥有、固定布局完整、可写，与源码隔离 |
| `NEWTON_DATA_ROOT` | 与数据挂载的 container path 一致 |
| `NEWTON_SMOKE_ASSET_IDENTITY` | 已验收入库且 drop 就绪的资产身份 |
| `NEWTON_SMOKE_ASSET_VERSION` | 完整 64 位小写 SHA-256 |
| `NEWTON_TEST_GIT_COMMIT` | 源码导出清单中的完整 40 位 commit |
| `NEWTON_TEST_PYTHON` | trial 内实际存在的预制 Python 3.12 绝对路径 |

SSH shell 的环境不会自动传入 trial；这些设置必须出现在审核后的 experiment 配置中。不要手填 `DET_*` 或伪造 GPU 可见性，它们应由调度器和容器运行时提供。

模板固定 `slots_per_trial: 1`、`max_slots: 1`、`max_restarts: 0`、single searcher、`hyperparameters.global_batch_size: 1`，因此不进行参数扫描。`checkpoint_policy: none` 和 `/tmp/newton-test-checkpoints` directory storage 避开默认共享 checkpoint 路径。Bash 中的 `timeout --signal=INT --kill-after=30s 8m` 限制整个 Python runner；物理循环另有 300 秒上限。GPU 首次编译也占用外层预算，预算内未完成即失败，不能因此把结果认作成功。

人工确认 `REPLACE_WITH_` 搜索结果为零后，可由操作员运行：

```text
det experiment create --paused <reviewed-config.yaml> <new-source-export-directory>
```

`--paused` 创建时不激活 trial，也不分配 GPU。本仓库不会自动运行该命令。操作员检查集群接受的最终配置、镜像环境和挂载后，才人工激活一次。目标集群的实际配置校验仍是部署验收的一部分。

配置结构已对照 Determined 0.38.1 官方 [experiment schema](https://github.com/determined-ai/determined/blob/v0.38.1/schemas/expconf/v0/experiment.json)、[resources schema](https://github.com/determined-ai/determined/blob/v0.38.1/schemas/expconf/v0/resources.json)、[single searcher schema](https://github.com/determined-ai/determined/blob/v0.38.1/schemas/expconf/v0/searcher-single.json) 和 [directory checkpoint schema](https://github.com/determined-ai/determined/blob/v0.38.1/schemas/expconf/v0/directory.json) 核对；本地测试解析 YAML 的真实层级和值类型。这不能替代目标 master 的配置验证。

入口依赖进程退出码表示运行结束，不调用训练指标 API；`integration_smoke_exit_code` 是配置中的 metric 名称，不是假造的训练指标。该版本的 [allocation 退出处理](https://github.com/determined-ai/determined/blob/v0.38.1/master/internal/task/allocation.go) 和 [trial 退出处理](https://github.com/determined-ai/determined/blob/v0.38.1/master/internal/trial.go) 将正常退出纳入完成路径。最终验收仍须同时查看调度状态和结构化产物。

## 4. GPU 执行边界与产物验收

资产完整性/readiness、Determined 元数据、源码及依赖必须先通过检查；之后才允许 Warp GPU 枚举、逻辑 `cuda:0` context、UUID 一致性检查、Newton 模型/solver 和固定 1000 步。`DET_SLOT_IDS` 必须是恰好一个非负整数的 JSON 列表；字符串、bool、负数或多 slot 均拒绝。共享几何测量和场景构造必须收到安全门签发的显式强类型许可，正式 profile 即使有许可也拒绝。原 CPU 录像函数仍仅接受 CPU 冒烟；独立 GPU recorder 只接受固定 video-smoke profile，并验证 NVIDIA EGL 与已分配设备身份。

环境变量及许可均是防误用机制；资源隔离由 Determined 和 NVIDIA container runtime 提供。只允许逻辑 `cuda:0`，不接受宿主机编号、不选择空闲卡、不回退 CPU。Warp UUID 正常时必须与分配 UUID 规范化匹配；不匹配或非空非法格式在 Newton 模型创建前失败。UUID 缺失只能记 `unavailable` 和 warning，不能伪造 matched。

首个 solver step 返回且 CUDA 同步成功前，`actual_compute_device=unknown`、`cuda_used=false`、`gpu_physics_started=false`、`completed_physics_steps=0`。逐步记录尝试序号及完成数；捕获到失败或 Ctrl+C 时落盘最后阶段。完成1000步才可置 `gpu_physics_completed=true`。每100步持久化进度，强制杀进程或主机失联可能保留较早的保守状态；这类运行不得验收，应结合调度日志人工判定。

一次合格集成冒烟需要同时满足：

1. 调度器显示一次 trial 正常结束，退出码为0，资源释放，没有自动重试。
2. `status.json` 和 `cases/001-medium-gpu-integration-smoke/case.json` 是 succeeded，manifest 退出码为0，各处 `authoritative=false`；完整 commit、源码指纹、依赖版本和资产版本与提交配置一致。
3. manifest 与 case 中的 allocation、逻辑设备、allocated/runtime UUID 和验证状态一致；正常设备应为 matched。unavailable 虽允许诊断运行，但必须明确记录为仍缺设备身份验证证据。
4. `completed_physics_steps=1000`、`attempted_physics_step=1000`、`solver_step_completed=true`、`gpu_physics_completed=true`、`actual_compute_device=cuda:0`、`cuda_used=true`、`cpu_fallback=false`，case `finite=true`。
5. `run.log`、`checksums.sha256` 存在且校验通过。默认入口仅保存结构化产物，`recording.status=not_attempted`。带录像入口则必须 `recording.status=succeeded`，校验 MP4、poster、final、preview，并检查 NVIDIA vendor 和 EGL/CUDA 设备匹配。
6. 录像使用 ffprobe 读取实际帧数、尺寸、编码和时长，并人工查看内容；case 中的帧数或文件存在本身不能替代这两步。固定视频规格见 [录像说明](guides/GPU_VIDEO_SMOKE.md)。

结果只证明受测资产在特定环境中的集成链路，不证明资产物理真实性或正式试验有效性。更换资产、驱动、节点或图形环境时须核对相应前置条件。

## 5. 服务器只读就绪检查与结果浏览

在已部署的服务器环境中，操作员可在分配 GPU 前执行：

```text
newton-test-remote check-readiness --json --data-root <approved-data-root> --port 8765
```

检查 Python 3.12、包/命令、锁定依赖、Git/来源、固定目录、所有权/隔离、访问权限、磁盘总量/余量、FFmpeg/H.264、ffprobe 和 `127.0.0.1` 端口。它不导入仿真栈，不初始化 CUDA，不调用 GPU 探针，不写文件或配置，不自动修复。可写检查是权限检查，不能代替目标文件系统实际写入验收；磁盘提示采用1GiB余量阈值，不是正式配额策略。

返回 `schema_version=1`、逐项 ready/blocked、`required_for_results` 和 `results_ready`。只读结果所需项目全部通过才返回0，否则返回2；正式 GPU 未开放、编码依赖或源码脏状态会如实展示，但不单独阻止浏览既有结果。

Windows 菜单先检查远端命令，再调用结构化 readiness。报告缺字段、损坏、超时或只读必要项 blocked 均不启动隧道；允许浏览时两端仍固定 `127.0.0.1`。不开放公网或 Web 提交/删除。

## 6. 当前验证状态

以 [当前状态与下一步](CURRENT_STATUS.md) 为准。历史本地记录见 [2026-09-13 验收记录](history/ACCEPTANCE_RUN_2026-09-13.md)，其中当日未提交和未运行 GPU 的描述保留为历史，不代表今天仍受相同阻塞。

本地 GPU 单元测试使用 fake/mock；实际 GPU 证据来自用户手动执行的 Determined trial、结果校验和播放确认。下一步是选定并复制一个真实资产，不重复已完成的小方块链路。

## 手动查看结果

以下是两个不同终端中的命令模板，实际协作时每次只指导一步。无需 GPU 分配，但开发容器必须能够读取共享结果。

开发容器终端在代码目录中启动服务：

```text
<prepared-python> -B -m experiment_runner.cli serve-results --data-root <shared-data-root> --host 127.0.0.1 --port 8765
```

服务准备静态页面和索引，然后只读提供网页、JSON 和文件；HTTP API 不支持提交、修改或删除。该进程不重新仿真或渲染，不申请 GPU。个人开发容器不一定是集群登录节点。

用户电脑上的另一个终端建立转发：

```text
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 <user>@<development-container-address>
```

浏览器打开 `http://127.0.0.1:8765/`。第一个回环地址属于用户电脑，转发目标回环地址属于 SSH 远端开发容器。视频来自共享存储，在本地浏览器解码和显示。

手动方式需要两个终端保持运行，关闭任一终端会中断查看，但不会删除结果或停止已独立提交的 Determined 任务。跨对话不假定两个进程仍存活；有访问问题时只检查当前服务和隧道，不重新运行仿真。菜单自动浏览的 readiness、远端 PATH 和个人配置要求独立适用，不能把手动通道成功当作自动菜单已验收。
