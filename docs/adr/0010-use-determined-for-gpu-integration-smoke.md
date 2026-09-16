---
status: accepted
---

# GPU 集成冒烟由 Determined 分配单个设备

后续扩展：`smoke-drop-gpu --record-video` 使用独立的非正式 video-smoke 配置，沿用单 GPU 与限时约束，要求 EGL 渲染和 H.264 录像。渲染检查随该任务执行，失败不回退、不声称录像完成。首版无录像行为保留为默认路径，正式配置仍不可运行。配置和设备映射见 [录像说明](../../deployment/GPU_VIDEO_SMOKE.md)。

目标集群使用 Determined 0.38.1。GPU 工作负载必须由操作者以 Determined experiment 提交，`slots_per_trial: 1` 负责把一块已分配 GPU 暴露给 trial；Newton-Test 只在 trial 内以前台命令运行，并选择容器逻辑 `cuda:0`，不接受宿主机编号、不自行提交任务、不启动后台服务、不回退 CPU。

入口在任何 Warp/CUDA 初始化前要求 Determined 注入的 experiment、trial、task、allocation、单 slot 和 `DET_TASK_TYPE=TRIAL` 元数据，并要求 `NVIDIA_VISIBLE_DEVICES` 恰好为一个规范 GPU UUID；随后 Warp 仍须恰好发现一个设备。这是防止在普通单卡主机上误用入口的操作安全门，不是不可伪造的认证；实际资源授权和隔离仍由 Determined 与 NVIDIA 容器运行时承担。安全门通过后的运行审计分开记录 CUDA context、模型设备、首个 solver step 和完整 1000 步，不能把“选中设备”提前写成“GPU 物理已运行”。

首个入口仅使用独立的 `mujoco-warp-cuda-dt1ms-integration-smoke-v1` profile 执行一个固定时限的中等高度摔落 integration smoke，始终 `authoritative=false`。正式 `mujoco-native-dt1ms-v1` 保持不可执行。目标 GPU 的无头 OpenGL/EGL 录像尚未验证，因此首版只保存真实结构化物理结果并明确记录录像未尝试。

trial 配置必须显式提供批准的镜像、资源池、数据挂载、完整资产版本、完整部署 commit 和预先准备好的 Python 3.12 可执行文件。依赖环境在 experiment 激活、占用 GPU 前准备；trial 内不执行普通 `uv run --frozen` 或任何下载、解析、安装依赖的路径。

本 ADR 取代 ADR 0004 中“目标环境没有调度器、通过普通 SSH 直接运行 GPU 工况”的执行假设；ADR 0004 的工况续行设计不适用于本次单工况、受限时 integration smoke，也不因此开放正式批次。

实现补充：`DET_SLOT_IDS` 必须是单个非负整数的 JSON 列表；安全门签发进程内强类型执行许可，共享 GPU 函数无许可或收到正式 profile 时直接拒绝。分配 UUID 与 Warp UUID 在模型创建前核对，不匹配或非法值直接失败；缺失值只记录 unavailable/warning。每个 solver step 必须完成 CUDA 同步才计入完成数，失败/中断记录正在尝试的序号及已完成步数。

GPU 部署来源只接受干净且 commit 匹配的 checkout，或 `experiment_runner.release` 生成的内容校验导出；不允许仅填一个 commit 字符串指代脏代码。模板默认环境位于镜像内，也允许经管理员批准的独立只读环境挂载。trial 在 GPU 枚举前检查 Python 3.12 和锁定运行依赖。详细操作步骤与尚未完成的目标验证见 [部署说明](../../deployment/README.md)。
