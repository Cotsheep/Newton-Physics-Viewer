# 单 GPU 带录像冒烟

在原 `smoke-drop-gpu` 命令后增加 `--record-video`，选择独立的
`mujoco-warp-cuda-dt1ms-video-smoke-v1` 配置。默认命令仍不录像，正式试验配置仍不可执行。
新入口需要目标集群实际运行验证，本地测试不能证明目标容器的渲染兼容性。

## 运行范围

- Determined 单 trial、单 GPU、一次中等高度摔落，物理时间 1 秒，步长 0.001 秒。
- 录像先展示 0.5 秒初始画面，再每 20 个已完成的物理步取一帧；640×360、50 FPS，共 75 帧。
- 物理解算仍使用 MJWarp/CUDA，无 CPU fallback。H.264 使用已有 FFmpeg 的 CPU `libx264` 编码器，不需要 NVENC。
- 数据目录可写，预制环境可只读挂载；缓存写到容器 `/tmp`。trial 内不安装依赖。
- 程序保留物理循环 300 秒限时，并对渲染、录制过程检查 300 秒限时；外部 `timeout` 限制整个 Python 进程为 8 分钟加 30 秒终止宽限。外部限制也覆盖阻塞的渲染或编码调用。

## 配置与运行

使用 [带录像的任务模板](determined-gpu-video-smoke-8min.yaml)，填写与现有部署相同的数据、环境挂载及资源池；源码必须重新提交、通过 Git 更新并导出为新的确定版本。不要复用旧源码导出后仅修改 commit 字符串。

独立挂载预制环境时，增加只读的环境挂载，覆盖基础 Python 和 venv 的共同父目录，并让 `NEWTON_TEST_PYTHON` 指向 trial 内的解释器。保持相同绝对路径。

与无录像模板相比，主要新增：

```yaml
environment_variables:
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  - PYOPENGL_PLATFORM=egl
  - PYGLET_HEADLESS=true
```

上面的列表需要合并进完整模板，而非覆盖其余变量。`graphics` 用于暴露 OpenGL/EGL 所需的驱动组件，必须在容器创建时配置；在已启动的 Python 中修改该变量不会补上驱动挂载。参见 [NVIDIA 驱动能力说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)。实际容器仍须具备相应用户态系统库。

命令格式：

```text
<prepared-python> -B -m experiment_runner.cli smoke-drop-gpu <identity> <version> --data-root <data-root> --git-commit <commit> --record-video
```

仍由操作者逐步创建 paused experiment、核对配置并激活。不需要另行提交渲染探针，渲染检查随这次录像运行执行。

## 设备与错误处理

执行前沿用原有的资产、源码、锁定依赖、调度元数据及单 CUDA 设备检查。
渲染器使用 EGL 无头模式，根据 [EGL_NV_device_cuda](https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt)
将 EGL 设备映射到当前进程唯一可见的逻辑 CUDA 设备，并核对创建的 OpenGL context 所属 EGL 设备和 NVIDIA vendor。
无法唯一映射时直接失败，不遍历尝试其他 GPU，不改用软件渲染。

第一帧的真实 CUDA/OpenGL 取帧发生在推进物理前。渲染、取帧或编码失败会令整个录像运行失败；已实际完成的物理步数仍保留。物理完成但编码失败，也不能报告录像成功。原 CPU 冒烟的强制软件渲染规则不变。

`case.json` 和 manifest 的 `recording` 字段记录未尝试、运行中、成功、失败或中断状态，并在失败时保留阶段：`renderer_setup`、`frame_readback`、`encoding`、`simulating`、`encoder_finalize` 或 `renderer_close`。结合 trial 日志中的原始异常定位系统库或驱动问题。

成功的录像运行将保存每个工况的 `video.mp4`、`poster.jpg`、`final.jpg` 及运行预览 `preview.jpg`，并纳入结果校验清单。现有结果索引可提供对应的播放和图片链接。仍需人工检查画面是否正常、相机是否覆盖完整运动过程；文件存在不能替代画面检查。
