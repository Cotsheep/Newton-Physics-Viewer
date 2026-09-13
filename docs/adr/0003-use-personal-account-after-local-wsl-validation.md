---
status: accepted
amended_by: 0010-use-determined-for-gpu-integration-smoke.md
---

# 本地 WSL 验证后使用个人集群账号

第一阶段先在操作者的本地 WSL 中开发和验证 Linux 侧代码，再用操作者的个人集群账号完成
远程 NVIDIA GPU 验收；不创建或共享 `newton-runner` 服务账号。这个选择减少对共享集群
账号和系统配置的改动，但 WSL 不具备 NVIDIA GPU，不能产生批准的物理试验结果；远程作业、
结果目录和管理员批准的权限也会与个人账号绑定。受管理集群使用其既有调度器；只有管理员
确认是独立 GPU 服务器后才使用个人 systemd user service。后续证据已确认当前 GPU 任务由
Determined 0.38.1 调度；具体 trial 内执行边界由 ADR 0010 修正，不再使用本 ADR 早期设想的
直接 Docker/SSH 工况托管方式。
