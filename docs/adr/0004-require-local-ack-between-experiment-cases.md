---
status: superseded
superseded_by: 0010-use-determined-for-gpu-integration-smoke.md
---

# 每个工况结束后要求本地续行确认

> 当前 GPU 执行环境已改由 Determined 0.38.1 调度；本 ADR 的“无调度器、普通 SSH 直接执行”假设已由 ADR 0010 取代。本文仅保留历史设计背景。

历史设计曾假设 Docker 服务器没有调度器或 systemd，且试验不应在本地电脑失联后自主展开。
远程编排器允许已经开始的一个有限工况完整收尾并保存结果，但只有收到本地控制程序针对该
工况的明确续行确认后才启动下一个工况；确认超时则把运行标记为 `interrupted` 并退出。
该边界用一次工况级握手替代持续心跳、总墙钟硬上限和 tmux 后台运行。
