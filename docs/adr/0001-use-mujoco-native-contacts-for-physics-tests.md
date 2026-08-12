---
status: accepted
---

# 物理试验使用 MuJoCo 原生接触

自动物理试验使用 `SolverMuJoCo(use_mujoco_contacts=True)`，不沿用本地 Viewer 的 Newton
接触检测路径。试验的目标是按资产新声明的 MuJoCo 接触参数检查物理表现；让
`mjc:solref`、`mjc:solimp`、摩擦及相关字段经过 MuJoCo 原生接触生成与混合，比保持远程
结果和现有 Viewer 一致更重要。由此产生的结果不与当前 `view.py` 结果直接比较。
