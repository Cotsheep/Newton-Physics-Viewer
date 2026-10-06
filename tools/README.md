# 辅助工具

`record_collision_comparisons.py` 读取原资产及同级修复副本，按原有参数完成筛选和
对比录像。由仓库根目录调用：

```powershell
uv run --frozen python tools/record_collision_comparisons.py --help
```

脚本从自身位置定位项目导入；输入及输出目录仍使用原有命令行参数。
实际使用说明见[Viewer 使用文档](../docs/VIEW_USAGE.md#碰撞体修复前后对比)。
个人准备、诊断和页面预览脚本归入忽略的 `deployment/local/tools/`，不随 Git 部署。
