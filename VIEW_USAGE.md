# view.py 使用文档

`view.py` 用 Newton Viewer 打开 Artiverse 数据集里的 URDF 模型，并提供关节滑条、资产快速切换、位置显示、双面渲染和右键拖拽力显示。

## 代码目录

`view.py` 仍是兼容启动入口，原来的命令不变。目录按职责整理为：

- `asset_viewer/assets.py`：资产发现、路径描述和 articulation metadata
- `asset_viewer/controls.py`：关节控制与 ImGui 交互
- `asset_viewer/traction.py`：右键拖拽牵引力的计算和显示
- `asset_viewer/app.py`：Newton 模型构建、Viewer 运行时和命令行
- `collision_probe.py`：独立的碰撞探针程序
- `third_party/artiverse_official/upstream/`：固定版本的 Artiverse 官方查看器代码

项目自己的官方查看器兼容代码只放在
`third_party/artiverse_official/` 的上层，不会修改 `upstream/` 快照。

默认读取目录：

```text
D:\Datasets\Artiverse\dataset_chunks\data
```

如果以后继续往这个目录里加入新类别，只要模型目录里有 `urdf_w_collider/*.urdf`，`view.py` 会自动递归发现。

## 基本用法

查看所有可打开的模型：

```powershell
python view.py --list
```

只查看某一类模型，例如微波炉：

```powershell
python view.py --category microwave --list
```

打开默认列表里的第一个模型：

```powershell
python view.py
```

打开某一类里的指定模型：

```powershell
python view.py --category microwave --index 12
```

打开某个具体目录或 URDF 文件：

```powershell
python view.py D:\Datasets\Artiverse\dataset_chunks\data\microwave
python view.py D:\path\to\model\urdf_w_collider\model.urdf
```

## 重要说明

`--index` 是当前搜索结果里的编号。

所以：

```powershell
python view.py --index 12
```

表示“所有类别排序后的第 12 个模型”。

如果想打开 microwave 的第 12 个模型，需要写：

```powershell
python view.py --category microwave --index 12
```

## Viewer 侧边栏

### Assets

用于快速切换资产：

- `Reload`：重新加载当前模型
- `Previous` / `Next`：切换上一个或下一个模型
- `Asset List`：按 `类别 / 来源 / 模型 ID` 分组列出所有模型，点击即可切换

切换模型后，程序会重新加载 URDF、关节、metadata 和当前 UI。

### Joint Controls

显示模型中可活动的关节。

常用按钮：

- `Reset all`：恢复初始关节状态
- `Zero all`：所有关节归零

每个关节会显示一个滑条。转动关节通常用角度显示，平移关节用数值显示。

程序不会根据资产类别或关节名称自动修改姿态。初次加载和 `Reset all` 都使用 URDF 声明的原始关节状态；`Zero all`、关节滑条和 metadata motion state 只在用户主动操作时生效。

### Position Display

显示每个 body 的世界坐标。

- `Show body positions`：是否在 UI 中显示 body 坐标
- `Print positions`：把所有 body 坐标打印到终端

### Articulation Metadata

读取同目录下的 `*.articulations.json`。

这里会显示：

- articulation 数量
- motion state，例如 `closed`
- 可能存在的 dependency 字段
- 原始 articulation 行

如果有 motion state，可以点击 `Apply closed` 之类的按钮应用状态。

### Rendering Options

常用项：

- `Double-sided Meshes`：双面显示 mesh，解决某些面从背面看不到的问题
- `Show Visual`：显示视觉 mesh
- `Show Inertia Boxes`：显示惯性盒

## 物理模拟和拖拽

默认是静态查看模式。如果要用右键拖拽物体并计算牵引力，需要开启模拟：

```powershell
python view.py --simulate
```

模拟默认暂停在资产的原始姿态，便于先检查模型。需要运行物理时可在 Viewer 中开始播放，或者使用下面的命令让它加载后立即运行：

```powershell
python view.py --simulate --start-running
```

开始播放后，程序遵循 URDF 自身的质量、惯量、关节限制和动力学参数。如果资产没有提供关节阻尼、摩擦或驱动，部件在重力下转动可能反映的是资产数据质量问题。

右键拖拽模型时，侧边栏会显示 `Traction Force`：

- `Force xyz`：右键拖拽施加到 body 上的力
- `Force magnitude`：力的大小
- `Torque xyz`：对应力矩
- `Pick point world`：被拖拽点的世界坐标
- `Target world`：鼠标拖拽目标点的世界坐标
- `Clamp`：拖拽力是否触发限幅

如果不想在终端打印拖拽力：

```powershell
python view.py --simulate --no-print-traction-force
```

调整终端打印频率：

```powershell
python view.py --simulate --traction-print-hz 30
```

## 碰撞探针

`collision_probe.py` 会创建一个运动学小球，让它沿指定方向穿过资产碰撞体，并显示、
记录实际 contact。它不依赖视觉 mesh 的外观，因此适合判断碰撞体是否错位、尺寸错误
或存在空洞。

```powershell
python collision_probe.py --category microwave --index 0
python collision_probe.py D:\path\to\model\urdf_w_collider\model.urdf
```

探针直接使用 `asset_viewer` 包的资产发现和渲染 Interface，不再通过兼容入口
`view.py` 导入。

## Artiverse 官方查看器对照

本项目已把官方 GitHub 仓库中的 `view_model.py`、BlenderProc 渲染脚本和 HDRI
按固定 commit 放入 `third_party/artiverse_official/upstream/`。官方查看器读取
`*.segmented.glb`，输出每个 part 的高亮/纹理视频；它不是 Newton URDF
交互窗口，适合用来对照“原始分割 GLB 是否正常”。

安装与运行说明见
`third_party/artiverse_official/README.md`。建议先执行：

```powershell
python third_party/artiverse_official/run_view_model.py `
  --model-path D:\path\to\category\source\MODEL_ID `
  --output-dir official_renders\MODEL_ID `
  --check
```

本地 Adapter 会兼容预发布数据只有 `*.articulations.json` 的情况，所有转换都在
临时目录完成，不会改写资产文件。上游来源、commit、包含范围和 SHA-256 记录在
`third_party/artiverse_official/PROVENANCE.md`。

## 常用参数

```powershell
python view.py --category scissors --index 3
python view.py --category microwave --index 12 --simulate
python view.py --category lighter --show-colliders
python view.py --category piano --scale 0.5
python view.py --category microwave --initial-motion-state closed
python view.py --headless --frames 1
```

参数说明：

- `--category NAME`：只搜索某个类别，可重复使用
- `--index N`：打开搜索结果中的第 N 个模型
- `--simulate`：初始化物理模拟，默认暂停在原始姿态
- `--start-running`：与 `--simulate` 一起使用，加载后立即开始运行物理
- `--scale VALUE`：导入模型时缩放
- `--z VALUE`：导入模型时增加高度偏移
- `--ground`：添加地面
- `--show-colliders`：显示碰撞 mesh
- `--double-sided / --no-double-sided`：开关双面渲染
- `--initial-motion-state NAME`：应用 articulation JSON 中的 motion state
- `--headless --frames 1`：无窗口快速测试能否加载

## 常见问题

### 为什么旧的 microwave index 变了？

因为现在默认搜索所有类别。旧的 microwave 索引需要加上类别过滤：

```powershell
python view.py --category microwave --index 12
```

### 为什么有些面背面看不到？

开启双面显示：

```powershell
python view.py --double-sided
```

也可以在 UI 的 `Rendering Options` 中勾选 `Double-sided Meshes`。

### 为什么拖拽力一直是 0？

需要满足两个条件：

1. 使用 `--simulate`
2. 在 Viewer 里右键拖住可运动 body

静态查看模式不会施加拖拽力。

### 为什么某些模型看起来很奇怪？

Artiverse 的 URDF 主要用于 articulated object 交互和仿真查看，不一定是高保真的真实物理参数。部分模型的 COM、质量、惯性和视觉 mesh 可能只是简化值。
