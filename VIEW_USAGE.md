# view.py 使用文档

`view.py` 用 Newton Viewer 打开 URDF 或 GLB 资产，并提供关节滑条、资产快速切换、自动相机取景、精细视角控制、位置显示、双面渲染和右键拖拽力显示。

## 代码目录

`view.py` 仍是兼容启动入口，原来的命令不变。目录按职责整理为：

- `asset_viewer/assets.py`：资产发现、路径描述和 articulation metadata
- `asset_viewer/controls.py`：关节控制与 ImGui 交互
- `asset_viewer/camera.py`：资产包围盒、自动取景和精细相机控制
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

打开某个具体目录、URDF 文件或 GLB 文件：

```powershell
python view.py D:\Datasets\Artiverse\dataset_chunks\data\microwave
python view.py D:\path\to\model\urdf_w_collider\model.urdf
python view.py D:\path\to\asset.glb
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

- `Import Asset...`：打开文件选择窗口，从任意路径选择 `.urdf` 或 `.glb` 文件并立即加载
- `Enable Self Collisions`：开启或关闭 URDF 内部各 link 之间的碰撞，切换后自动重新加载当前模型
- `Collapse Fixed Joints`：合并 URDF 中由 fixed joint 连接的 body；默认关闭，切换后自动重新加载
- `Floating Base`：让 URDF 根 body 通过 FREE joint 自由运动；默认开启，切换后自动重新加载
- `Physics Solver`：选择物理求解器；默认 MuJoCo，切换后自动重新加载并重置当前资产
- `Reload`：重新加载当前模型
- `Previous` / `Next`：切换上一个或下一个模型
- `Asset List`：按 `类别 / 来源 / 模型 ID` 分组列出所有模型，点击即可切换

切换模型后，程序会按文件类型重新构建 URDF articulation 或 GLB 单刚体及当前 UI。

### GLB 资产质量测试

GLB 会作为一个可运动的自由刚体导入，而不是固定的纯显示模型：

- 保留 GLB 的多个 mesh primitive、材质、纹理和 UV
- 默认用原始三角网格同时显示和碰撞，便于检查几何质量
- 使用指定质量和整体包围盒计算质心、惯量；即使网格不封闭也能建立刚体
- 右键拖拽即可施加 picking force；即使 Viewer 处于暂停状态，拖拽期间也会自动推进物理

示例：

```powershell
python view.py D:\path\to\asset.glb --z 0.5
python view.py D:\path\to\asset.glb --z 0.5 --start-running
python view.py D:\path\to\asset.glb --glb-mass 2.0 --glb-collision convex
```

`mesh` 模式最适合核对原始碰撞几何；`convex` 模式为每个 GLB primitive 创建一个凸包碰撞体，通常更适合稳定、快速的动态仿真。复杂凹物体若需要高质量仿真，仍建议提供专门的凸分解碰撞 GLB，而不是把视觉网格直接当碰撞体。

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

- `Camera Navigation / Frame Asset (F)`：让相机重新对准当前资产并按尺寸取景
- `Camera Navigation / Fine Camera Mode`：持续使用精细相机灵敏度
- `Move Speed`：调整 WASD/QE 移动速度；每次加载资产时会先按资产尺寸自动设置
- `Wheel Sensitivity`：调整滚轮前后移动灵敏度
- `Double-sided Meshes`：双面显示 mesh，解决某些面从背面看不到的问题
- `Show Visual`：显示视觉 mesh
- `Show Inertia Boxes`：显示惯性盒

### 相机操作与微调

默认开启自动取景。每次加载或切换模型后，程序会计算资产所有有限几何的世界坐标包围盒、排除无限地面，然后保持设定的观察方向，把资产中心放到相机正前方并调整距离。因此模型不会再因为固定的绝对相机姿态而出现在相机背面，小物体也不会被强制按 1 米场景取景。仿真中按 `F` 会用物体当时的位置重新计算包围盒，不会跳回初始高度。

常用操作：

- `F` 或 `Frame Asset (F)`：随时重新聚焦当前资产
- `W/A/S/D`：前后左右移动；`Q/E`：上下移动
- 按住 `Shift`：临时把移动、滚轮、环绕和中键推拉灵敏度降到默认的 10%
- 勾选 `Fine Camera Mode`：持续使用低灵敏度，适合长时间检查小零件
- 鼠标滚轮：沿视线靠近/远离；`Ctrl + 滚轮`：调整视场角
- 中键拖动：环绕；`Shift + 中键拖动`：平移；`Ctrl + 中键拖动`：前后推拉

WASD/QE 的基础速度会按当前资产尺寸自动计算。也可以用 UI 滑条或 `--camera-speed` 指定固定值。若需要严格使用 `--camera-x/y/z`、`--pitch` 和 `--yaw` 的绝对姿态，可传入 `--no-auto-frame`。

## 物理模拟和拖拽

URDF 默认是静态查看模式。如果要用右键拖拽 URDF 的可运动 body 并计算牵引力，需要开启模拟：

```powershell
python view.py --simulate
```

GLB 为了支持质量测试和拖拽会自动初始化物理，不要求添加 `--simulate`。模型默认暂停，右键拖拽期间会自动步进；需要持续运行重力和碰撞时，可在 Viewer 中开始播放或使用 `--start-running`。

URDF 模拟默认暂停在资产的原始姿态，便于先检查模型。需要运行物理时可在 Viewer 中开始播放，或者使用下面的命令让它加载后立即运行：

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

### 选择物理求解器

默认求解器是 MuJoCo。可通过命令行或 Viewer 的 `Physics Solver` 下拉框切换；在 Viewer 中切换会重新加载当前资产并重置姿态。

```powershell
python view.py --category microwave --index 0 --simulate --solver xpbd
python view.py --category microwave --index 0 --simulate --solver semi-implicit
python view.py --category microwave --index 0 --simulate --solver featherstone
python view.py D:\path\to\asset.glb --solver vbd
python view.py D:\path\to\asset.glb --solver mujoco
```

可选项：

- `xpbd`：使用最大坐标约束，适合通用刚体和关节测试
- `semi-implicit`：半隐式最大坐标求解，速度快，但对时间步长和约束刚度更敏感
- `featherstone`：约化坐标 articulation 求解，适合 URDF 机器人和关节机构
- `vbd`：隐式 VBD/AVBD，当前 Newton 中仍属于实验性功能
- `mujoco`：默认值，MuJoCo 约化坐标求解；本程序使用 Newton contacts，以保持碰撞显示、拖拽和跨求解器比较一致

`--iterations` 用于 XPBD、VBD 和 MuJoCo；Semi-Implicit 与 Featherstone 不使用该参数。不同求解器支持的关节属性、摩擦、驱动和碰撞算法并不完全相同，因此结果不保证完全一致。

对比 fixed joint 折叠前后的仿真：

```powershell
python view.py --category microwave --index 0 --simulate --collapse-fixed-joints
python view.py --category microwave --index 0 --simulate --no-collapse-fixed-joints
```

折叠后，fixed joint 会被移除，其 shape、质量和惯量会合并到最近的保留 body；固定到 world 的根 body 会直接成为静态 world geometry。这样通常更快、更稳定，但 body/joint 数量、索引、contact 归属和逐 link 观测结果会改变。

对比固定根与自由根：

```powershell
python view.py --category microwave --index 0 --simulate --no-floating
python view.py --category microwave --index 0 --simulate --floating --z 0.5
```

`floating=True` 会为 URDF 根 body 创建一个具有 7 个坐标（位置和四元数）的 FREE joint，使整个资产能受重力、碰撞和拖拽影响。FREE joint 不会被 `collapse_fixed_joints` 折叠；该选项仍会处理 URDF 内部的 fixed joints。

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
- `--solver NAME`：选择 `xpbd`、`semi-implicit`、`featherstone`、`vbd` 或 `mujoco`，默认 `mujoco`
- `--iterations N`：XPBD、VBD 和 MuJoCo 的迭代次数，默认 10
- `--start-running`：加载后立即开始运行物理；GLB 可直接使用，URDF 需配合 `--simulate`
- `--scale VALUE`：导入模型时缩放
- `--glb-mass VALUE`：GLB 单刚体质量，默认 1 kg
- `--glb-collision mesh|convex`：GLB 使用精确三角网格或逐 primitive 凸包碰撞
- `--glb-max-hull-vertices N`：凸包最大顶点数，默认 64
- `--z VALUE`：导入模型时增加高度偏移，默认 5 m
- `--ground / --no-ground`：开启或关闭 `z=0` 地面，默认开启
- `--show-colliders`：显示碰撞 mesh
- `--self-collisions / --no-self-collisions`：启动时开启或关闭资产自碰撞，默认关闭
- `--collapse-fixed-joints / --no-collapse-fixed-joints`：开启或关闭 fixed joint 折叠，默认关闭
- `--floating / --no-floating`：让 URDF 根 body 自由运动或固定到 world，默认开启
- `--double-sided / --no-double-sided`：开关双面渲染，默认开启
- `--auto-frame / --no-auto-frame`：加载后按资产几何自动取景，默认开启
- `--camera-padding VALUE`：自动取景时四周的留白系数，默认 1.35
- `--camera-speed VALUE`：固定键盘移动速度（m/s）；省略时按资产尺寸自动计算
- `--camera-wheel-sensitivity VALUE`：滚轮前后移动灵敏度，默认 0.08
- `--camera-fine-scale VALUE`：Shift/精细模式的灵敏度倍率，默认 0.1
- `--camera-x/y/z`：关闭自动取景时使用的相机位置，默认 `(0.65, -0.75, 5.0)`
- `--pitch / --yaw`：初始观察方向，默认 `-25° / 40°`；自动取景也会沿这个方向看向资产
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

对于 URDF，需要满足两个条件：

1. 使用 `--simulate`
2. 在 Viewer 里右键拖住可运动 body

静态查看模式不会施加拖拽力。

GLB 会自动启用物理，只需在 Viewer 里用右键拖拽物体；左键仍用于相机/界面操作。

### 为什么某些模型看起来很奇怪？

Artiverse 的 URDF 主要用于 articulated object 交互和仿真查看，不一定是高保真的真实物理参数。部分模型的 COM、质量、惯性和视觉 mesh 可能只是简化值。
