# view.py 使用文档

`view.py` 用 Newton Viewer 打开 URDF、USD 或 GLB 资产，并提供关节滑条、资产快速切换、自动相机取景、精细视角控制、位置显示、双面渲染和右键拖拽力显示。项目用于查看和检测资产是否适合作为机器人仿真模型，不会自动猜测缺失的质量、关节或碰撞属性。

项目安装、统一中文菜单和当前能力边界参见 [`README.md`](README.md)。远程服务器运行和
Web UI 的架构、第一阶段范围及后续演进建议参见
[`docs/远程物理试验运行与结果查看路线.md`](docs/远程物理试验运行与结果查看路线.md)。
桌面 Viewer 与自动试验是两条不同管线：Viewer 可按所选求解器使用本机 GPU，并使用
Newton contacts 支持交互；菜单中的摔落和固定 25° 坡度冒烟则固定为非正式 MuJoCo CPU、
原生 MuJoCo contacts，物理解算不使用 CUDA。Windows 冒烟录像使用系统 OpenGL，
可能使用本机图形 GPU；Linux 冒烟只允许经过验证的 Mesa 软件 OpenGL。图形渲染设备不改变冒烟的
非正式 CPU 物理边界。二者结果不能直接比较，坡度冒烟也不产生正式摩擦结论。

## 代码目录

`view.py` 仍是兼容启动入口。目录按职责整理为：

- `asset_viewer/assets.py`：资产发现、路径描述和 articulation metadata
- `asset_viewer/controls.py`：关节控制与 ImGui 交互
- `asset_viewer/camera.py`：资产包围盒、自动取景和精细相机控制
- `asset_viewer/solvers.py`：求解器选择与构建
- `asset_viewer/traction.py`：右键拖拽牵引力的计算和显示
- `asset_viewer/app.py`：Newton 模型构建、Viewer 运行时和命令行
- `record_collision_comparisons.py`：修复前后碰撞体的筛选、仿真和对比录像

默认读取目录：

```text
D:\Datasets\Artiverse\dataset_chunks\data
```

如果以后继续往这个目录里加入新类别，`view.py` 会递归列出所有支持的
`.urdf`、`.usd`、`.usda`、`.usdc`、`.usdz` 和 `.glb` 文件。混合目录中的格式
会全部保留，不做优先级过滤。

## 基本用法

日常使用可以执行 `uv run newton-test`，然后选择“打开桌面资产 Viewer”。下面的命令适合
需要精确指定参数的进阶使用；它们假定已经进入正确的 Python 环境。使用仓库锁定环境时，
在每条命令前添加 `uv run`，例如 `uv run python view.py --list`。

查看所有可打开的模型。输出使用文件类型标签和相对路径，不使用易变化的数字索引：

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

打开某一类目录中的第一个模型：

```powershell
python view.py --category microwave
```

打开某个具体目录、URDF、USD 或 GLB 文件：

```powershell
python view.py D:\Datasets\Artiverse\dataset_chunks\data\microwave
python view.py D:\path\to\model\urdf_w_collider\model.urdf
python view.py D:\path\to\simulation_asset.usda
python view.py D:\path\to\asset.glb
```

## 资产定位

Viewer 不再公开不稳定的数字资产索引。需要启动指定资产时，直接传入其目录或文件路径；
运行后也可以在 `Asset Tree` 中按真实目录结构选择文件。Previous/Next 仍可按目录树顺序
切换相邻资产。

## Viewer 侧边栏

### Assets

用于快速切换资产：

- `Import Asset...`：打开文件选择窗口，从任意路径选择 URDF、USD 或 GLB 文件并立即加载
- `Enable Self Collisions`：开启或关闭 URDF/USD articulation 内部各 link 之间的碰撞，切换后自动重新加载
- `Collapse Fixed Joints`：合并 URDF/USD 中由 fixed joint 连接的 body；默认关闭，切换后自动重新加载
- `Floating Base`：让 URDF 根 body 通过 FREE joint 自由运动；默认开启，切换后自动重新加载
- `USD Root Mode`：选择 `Authored`、`Floating` 或 `Fixed`；默认尊重 USD 自身定义
- `Physics Solver`：选择物理求解器；默认 MuJoCo，切换后自动重新加载并重置当前资产
- `Reload`：重新加载当前模型
- `Previous` / `Next`：切换上一个或下一个模型
- `Asset Tree`：从资产源根目录开始，按磁盘上的真实包含关系递归显示目录；目录后显示其中
  可加载资产总数，文件使用 `[URDF]`、`[GLB]`、`[USD]`、`[USDA]`、`[USDC]` 或
  `[USDZ]` 标签。只显示受支持文件及其祖先目录，当前文件所在分支会自动展开
- `Imported Assets`：在资产源根目录之外导入的文件按其绝对目录结构归入这个独立节点；
  多个文件会合并共享父目录，不会破坏原始数据集树

切换模型后，程序会按文件类型重新构建 URDF/USD 仿真模型或 GLB 单刚体及当前 UI。

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

### USD 仿真资产质量测试

USD 通过 Newton 的 `ModelBuilder.add_usd()` 导入，不会退化为单纯的视觉 mesh。程序会读取
USD Physics/Newton schema 中的刚体、关节、碰撞体、质量、惯量和相关物理属性，并使用
stage 的 `metersPerUnit` 和 up axis。支持 `.usd`、`.usda`、`.usdc` 和 `.usdz`。

```powershell
python view.py D:\path\to\asset.usda
python view.py D:\path\to\asset.usdc --show-colliders
python view.py D:\path\to\asset.usd --simulate --start-running
python view.py D:\path\to\asset.usdz --simulate --usd-root-mode floating
```

USD 根节点模式：

- `authored`：默认值，保留 USD 已声明的关节和根节点行为；没有父关节的刚体采用 Newton
  的 USD 格式默认行为。
- `floating`：强制没有父关节的根刚体通过 FREE joint 连接到 world。
- `fixed`：强制根刚体固定到 world。

如果 USD 只包含视觉几何而没有刚体，程序仍会显示 Newton 能导入的静态几何，但界面和
终端会提示该资产没有形成可动态操纵的刚体。程序不会自动生成质量、碰撞体或关节，因为
这些缺失本身就是仿真资产质量问题。

USD 需要 OpenUSD Python 绑定，即能够执行 `from pxr import Usd`。缺少依赖、USD
语法错误、外部引用丢失或不受支持的 physics schema 会显示为明确的导入错误。`--scale`
只适用于 URDF/GLB；USD 应通过 stage 的 `metersPerUnit` 正确声明单位。

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

URDF 和 USD 默认都是暂停的静态查看模式。如果要用右键拖拽可运动 body、运行重力或计算牵引力，需要开启模拟：

```powershell
python view.py --simulate
```

GLB 为了支持质量测试和拖拽会自动初始化物理，不要求添加 `--simulate`。模型默认暂停，右键拖拽期间会自动步进；需要持续运行重力和碰撞时，可在 Viewer 中开始播放或使用 `--start-running`。

URDF/USD 模拟默认暂停在资产的原始姿态，便于先检查模型。需要运行物理时可在 Viewer 中开始播放，或者使用下面的命令让它加载后立即运行：

```powershell
python view.py --simulate --start-running
```

开始播放后，程序遵循 URDF/USD 自身的质量、惯量、关节限制和动力学参数。如果资产没有提供关节阻尼、摩擦或驱动，部件在重力下转动可能反映的是资产数据质量问题。

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
python view.py --category microwave --simulate --solver xpbd
python view.py --category microwave --simulate --solver semi-implicit
python view.py --category microwave --simulate --solver featherstone
python view.py D:\path\to\asset.usda --simulate --solver mujoco
python view.py D:\path\to\asset.glb --solver vbd
python view.py D:\path\to\asset.glb --solver mujoco
```

可选项：

- `xpbd`：使用最大坐标约束，适合通用刚体和关节测试
- `semi-implicit`：半隐式最大坐标求解，速度快，但对时间步长和约束刚度更敏感
- `featherstone`：约化坐标 articulation 求解，适合 URDF/USD 机器人和关节机构
- `vbd`：隐式 VBD/AVBD，当前 Newton 中仍属于实验性功能
- `mujoco`：默认值，MuJoCo 约化坐标求解；本程序使用 Newton contacts，以保持碰撞显示、拖拽和跨求解器比较一致

`--iterations` 用于 XPBD、VBD 和 MuJoCo；Semi-Implicit 与 Featherstone 不使用该参数。不同求解器支持的关节属性、摩擦、驱动和碰撞算法并不完全相同，因此结果不保证完全一致。

对比 URDF/USD fixed joint 折叠前后的仿真：

```powershell
python view.py D:\path\to\microwave.urdf --simulate --collapse-fixed-joints
python view.py D:\path\to\microwave.urdf --simulate --no-collapse-fixed-joints
```

折叠后，fixed joint 会被移除，其 shape、质量和惯量会合并到最近的保留 body；固定到 world 的根 body 会直接成为静态 world geometry。这样通常更快、更稳定，但 body/joint 数量、索引、contact 归属和逐 link 观测结果会改变。

对比固定根与自由根：

```powershell
python view.py D:\path\to\microwave.urdf --simulate --no-floating
python view.py D:\path\to\microwave.urdf --simulate --floating --z 0.5
```

`floating=True` 会为 URDF 根 body 创建一个具有 7 个坐标（位置和四元数）的 FREE joint，使整个资产能受重力、碰撞和拖拽影响。FREE joint 不会被 `collapse_fixed_joints` 折叠；该选项仍会处理 URDF 内部的 fixed joints。

USD 使用独立的 `--usd-root-mode authored|floating|fixed`，不会受到 URDF
`--floating/--no-floating` 默认值的干扰。

## 碰撞体修复前后对比

`record_collision_comparisons.py` 属于资产质量检测工具。它读取原资产及同级的
`<模型> - fix`，可以先按碰撞数量或短时落体仿真筛选，再为指定资产生成修复前后的
确定性对比录像。它不修改资产。

筛选候选资产：

```powershell
python record_collision_comparisons.py --screen
python record_collision_comparisons.py --screen --counts-only --screen-limit 20
```

为一个或多个资产录像：

```powershell
python record_collision_comparisons.py `
  --asset-dir D:\path\to\category\source\MODEL_ID `
  --output-root D:\path\to\comparison_output
```

可重复使用 `--asset-dir`。数据集不在默认位置时使用 `--data-root`。修复程序本身已迁移到
同级独立项目 `Artiverse-Collision-Fixer`，当前项目只保留查看、检测和对比能力。

## 常用参数

```powershell
python view.py --category scissors
python view.py D:\path\to\microwave.urdf --simulate
python view.py --category lighter --show-colliders
python view.py --category piano --scale 0.5
python view.py --category microwave --initial-motion-state closed
python view.py --headless --frames 1
```

参数说明：

- `--category NAME`：只搜索某个类别，可重复使用
- `--list`：按文件类型标签和相对路径列出可加载资产，不添加数字编号
- `--simulate`：初始化物理模拟，默认暂停在原始姿态
- `--solver NAME`：选择 `xpbd`、`semi-implicit`、`featherstone`、`vbd` 或 `mujoco`，默认 `mujoco`
- `--iterations N`：XPBD、VBD 和 MuJoCo 的迭代次数，默认 10
- `--start-running`：加载后立即开始运行物理；GLB 可直接使用，URDF/USD 需配合 `--simulate`
- `--scale VALUE`：缩放 URDF/GLB；USD 使用 stage 声明的单位
- `--glb-mass VALUE`：GLB 单刚体质量，默认 1 kg
- `--glb-collision mesh|convex`：GLB 使用精确三角网格或逐 primitive 凸包碰撞
- `--glb-max-hull-vertices N`：凸包最大顶点数，默认 64
- `--z VALUE`：导入模型时增加高度偏移，默认 5 m
- `--ground / --no-ground`：开启或关闭 `z=0` 地面，默认开启
- `--show-colliders`：显示碰撞 mesh
- `--self-collisions / --no-self-collisions`：启动时开启或关闭资产自碰撞，默认关闭
- `--collapse-fixed-joints / --no-collapse-fixed-joints`：开启或关闭 fixed joint 折叠，默认关闭
- `--floating / --no-floating`：让 URDF 根 body 自由运动或固定到 world，默认开启
- `--usd-root-mode authored|floating|fixed`：USD 根节点行为，默认 `authored`
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

### 如何直接打开指定模型？

数字资产索引已经移除，因为目录内容或筛选条件变化后编号并不稳定。请直接传入模型目录
或具体文件：

```powershell
python view.py D:\Datasets\Artiverse\dataset_chunks\data\microwave\3dc200\7e_000
python view.py D:\path\to\model\urdf_w_collider\model.urdf
```

### 为什么有些面背面看不到？

开启双面显示：

```powershell
python view.py --double-sided
```

也可以在 UI 的 `Rendering Options` 中勾选 `Double-sided Meshes`。

### 为什么拖拽力一直是 0？

对于 URDF/USD，需要满足两个条件：

1. 使用 `--simulate`
2. 在 Viewer 里右键拖住可运动 body

静态查看模式不会施加拖拽力。

GLB 会自动启用物理，只需在 Viewer 里用右键拖拽物体；左键仍用于相机/界面操作。

### 为什么某些模型看起来很奇怪？

Artiverse 的 URDF/USD 主要用于 articulated object 交互和仿真查看，不一定是高保真的真实物理参数。部分模型的 COM、质量、惯性、材质和视觉 mesh 可能只是简化值。
