# Artiverse 官方查看器快照

这个目录明确分成两部分：

- `upstream/`：来自 Artiverse 官方 GitHub 仓库的查看器文件，保持原样。
- `run_view_model.py`：本项目的本地 Adapter，不属于 Artiverse 官方代码。

官方查看器不是 Newton/URDF 交互窗口。它读取 `*.segmented.glb` 和
articulation JSON，通过 BlenderProc 为每个 part 输出高亮、纹理以及左右视角视频，
适合与本项目的 Newton URDF 查看结果进行交叉检查。

## 安装

在要运行查看器的 Python 环境中执行：

```powershell
python -m pip install -r third_party/artiverse_official/requirements.txt
blenderproc pip install pygltflib meshcat open3d
```

`pygltftoolkit` 已按上游 `.gitmodules` 指向的官方仓库固定版本放入
`upstream/external/pygltftoolkit/`。依赖需要分别安装到普通 Python 环境和 Blender
自带的 Python 环境；第二条命令完成后者。BlenderProc 首次运行还会下载 Blender。
`static-ffmpeg` 为官方脚本中直接调用的 `ffmpeg` 提供可执行文件。

## 先检查

```powershell
python third_party/artiverse_official/run_view_model.py `
  --model-path D:\Datasets\Artiverse\dataset_chunks\data\lidded_bin\fpModel\MODEL_ID `
  --output-dir official_renders\MODEL_ID `
  --check
```

## 正式渲染

去掉 `--check` 即可。官方入口会对分割出的每个 pid 生成四组结果，耗时和输出量
会随 part 数量增加：

```powershell
python third_party/artiverse_official/run_view_model.py `
  --model-path D:\path\to\category\source\MODEL_ID `
  --output-dir official_renders\MODEL_ID
```

本地 Adapter 只负责兼容官方入口的两个已知假设：

1. 官方入口硬编码 `*.corrected.articulations.json`；预发布数据只有
   `*.articulations.json` 时，Adapter 会在临时目录用官方期望的文件名运行，
   不修改原始资产。
2. 官方入口依赖仓库根目录和 POSIX 风格路径；Adapter 会设置工作目录、依赖路径和
   临时输出目录，再把结果复制到指定位置。

在当前 Windows + Blender 4.2.1 环境中，官方 HDRI 文件会出现 EXR 读取警告；
最小烟雾测试仍成功导入几何并生成了 128×128 VP9 WebM。为保持上游快照可核验，
这里没有改写该 HDRI。

上游版本、文件范围和校验值见 [PROVENANCE.md](PROVENANCE.md)。
