# ToothForge3D

MCDG（Multi-Condition Dental Geometry）首版条件编码器与 TRELLIS.2 Flow 适配器。输入三视图、牙位、颈缘线、毫米尺寸和可选同颌牙点云，输出可供 TRELLIS.2 Sparse Structure Flow 与 Shape Flow 使用的条件特征。

## 当前范围

- 通过 SindrePy 的 `DINOv3Encoder.encode(..., return_tokens=True)` 提取冻结三视图特征；按颊侧、舌侧、咬合侧的固定顺序融合。
- 将上下颌、1～16 牙位、牙型和侧别编码为目标牙条件。
- 对颈缘线和毫米尺寸使用 Fourier 特征。
- 可选地把 SindrePy `PointTransformerV3` 作为同颌牙点云编码器，并压缩成局部几何 tokens。
- `TrellisFlowAdapter` 包装官方 TRELLIS.2 Flow 模型，在各个原始 Flow block 中传递条件 tokens，并把全局条件加入 AdaLN 的时间嵌入。

本仓库不复制 TRELLIS.2 主干。SindrePy 的 `Trellis2` 是推理封装；训练或修改条件路径时，应从官方 TRELLIS.2 训练代码加载 Flow 模型，再交给本仓库的适配器。适配器会产生 Flow 速度预测；数据准备、Flow Matching 训练循环、Shape VAE/O-Voxel 解码和最终 CAD 几何修正仍由项目训练流程负责。

## 环境

环境按 [sindre_ext](https://github.com/SindreYang/sindre_ext) 的 Windows 基线固定为 Python 3.12 和 PyTorch 2.7.1/CUDA 12.8。`uv sync` 会创建环境并安装 PTv3 所需的 `spconv`、`torch-scatter` 等 Windows 扩展；SindrePy 源码和 DINOv3/TRELLIS.2 模型权重需单独提供。

```powershell
uv sync
```

在 `SindrePy` 与本仓库同级的工作区中，可将本地库安装到环境：

```powershell
uv pip install --python .venv\Scripts\python.exe --no-deps --editable ..\SindrePy
```

## 输入约定

| 输入 | 形状 | 约定 |
| --- | --- | --- |
| `view_features` | `[B, 3, T, C]` | DINOv3 patch tokens；视图顺序为颊、舌、咬合 |
| `jaw` | `[B]` | `0` 上颌，`1` 下颌 |
| `target_position` | `[B]` | 目标牙位 `1..16` |
| `margin_line` | `[B, L, 3]` | 同一局部坐标系中的颈缘线，单位毫米，点按曲线顺序采样 |
| `size` | `[B, 3]` | 近远中宽、颊舌厚、生成高度，单位毫米 |
| `teeth_points` | `[B, M, N, 6]` | 可选同颌牙 XYZ + Normal |
| `teeth_positions` | `[B, M]` | 每颗上下文牙的 `1..16` 牙位 |
| `teeth_mask` | `[B, M]` | `True` 表示该位置有牙；允许零颗同颌牙 |

`view_features` 由冻结图像编码器预先计算。传入 DINOv3 patch tokens 可以避免在每次 Flow 迭代中重复运行图像模型。点云条件需要给 `MCDGConditionEncoder` 传入一个每点输出 `.feat` 的 PTv3 模块。

## 使用方式

```python
import numpy as np
import torch

from mcdg import MCDGConditionEncoder, TrellisFlowAdapter
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from sindre.ai.vision3d.pointcloud_utils.point_transformerV3 import PointTransformerV3

# 每个病例依次提供颊侧、舌侧、咬合视图。
with DINOv3Encoder() as dino:
    view_features = np.stack([
        np.stack([dino.encode(image, return_tokens=True) for image in case_views])
        for case_views in views
    ])
view_features = torch.as_tensor(view_features, dtype=torch.float32, device=device)

ptv3 = PointTransformerV3(in_channels=6, cls_mode=False).to(device)
condition_encoder = MCDGConditionEncoder(
    image_dim=view_features.shape[-1], point_encoder=ptv3
).to(device)
conditions = condition_encoder(
    view_features=view_features,
    jaw=jaw,
    target_position=target_position,
    margin_line=margin_line,
    size=size,
    teeth_points=teeth_points,
    teeth_positions=teeth_positions,
    teeth_mask=teeth_mask,
)

# 从官方 TRELLIS.2 训练配置加载两个 Flow 模型。
structure_flow = TrellisFlowAdapter(structure_flow, condition_encoder.hidden_dim).to(device)
shape_flow = TrellisFlowAdapter(shape_flow, condition_encoder.hidden_dim).to(device)
structure_velocity = structure_flow(noisy_structure, timestep, conditions)
shape_velocity = shape_flow(noisy_shape_latent, timestep, conditions)
```

不使用同颌牙时，省略 `teeth_points`、`teeth_positions` 和 `teeth_mask`。只有在提供点云时才需要安装 SindrePy PTv3 所需的 `spconv`、`torch-scatter` 等依赖。DINOv3、TRELLIS.2 CUDA 扩展和模型权重按各自上游项目安装。

## 版本边界

Flow 适配器复用官方模型的 `input_layer`、`blocks`、`out_layer` 和稀疏张量类型，需与加载的 TRELLIS.2 版本保持一致。TRELLIS.2 的官方仓库提供 Flow 训练代码；SindrePy 高层 `Trellis2` 接口面向推理，不暴露这些训练模块。[官方 TRELLIS.2](https://github.com/microsoft/TRELLIS.2)

当前代码是网络条件路径的研究实现，不代表已完成牙冠生成质量、毫米尺寸精度、颈缘线拟合精度或临床可用性的验证。CAD 后处理与可微/非可微损失需要结合实际数据集和网格流程单独接入。
