# ToothForge3D

基于 SindrePy 和 O-Voxel / FDG 的**专用开放牙冠生成**研究工程。目标是给定真实牙颌、FDI、颈缘线及用户指定的近远中宽/颊舌厚/高度，生成可在 CAD 中进一步编辑的开放牙冠 Mesh。

> 当前代码已包含轻量网络、成对数据增强、训练与推理接口；**尚未在真实 4090、O-Voxel CUDA 与独立病例上完成端到端实测**。不要将代码实现等同于已验证的牙冠生成精度。

## 目录职责

| 文件 | 职责 |
| --- | --- |
| `model.py` | **唯一网络结构文件**：SindrePy MCDG 条件编码、轻量 coarse-to-fine FDG 预测、MD/BL/H 物理尺度条件 |
| `train.py` | **唯一训练文件**：train/val、tqdm、BF16、Profiler、分组验证、最佳/最近权重 |
| `infer.py` | **唯一推理入口**：从病例 JSON 或原始 Sindre LMDB 的 `path` 读取、预测开放冠、世界坐标保存 Mesh |
| `datasets.py` | Sindre LMDB 数据集：`FDGDataset` 在线邻牙增强；保留旧 Occupancy 数据读取/预处理作数据迁移参考 |
| `utils.py` | O-Voxel FDG 转换、固定颈缘线的目标冠尺寸增强、独立 LMDB 写入和边界小幅修正 |
| `losses.py` | **唯一 Loss 文件**：结构 BCE+Dice、offset SmoothL1、intersection BCE、MD/BL/H 实际尺寸与边界到颈缘线双向软约束 |
| `process.py` | 真实病例字段、FDI、局部坐标与无设计冠的推理条件构建 |
| `views.py` | 真实牙颌颊/舌/咬合三视图与 SindrePy DINOv3 编码；不输入设计冠 Mesh |

不采用 argparse 或命令行参数解析；所有入口均接受显式 Python 参数。公共接口的参数、返回值和异常使用中文 Google-style docstring。

## 条件与目标形态

```text
原始牙颌扫描 + opposite_mesh（仅三视图背景）
          │
          ├── FDI 牙位 + 颈缘线 + 用户目标 size=[MD, BL, H]（mm）
          ├── 缓存的颊/舌/咬合 DINOv3 tokens
          └── 同颌 PTv3 点云（可缺牙、裁剪、局部上下文）
                          ↓
             model.MCDGConditionEncoder
                          ↓
             32³ 轻量粗结构预测（含物理长宽高特征）
                          ↓
             细体素存在 + dual_offsets + intersected
                          ↓
             官方 O-Voxel FDG 网格提取
                          ↓
                 开放牙冠 Mesh
                          ↓
            软颈缘贴合 + 有限后处理
```

**长宽高始终保留为三个独立参数**。在 `MCDGConditionEncoder` 中做 Fourier 尺寸 Embedding；在 FDG 粗结构网格与细体素查询头中，将空间坐标分别除以 MD、BL、H，使三个尺寸明确决定模型感知到的物理位置。不得把这三个尺寸简化成一个等比例缩放值。

## 为什么尺寸增强必须放在预处理？

单纯给 `size` 乘随机数、但不修改真实冠标签，会让网络收到自相矛盾的监督，最终忽略尺寸。

`utils.CrownFDG.augment_crown_size(...)` 在**真实冠生成 FDG 标签之前**，对牙冠冠身做近远中、颊舌和高度三个方向相互独立的柔性形变。边界顶点**保持固定**，其余顶点随离颈缘线的距离逐步增加形变。每个增强版本都同时生成：

- 新的 `target_vertices` 与 FDG `coords/dual_offsets/intersected` 监督；
- 与实际增强几何对应的新 `size=[MD,BL,H]`；
- **不变的** FDI、颈缘线、牙颌扫描、DINOv3 输入特征。

默认 `augment_variants=4`（1 原始 + 3 对应的尺寸增强），三个轴的变化幅度各为 ±10%。这是**监督成对增强**，不涉及训练期重新计算 FDG。

在线 `FDGDataset` 仅随机裁剪同颌邻牙、以小概率删除整颗邻牙、对同颌 XYZ 加 0.02mm 轻噪声，以及以小概率遮蔽一张缓存视图。**不会在线单独抖动 `size` 或颈缘线**。需要增加尺寸增强版本时，重新预处理原始冠并生成新 FDG 数据库。

## 一次性数据预处理

原始病例应具有 `current_mesh`、`opposite_mesh`、`design_mesh/design`、FDI、真实颈缘线和**仅由输入扫描/预备体确定的 `local_frame`**。允许原始开放冠；已封底的冠尝试用 SindrePy MeshLib 沿颈缘线裁剪后再转换。对冠体异常、边界太远、FDG 重复/越界坐标直接报错。**若没有外部 local_frame，原始流程可能从设计冠推断牙位方向；新版 FDG 预处理会拒绝这种来源，防止训练泄漏目标形态。**

```python
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from utils import CrownFDG, FDGConfig

converter = CrownFDG(FDGConfig(resolution=128))

with DINOv3Encoder() as dino:
    report = converter.build_database(
        "data/raw_cases.db",
        "data/toothforge_fdg_v1.db",
        image_encoder=dino,
        augment_variants=4,
        augment_scale=0.10,
        max_cases=5,  # 首次先检查截图，确认后改为 None
    )

print(report)
```

若需要查看边界和颈缘线可提前设置 `import utils; utils.DEBUG = True`，生成 Vedo 离屏截图。此过程可慢一点，**训练前一次性完成**。O-Voxel 原生 FDG 接口由上游实现，不在本项目重新写 CUDA 网格提取。

新库使用 `schema_version=toothforge3d_fdg_v1` 和 `coordinate_frame=margin_size_v1`，保存 `source_groups` 元数据。同一个原始病例的尺寸增强副本全部位于同一训练集或验证集，避免验证泄漏。

**真实长宽高来源要明确**：训练集可使用设计冠实测尺寸作为学习目标标签，或读取订单指定尺寸。但**推理和正式评估**必须从订单/预备体/用户输入给出尺寸，不能读取未知目标冠的真实包围盒，否则属于信息泄漏。

## 训练

```python
from train import train, FDGTrainingConfig
from model import LightFDGConfig
from losses import FDGLossConfig

result = train(
    "data/toothforge_fdg_v1.db",
    config=FDGTrainingConfig(
        epochs=100,
        batch_size=2,
        margin_every=16,    # 调用 O-Voxel CUDA 边界 Loss 的间隔
        profile_steps=20,   # 分析性能后改为 0
    ),
    model_config=LightFDGConfig(
        resolution=128,
        coarse_resolution=32,
        hidden_dim=128,
    ),
    loss_config=FDGLossConfig(
        coarse_weight=1.0,
        fine_weight=1.0,
        offset_weight=0.5,
        intersection_weight=0.5,
        margin_weight=0.2,
    ),
)
print(result)
```

损失只放在 `losses.py`，并在 FP32 计算：`L = 1.0 L_coarse + 1.0 L_fine + 0.5 L_offset + 0.5 L_intersection + 0.2 L_margin + 0.2 L_size`。其中 `L_size` 对从 O-Voxel 解码的真实 Mesh 测量 MD/BL/H，沿颈缘线局部坐标直接监督输出几何，而不仅是尺寸 Embedding。类别不平衡通过有限正例权重和 Dice 稳定处理；Margin Loss 允许约 0.15mm 柔性容差。**不使用视图 Loss**，三视图只在输入预处理时编码。

训练 tqdm 显示 coarse/fine/offset/召回率及间隔性实际尺寸误差（mm）。训练采用 AdamW、BF16、梯度累积/裁剪、SindrePy best/last checkpoint 和可选 torch.profiler。CPU 数学测试可以使用 `margin_every=0`，但完整 FDG 网格训练与推理依赖官方 O-Voxel CUDA 环境。

## 推理

`infer.py` 接收**真实牙颌病例路径**和最佳权重，病例不能依赖目标设计冠 Mesh。要求输入：

- `target_fdi`：修复目标牙位；
- `margin_line`：输入扫描坐标下的真实颈缘线（毫米）；
- `size_mm=(MD,BL,H)`：用户指定的冠体长宽高（毫米）；
- `local_frame`：近远中、颊舌、冠向构成的 3×3 右手正交矩阵；
- `checkpoint_path`：训练生成的 best.pt 或 last.pt。

必要条件既可保存在病例 JSON 中，也可通过函数传参覆盖。病例没有缓存的三视图特征时，由 SindrePy DINOv3 在推理准备阶段计算。**真正推理不读取目标冠坐标或 FDG 监督**。

```python
import numpy as np
from infer import infer

result = infer(
    path="data/case.json",
    checkpoint_path="checkpoints/toothforge_fdg/best.pt",
    target_fdi=36,
    margin_line=np.asarray(margin_points_world, dtype=np.float32),
    size_mm=(10.5, 9.1, 7.2),  # MD / BL / H，毫米
    local_frame=np.asarray(local_axes, dtype=np.float32),
    output_path="output/crown_36.stl",
    refine_margin=True,
)

print(result["vertices"].shape, result["faces"].shape)
```

输出 Mesh 会从局部 FDG 坐标还原到**原始扫描世界毫米坐标**。如预测空体素、CUDA 解码失败或几何边界不符合约束，将明确报错；不会伪造成功的牙冠。

## 验证与部署限制

```powershell
uv lock
uv sync
uv run pytest -q
```

SindrePy 私有依赖、Vedo、O-Voxel 的 CPU/CUDA 原生扩展和 DINOv3 权重需要按照其各自项目方式在开发机安装。GPU 性能优化先使用 Profiler 识别瓶颈，必要时再编译专用 CUDA 算子，不为不明确的瓶颈重复造轮子。

当前代码已增加结构、尺寸敏感性、颈缘固定增强、同病例分组验证、无目标 Mesh 推理、世界坐标回写等测试；**尚未在真实病例、RTX 4090 和 O-Voxel CUDA 环境中执行完整端到端回归**。
