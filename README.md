# ToothForge3D

基于 SindrePy 和 O-Voxel / FDG 的**专用开放牙冠生成**研究工程。目标是给定真实牙颌、FDI、颈缘线，自动生成 CAD 可编辑的开放牙冠；**MD、BL、H 为可选设计约束**，用户不知道时由模型先估计，生成后再测量真实尺寸。

> 当前代码已包含轻量网络、成对数据增强、训练与推理接口；**尚未在真实 4090、O-Voxel CUDA 与独立病例上完成端到端实测**。不要将代码实现等同于已验证的牙冠生成精度。

## 目录职责

| 文件 | 职责 |
| --- | --- |
| `model.py` | **唯一网络结构文件**：SindrePy MCDG 条件编码、轻量 coarse-to-fine FDG 预测、MD/BL/H 物理尺度条件 |
| `train.py` | **唯一训练文件**：train/val、tqdm、BF16、Profiler、分组验证、最佳/最近权重 |
| `infer.py` | **唯一推理入口**：从病例 JSON 或原始 Sindre LMDB 的 `path` 读取、预测开放冠、世界坐标保存 Mesh |
| `datasets.py` | 仅 FDG v2 的 `FDGDataset`、训练查询采样与在线邻牙增强（不再提供 Occupancy 构建接口） |
| `utils.py` | 原始病例直转 O-Voxel FDG、固定颈缘线的目标冠尺寸增强、独立 LMDB 写入和边界小幅修正 |
| `losses.py` | **唯一 Loss 文件**：结构 BCE+Dice、offset SmoothL1、intersection BCE、MD/BL/H 实际尺寸与边界到颈缘线双向软约束 |
| `process.py` | 真实病例字段、FDI、局部坐标与无设计冠的推理条件构建 |
| `views.py` | 真实牙颌颊/舌/咬合三视图与 SindrePy DINOv3 编码；不输入设计冠 Mesh |

不采用 argparse 或命令行参数解析；所有入口均接受显式 Python 参数。公共接口的参数、返回值和异常使用中文 Google-style docstring。

## 条件与目标形态

```text
原始牙颌扫描 + opposite_mesh（仅三视图背景）
          │
          ├── FDI 牙位 + 颈缘线 + 可选 size=[MD, BL, H]（mm）
          ├── 缓存的颊/舌/咬合 DINOv3 tokens
          └── 同颌 PTv3 点云（可缺牙、裁剪、局部上下文）
                          ↓
             模型轻量 MD/BL/H 尺寸先验预测
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

**长宽高是可选的三个独立参数**。用户不填时，MCDG 先用 DINOv3 三视图、FDI、颈缘线和同颌 PTv3 特征估计 MD/BL/H；用户填写时使用明确的尺寸约束。最终尺寸仍由生成的 Mesh 实测。估计值经过 Fourier 尺寸 Embedding，并用于粗结构和细体素相对位置，不把三个尺寸简化为等比例缩放。

## 为什么尺寸增强必须放在预处理？

单纯给 `size` 乘随机数、但不修改真实冠标签，会让网络收到自相矛盾的监督，最终忽略尺寸。

`utils.CrownFDG.augment_crown_size(...)` 在**真实冠生成 FDG 标签之前**，对牙冠冠身做近远中、颊舌和高度三个方向相互独立的柔性形变。边界顶点**保持固定**，其余顶点随离颈缘线的距离逐步增加形变。每个增强版本都同时生成：

- 新的 `target_vertices` 与 FDG `coords/dual_offsets/intersected` 监督；
- 与实际增强几何对应的新 `size=[MD,BL,H]`；
- **不变的** FDI、颈缘线、牙颌扫描、DINOv3 输入特征。

默认 `augment_variants=4`（1 原始 + 3 对应的尺寸增强），三个轴的变化幅度各为 ±10%。这是**监督成对增强**，不涉及训练期重新计算 FDG。

**邻牙与咬合安全门槛**：默认 `collision_check=True`，尺寸增强前使用真实病例的同颌牙位标签与已配准对颌扫描构建局部环境 Mesh。复用 SindrePy `AdjacentCollisionDetector(solid=False)` 和 MeshLib 检查每个增强后的牙冠是否新增邻牙/对颌三角面相交；每份增强最多尝试12次，冲突或变形产生自交时拒绝并重新采样。原始设计冠如果已存在相交/接触，则保留原始样本但**停止为该病例生成尺寸增强**，因为开放 Mesh 无法测量穿透深度，也就无法证明形变没有加重穿透。无对颌扫描或原始病例环境缺失则报错；仅接受原始病例 LMDB，不再提供 Occupancy/SDF 迁移接口。测试/实验可显式设 `collision_check=False`，**但未经检查的样本不能视为几何合格**。

`build_database` 返回的报告包含 `rejected_collision`、`skipped_augmentation` 和 `warnings`，因此碰撞筛选后每病例实际训练样本可能少于4份。该检测**只确认三角面是否相交/接触，不计算实体穿透深度、咬合力或真实动态咬合**；还需检查对颌是否在同一毫米坐标系完成配准。

在线 `FDGDataset` 每次读取均重新采样同颌邻牙、以小概率删除整颗邻牙、对同颌 XYZ 加 0.02mm 轻噪声，以及以小概率遮蔽一张缓存视图；持久 DataLoader worker 的随机数状态持续前进。**不会在线单独抖动 `size` 或颈缘线**。需要增加尺寸增强版本时，重新预处理原始冠并生成新 FDG 数据库。

## 一次性数据预处理

原始病例应具有 `current_mesh`、`opposite_mesh`、`design_mesh`、FDI、真实颈缘线和**仅由输入扫描/预备体确定的 `local_frame`**。允许原始开放冠；已封底的冠尝试用 SindrePy MeshLib 沿颈缘线裁剪后再转换。对冠体异常、边界太远、FDG 重复/越界坐标直接报错。**若没有外部 local_frame，原始流程可能从设计冠推断牙位方向；新版 FDG 预处理会拒绝这种来源，防止训练泄漏目标形态。**

```python
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from utils import CrownFDG, FDGConfig

converter = CrownFDG(FDGConfig(resolution=128))

with DINOv3Encoder() as dino:
    report = converter.build_database(
        "data/raw_cases.db",
        "data/toothforge_fdg_v2.db",
        image_encoder=dino,
        augment_variants=4,
        augment_scale=0.10,
        collision_check=True,  # 默认校验邻接牙/对颌表面相交
        max_cases=5,  # 首次先检查截图，确认后改为 None
    )

print(report)
```

若需要查看边界和颈缘线可提前设置 `import utils; utils.DEBUG = True`，生成 Vedo 离屏截图。此过程可慢一点，**训练前一次性完成**。O-Voxel 原生 FDG 接口由上游实现，不在本项目重新写 CUDA 网格提取。

新库使用 `schema_version=toothforge3d_fdg_v2` 和 `coordinate_frame=margin_only_v2`，保存 `source_groups` 元数据。同一个原始病例的尺寸增强副本全部位于同一训练集或验证集，避免验证泄漏。

**真实长宽高来源要明确**：训练集可以使用已完成设计冠的 MD/BL/H 作为**监督标签**，但不能把它们当作所有样本的必备模型输入。训练默认 65% 的病例隐藏输入尺寸，独立的尺寸回归头始终接受目标尺寸监督；推理默认不需要用户填写，只有明确指定尺寸时才使用用户条件。

## 训练

```python
from train import train, FDGTrainingConfig
from model import LightFDGConfig
from losses import FDGLossConfig

result = train(
    "data/toothforge_fdg_v2.db",
    config=FDGTrainingConfig(
        epochs=100,
        batch_size=2,
        margin_every=16,    # 调用 O-Voxel CUDA 边界 Loss 的间隔
        size_dropout=0.65,  # 训练时65%的病例隐藏目标尺寸输入
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

损失只放在 `losses.py`，并在 FP32 计算：`L = 1.0 L_coarse + 1.0 L_fine + 0.5 L_offset + 0.5 L_intersection + 0.2 L_margin + 0.2 L_size + 0.4 L_size_prior`。其中 `L_size` 对从 O-Voxel 解码的真实 Mesh 测量 MD/BL/H，沿颈缘线局部坐标直接监督输出几何，而不仅是尺寸 Embedding。类别不平衡通过有限正例权重和 Dice 稳定处理；Margin Loss 允许约 0.15mm 柔性容差。**不使用视图 Loss**，三视图只在输入预处理时编码。

训练 tqdm 显示 coarse/fine/offset/召回率及间隔性实际尺寸误差（mm）。训练采用 AdamW、BF16、梯度累积/裁剪、SindrePy best/last checkpoint 和可选 torch.profiler。CPU 数学测试可以使用 `margin_every=0`，但完整 FDG 网格训练与推理依赖官方 O-Voxel CUDA 环境。

## 推理

`infer.py` 接收**真实牙颌病例路径**和最佳权重，病例不能依赖目标设计冠 Mesh。要求输入：

- `target_fdi`：修复目标牙位；
- `margin_line`：输入扫描坐标下的真实颈缘线（毫米）；
- `size_mm=(MD,BL,H)`：**可选**，用户指定的冠体长宽高（毫米）；不指定则由模型自动估计；
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
    # size_mm=(10.5, 9.1, 7.2),  # 可选：仅用户有明确目标时指定
    local_frame=np.asarray(local_axes, dtype=np.float32),
    output_path="output/crown_36.stl",
    refine_margin=True,
    show=True,            # Vedo 交互窗口
    show_context=True,    # 半透明同颌扫描
    show_opposite=True,   # 半透明对颌扫描
)

print(result["vertices"].shape, result["faces"].shape)
print("模型预估尺寸 (mm):", result["estimated_size_mm"])
print("生成冠实测尺寸 (mm):", result["generated_size_mm"])
```

生成 Mesh 默认校验单一开放边界环、连通性、自交、颈缘误差和邻牙/对颌碰撞。未通过则明确报错；只有调试时才显式传入 `validate_geometry=False`，结果将标记 `geometry_quality.verified=False`。通过验证的 Mesh 会从局部 FDG 坐标还原到**原始扫描世界毫米坐标**。`show=True` 将以 Vedo 打开交互窗口：蓝色生成冠、半透明灰色同颌与对颌、红色闭合颈缘线，支持旋转缩放；窗口关闭后函数返回。也可以直接调用 `infer.show_mesh(vertices, faces, case_data=...)`。如预测空体素、CUDA 解码失败或几何边界不符合约束，将明确报错；不会伪造成功的牙冠。

## 可选 MD/BL/H：训练和推理如何一致？

用户通常并不知道牙冠最终长度、宽度和高度。因此，尺寸是**模型可以估计、用户可以覆盖**的控制条件，而不是完成推理的前置条件。

1. **没有尺寸时**：仅根据 FDI、颈缘线、同颌邻牙和三视图的条件编码预测 `estimated_size_mm`，再以它作为形状条件生成冠体。
2. **有尺寸时**：使用 `size_mm=(MD,BL,H)` 覆盖形状条件；尺寸预测头仍可输出参考估计，但不会覆盖用户明确输入。
3. **生成完成后**：从 FDG Mesh 的局部 MD/BL 全宽与颈缘原点至冠顶高度直接测量 `generated_size_mm`。它才是最终实际尺寸，不等于先验估计或用户输入。
4. **训练策略**：始终保存设计冠真实尺寸作为监督标签；通过 `size_dropout=0.65` 随机遮住长宽高输入，独立 `size_prior` SmoothL1 在无已知尺寸时学习恢复尺寸。验证集默认隐藏尺寸，检查自动生成能力。
5. **坐标空间**：v2 FDG 统一用真实颈缘线确定空间尺度；模型预估尺寸与用户条件不会改变 FDG 网格本身。这解决了旧版训练依赖 GT 尺寸确定网格、无尺寸推理时网格不一致的问题。

**升级注意**：`toothforge3d_fdg_v1 / margin_size_v1` 训练库与权重不得直接作为 v2 使用，必须重新预处理、重新训练并验证原生 O-Voxel CUDA 解码。自动估计只提供初值，最终邻接/咬合质量仍需检查；当前不宣称已实现临床级自动定尺寸。

## 验证与部署限制

```powershell
uv lock
uv sync
uv run pytest -q
```

SindrePy 私有依赖、Vedo、O-Voxel 的 CPU/CUDA 原生扩展和 DINOv3 权重需要按照其各自项目方式在开发机安装。GPU 性能优化先使用 Profiler 识别瓶颈，必要时再编译专用 CUDA 算子，不为不明确的瓶颈重复造轮子。

当前代码已增加结构、尺寸敏感性、颈缘固定增强、同病例分组验证、无目标 Mesh 推理、世界坐标回写等测试；**尚未在真实病例、RTX 4090 和 O-Voxel CUDA 环境中执行完整端到端回归**。
