# ToothForge3D

**MCDG — Multi-Condition Dental Geometry**

以 SindrePy 为基础的全冠三维形态生成研究项目。当前代码包含真实病例的同颌牙弓上下文处理、三视图/牙位/颈缘线/尺寸条件编码，以及 TRELLIS.2 Sparse Structure / Shape Flow 的条件适配器。

**当前并非完整的牙冠生成产品**：本仓库尚未提供训练好的权重、Flow Matching 训练循环、Shape VAE 网格解码整合、颈缘线硬约束与临床 CAD 验收；这些不能从编码器的输出自动推断为已实现。

## 1. 输入契约

| 输入 | 类型 / 形状 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| `view_features` | `[B,3,T,C]` | 是 | 使用真实设计冠三视图（重建模式）或独立参考牙三视图（设计模式）经 SindrePy DINOv3 提取的 tokens，顺序颊 / 舌 / 咬合 |
| `jaw` | `[B]` | 是 | 0 上颌，1 下颌 |
| `target_position` | `[B]` | 是 | SindrePy FDI→1~16 索引 |
| `margin_line` | `[B,128,3]` | 是 | 真实闭合颈缘曲线，毫米 |
| `size` | `[B,3]` | 是 | 近远中宽、颊舌厚、高，毫米 |
| `jaw_points` | `[B,N,6]` | 否 | 同颌牙弓 XYZ + Normal；不输入对颌 |
| `jaw_labels` | `[B,N]` | 有牙弓时必须 | 每点 0（背景/未知）或 1~16 牙位 |
| `jaw_present` | `[B]` | 有牙弓时可选 | 显式标识整条牙弓条件是否存在；允许部分样本无牙弓 |

**牙位规则要严格统一**：预处理直接读取设计记录中的 `fdi` 自动推导上下颌（11~28 为上颌；31~48 为下颌且需满足标准象限），对颌类别等于 `1-jaw`，但 **不输入对颌 Mesh**。SindrePy 的 `convert_fdi2idx` 将 11~18→1~8、21~28→9~16，31~38→1~8、41~48→9~16。示例：36→6、46→14。输入中没有额外的“近中/远中角色”；牙位与空间坐标足以提供上下文。

所有几何应统一到同一局部坐标系。牙弓没有任何可用同颌牙时，令 `jaw_present=False`；绝不生成伪邻牙。就位方向不是网络必选输入；本阶段不参与对颌咬合建模。

## 2. 结构与复用边界

```text
三视图 DINOv3 tokens ──> 共享投影 + 各视图重采样 + 跨视图 Attention ─┐
jaw + target_position ──> jaw / tooth type / anatomical side Embeddings ──┤
margin_line ────────────> Fourier + Curve Attention ───────────────────────┤
size (mm) ──────────────> Fourier + MLP ───────────────────────────────────┤
同颌点云+逐点牙位 ──────> SindrePy PTv3 + tooth embeddings + Resampler ────┤
                                                                       ↓
                                                条件 Tokens + FDI 强全局条件
                                                                       ↓
                                        TRELLIS.2 Structure Flow / Shape Flow
                                                                       ↓
                                       后续接 Shape VAE + CAD 几何硬约束
```

图像编码复用 `sindre.ai.vision2d.dinov3_utils.DINOv3Encoder`；整颌几何复用 `sindre.ai.vision3d.pointcloud_utils.point_transformerV3.PointTransformerV3`；Attention / Fourier / MLP 均复用 SindrePy。没有复制 TRELLIS.2 预训练主干。

- 三视图单独保留局部 token，避免把三视图直接平均成单个向量。
- 目标牙位进入 token 流，也以残差强条件进入全局向量。
- 整颌只运行一次 PTv3；逐点标签传递牙位和相对牙弓位置，不强制分牙成固定两颗邻牙。
- `TrellisFlowAdapter` 依赖匹配版本的官方 `trellis2.models.sparse_structure_flow` 与 `structured_latent_flow`，不能直接以 SindrePy 推理 `Trellis2.generate` 代替训练。

## 3. 真实冠病例预处理

`process.py` 支持原始病例预览所示的 `current_mesh/design_mesh` 结构，也接受 `design` 键值对（可嵌套 `mesh`），可不提供 `is_upper`。**注意：仓库现有 `datasets/*.json` 是仅有 vertices/faces 的独立牙冠 Mesh，而不是完整的病例记录。** 两种格式分别处理，不从 Mesh 文件猜测 FDI：

```python
{
    "is_upper": False,        # 可省略：由 design_mesh.fdi 推导
    "current_mesh": {
        "vertices": ...,
        "faces": ...,
        "vertex_labels": ...,
        "fdi": [0, 32, 41, 2031],  # 标签字典索引或真实 FDI；特殊值不当牙位
    },
    "design_mesh": {
        "vertices": ...,
        "faces": ...,
        "fdi": "31",
    },
    # margin_line 可从设计冠的唯一开放边界提取
    "margin_line": ...,
    # 不提供 size_mm 时，在局部坐标系中测量生成尺寸
    "size_mm": [7.2, 6.0, 8.3],
    # 不提供 local_frame 时，从设计冠与同颌几何自动估计
    "local_frame": ...,
    # 可直接提供 DINOv3 特征，也可按下方示例自动渲染/编码
    "view_features": ...,
}
```

`current_mesh` 的目标冠、特殊预备体编号（例如 2031）和对颌不参与“同颌牙齿”条件。标签表索引与真实 FDI 两种编码均可辨别。使用真实设计冠 `design_mesh` 作为监督目标。

```python
from process import DentalCase, get_reference_mesh, process, collate_conditions

# 现有 0.json 只有独立牙冠网格，用于几何重建或作为参考 Mesh。
mesh = get_reference_mesh("datasets/0.json")
print(mesh["vertices"].shape, mesh["faces"].shape)

# 完整病例需有 current_mesh、design_mesh(或 design)，无需填写 is_upper。
case = DentalCase(raw_case)
first = case.get_condition(crop_mode="full", num_points=4096)
second = case.get_condition(crop_mode="random", num_points=4096)
print(first["source_fdi"], first["target_position"], first["opposite_jaw"])
print(second["crop_mode"], second["jaw_points"].shape)

# 从设计冠渲染真正的三张标准正交视图，[3,H,W,3] uint8 RGB。
# 用于多视图重建训练。无 VTK 屏幕外渲染支持时会报错。
preview = case.get_condition(crop_mode="full", render_views=True, image_size=256)
images = preview["view_images"]
print(images.shape, preview["frame_source"], preview["view_anatomical_verified"])

# 用 SindrePy 提取三视图特征（建议预计算/缓存，避免每轮训练重复渲染）。
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder

with DINOv3Encoder() as dino:
    first = case.get_condition(crop_mode="full", image_encoder=dino)
    second = case.get_condition(crop_mode="random", image_encoder=dino)

batch = collate_conditions([first, second], num_points=4096)
```

随机增强在同颌真实点云上进行：完整牙弓、不同大小的目标位置引导空间裁剪、随机按牙位丢弃、以及空牙弓。训练集/验证集必须先按病例或订单分组，再展开随机增强，避免同一设计冠泄漏到两个数据集。所有结果保持同一病例坐标系，目标设计 Mesh 不混入上下文。随机种子影响裁剪/丢牙；没有人工构造不存在的邻牙。

**尚需补齐的数据**：

- **三视图**：提供 `render_views=True` 或 `image_encoder=DINOv3Encoder()` 后，自动用 `CrownViews` 将设计冠投影成颊侧、舌侧、咬合三个固定正交方向；其三视图只用于多视图重建。新冠独立设计需要把参考图替换为非目标设计冠的独立来源。
- **颈缘线**：优先读取真实标注。仅当设计冠具有唯一闭合开放边界时可自动重采样。封闭冠或多边界环不能猜测。
- **生成尺寸**：优先使用 `size_mm`；否则从设计冠局部坐标系沿 MD / BL / 冠向测量尺寸。该尺寸是基于估计坐标轴的几何标注，而非临床验证的解剖测量。
- **相机和坐标**：使用 SindrePy 的单牙长轴估计结合可用的同颌牙位推导局部坐标系，固定相机尺度、背景和方向。孤立牙仅能可靠得到冠向与两条相对侧视角，无法单凭 PCA 识别真正的颊、舌方向；此时 `frame_verified=False`，应人工检查或提供 `local_frame`。
- **对颌信息**：FDI 可推导对颌是上颌还是下颌，但不意味着模型需要输入 `opposite_mesh`。

未提供或生成 `view_features` 时，`ready_for_training=False`；仅获得三视图 RGB 而未经过 DINOv3 编码时，也还未完成网络输入准备。

## 4. 条件编码示例

```python
import torch
from mcdg import MCDGConditionEncoder, TrellisFlowAdapter
from sindre.ai.vision3d.pointcloud_utils.point_transformerV3 import PointTransformerV3

ptv3 = PointTransformerV3(
    in_channels=6, cls_mode=False, enable_flash=False,
).to("cuda")

encoder = MCDGConditionEncoder(
    image_dim=768, point_dim=64, point_encoder=ptv3,
).to("cuda")

# batch 由 collate_conditions 提供，需将所有 tensor 转到 CUDA
conditions = encoder(**{k: value.cuda() for k, value in batch.items()})

# 接入官方 TRELLIS.2 的预训练 Flow（必须使用与当前接口匹配的版本）
structure_adapter = TrellisFlowAdapter(structure_flow, encoder.hidden_dim).cuda()
shape_adapter = TrellisFlowAdapter(shape_flow, encoder.hidden_dim).cuda()
structure_velocity = structure_adapter(noisy_structure, timestep, conditions)
shape_velocity = shape_adapter(noisy_shape_latent, timestep, conditions)
```

`point_dim` 必须与 PTv3 输出的逐点特征维度相符。上面 `768` 是示例图像维度，需以实际 DINOv3 模型输出为准。`view_features` 中的特殊 token 是否保留，须在整个训练/推理流程中保持一致。

## 5. 环境与验证

基线：Python 3.12 / PyTorch 2.7.1 / CUDA 12.8。SindrePy 私有源码应在本地单独安装；官方 TRELLIS.2 权重和扩展也需按其项目安装。本仓库并不声称 `uv sync` 会安装这些私有源码或权重。

```powershell
uv sync
uv pip install --python .venv\Scripts\python.exe --no-deps --editable ..\SindrePy
uv pip install --python .venv\Scripts\python.exe pytest
uv run pytest -q
```

单次数据预处理也可直接调用 `process.main(input_path, num_points=4096, augment_count=4)`，无需 argparse。建议先预生成三视图、DINOv3 tokens 和局部坐标，再进行随机牙弓裁剪训练，避免每个 epoch 重复渲染。

测试当前覆盖条件 token 形状、无牙弓掩码、真实 FDI 转换、唯一开放边界提取、缺失条件识别。尚需在配有官方 CUDA 扩展与模型权重的设备上做实际 Flow 前向、反向和完整生成测试，并验证牙冠几何质量。
