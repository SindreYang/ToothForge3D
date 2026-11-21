# ToothForge3D

**MCDG — Multi-Condition Dental Geometry**

以 SindrePy 为基础的全冠三维形态生成研究项目。当前代码包含真实病例的同颌牙弓上下文处理、三视图/牙位/颈缘线/尺寸条件编码，以及 TRELLIS.2 Sparse Structure / Shape Flow 的条件适配器。

**当前并非完整的牙冠生成产品**：本仓库尚未提供训练好的权重、Flow Matching 训练循环、Shape VAE 网格解码整合、颈缘线硬约束与临床 CAD 验收；这些不能从编码器的输出自动推断为已实现。

## 1. 输入契约

| 输入 | 类型 / 形状 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| `view_features` | `[B,3,T,C]` | 是 | 设计冠只用于辅助定向；颊侧/舌侧渲染 `current_mesh + opposite_mesh`，咬合侧仅渲染 `current_mesh`，再经 SindrePy DINOv3 编码 |
| `jaw` | `[B]` | 是 | 0 上颌，1 下颌 |
| `target_position` | `[B]` | 是 | SindrePy FDI→1~16 索引 |
| `margin_line` | `[B,128,3]` | 是 | 真实闭合颈缘曲线，毫米 |
| `size` | `[B,3]` | 是 | 近远中宽、颊舌厚、高，毫米 |
| `jaw_points` | `[B,N,6]` | 否 | 同颌牙弓 XYZ + Normal；不输入对颌 |
| `jaw_labels` | `[B,N]` | 有牙弓时必须 | 每点 0（背景/未知）或 1~16 牙位 |
| `jaw_present` | `[B]` | 有牙弓时可选 | 显式标识整条牙弓条件是否存在；允许部分样本无牙弓 |

**牙位规则要严格统一**：预处理直接读取设计记录中的 `fdi` 自动推导上下颌（11~28 为上颌；31~48 为下颌且需满足标准象限），对颌类别等于 `1-jaw`，但**仅将对颌用于侧视图渲染，不作为 PTv3 点云几何输入**。SindrePy 的 `convert_fdi2idx` 将 11~18→1~8、21~28→9~16，31~38→1~8、41~48→9~16。示例：36→6、46→14。输入中没有额外的“近中/远中角色”；牙位与空间坐标足以提供上下文。

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
    "opposite_mesh": {     # 必须：颊舌侧两张图需要上下颌
        "vertices": ...,
        "faces": ...,
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

`current_mesh` 中其他同颌牙和牙龈进入 PTv3；目标成品冠会被排除。类似 `2031=2000+31` 的预备体几何保留在 **vedo 渲染场景及相机取景**中，但不伪装为普通邻牙。标签表索引与真实 FDI 均可辨别。目标设计冠 `design_mesh` 只用于方向、尺寸和训练监督。

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

# 使用设计冠辅助确定方向；侧视图拍上下颌，咬合图只拍当前牙颌。
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

- **三视图**：提供 `render_views=True` 或 `image_encoder=DINOv3Encoder()` 后，自动从设计冠确定相机局部方向，**随后移除设计冠**；实际拍摄时颊侧和舌侧使用 `current_mesh + opposite_mesh`，咬合视图只拍目标牙颌 `current_mesh`。未提供 `opposite_mesh` 时无法生成符合要求的两张侧视图，会明确报错。三个相机都注视颈缘线中心，距离来自目标牙预备体 `2000+FDI`、SindrePy MeshLib 裁剪得到的基座或颈缘线范围；侧视图正交视场在近邻对颌存在时适度放大以覆盖两颌。三视图不含目标答案，因此能作为牙颌条件输入，但仍需评价模型是否真正受 FDI、尺寸和局部几何约束。
- **颈缘线**：优先读取真实标注。仅当设计冠具有唯一闭合开放边界时可自动重采样。封闭冠或多边界环不能猜测。
- **生成尺寸**：优先使用 `size_mm`；否则从设计冠局部坐标系沿 MD / BL / 冠向测量尺寸。该尺寸是基于估计坐标轴的几何标注，而非临床验证的解剖测量。
- **相机和坐标**：使用 SindrePy 的单牙长轴估计结合可用的同颌牙位推导局部坐标系，固定相机尺度、背景和方向。孤立牙仅能可靠得到冠向与两条相对侧视角，无法单凭 PCA 识别真正的颊、舌方向；此时 `frame_verified=False`，应人工检查或提供 `local_frame`。
- **对颌信息**：FDI 可推导对颌是上颌还是下颌。`opposite_mesh` 不送入 PTv3 点云分支，但会通过颊/舌侧图像间接影响网络；应在训练中验证其错误咬合的鲁棒性。

未提供或生成 `view_features` 时，`ready_for_training=False`；仅获得三视图 RGB 而未经过 DINOv3 编码时，也还未完成网络输入准备。

## 4. 原始病例数据字段利用情况

| 字段 | 当前用途 | 备注 |
| --- | --- | --- |
| `design_mesh` / `design` | FDI、生成冠形态监督、局部方向、尺寸测量 | 不进入视图渲染对象 |
| `current_mesh.vertices/faces` | 同颌网格、三个视图的目标颌与局部基座 | Vedo 渲染；几何法向由 SindreMesh 提取 |
| `current_mesh.vertex_labels/fdi` | 同颌逐点位置标签、目标成品冠排除、预备体 2000+FDI 提取 | 目标成品牙不回流到输入 |
| `margin_line` 或唯一开放边界 | 曲线 tokens、相机焦点、基座裁剪、局部原点 | `merge_info.margin_vertices` 在坐标匹配时可复用 |
| `size_mm` | 明确 MD/BL/H 约束 | 未提供时在牙冠局部坐标下推导 |
| `is_upper`、`order_info.tooth_fdi` | 与设计 FDI 互相检查 | 不需要重复输入到网络 |
| `design_info` | 仅识别记录是否存在，尚未作为训练监督 | 旧隐式模型内的 Occupancy 查询点可能在归一化坐标，不能直接混入当前毫米空间 |
| `opposite_mesh` | 仅用于颊/舌侧渲染，不参与咬合视图和 PTv3 点云 | 对颌咬合可能有误，图像条件仍会间接受其影响 |

使用 `vedo.Plotter(offscreen=True)` 渲染；`SindreMesh` 用于计算网格几何，曲线裁剪复用 SindrePy MeshLib 实现 `sindre.utils3d.algorithm.cut_mesh_by_meshlib`。所有相机焦点为 **颈缘线中心**，相机位置和正交投影视场由颈缘线/预备体确定，不受目标设计冠的几何尺寸直接控制。

训练时牙弓点云随机裁剪/丢牙；三视图使用目标周边的局部正交视场，因此远处牙颌通常不在画面中。

## 5. 条件编码示例

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

## 6. 环境与验证

基线：Python 3.12 / PyTorch 2.7.1 / CUDA 12.8。SindrePy 私有源码应在本地单独安装；官方 TRELLIS.2 权重和扩展也需按其项目安装。本仓库并不声称 `uv sync` 会安装这些私有源码或权重。

```powershell
uv sync
uv pip install --python .venv\Scripts\python.exe --no-deps --editable ..\SindrePy
uv pip install --python .venv\Scripts\python.exe pytest
uv run pytest -q
```

单次数据预处理也可直接调用 `process.main(input_path, num_points=4096, augment_count=4)`，无需 argparse。建议先预生成三视图、DINOv3 tokens 和局部坐标，再进行随机牙弓裁剪训练，避免每个 epoch 重复渲染。

测试当前覆盖条件 token 形状、无牙弓掩码、真实 FDI 转换、唯一开放边界提取、缺失条件识别。尚需在配有官方 CUDA 扩展与模型权重的设备上做实际 Flow 前向、反向和完整生成测试，并验证牙冠几何质量。


## 7. 新版 MyDataset 与占据场训练数据库

已新增 `datasets.py`，不再使用旧 `ShapeNet_Atten/datasets.py` 的固定邻牙输入或把 0/1 Occupancy 误称为 SDF。旧 `design_info` 的三档采样是在旧归一化坐标下生成的，新版全部在设计冠局部 **mm** 空间重新采样，并保存真实 SDF（负值为冠内）与独立 Occupancy（0外、1内）。

**固定的初始采样配置（可通过 `SamplingConfig` 调整）**

| 区域 | 点数 | 噪声 / 分布 |
| --- | ---: | --- |
| 冠表面极近层 | 8,192 | 沿三角面法向高斯噪声 `σ=0.05 mm` |
| 冠表面近层 | 4,096 | 法向高斯 `σ=0.15 mm` |
| 冠表面中层 | 4,096 | 法向高斯 `σ=0.35 mm` |
| 空间随机 | 16,384 | 设计冠 AABB 每边外扩 `0.8 mm` 后均匀采样 |
| **总计** | **32,768** | 法向噪声还包含 `0.15×σ` 的切向分量 |

此配置用于首批实验，**不是经过模型精度实验验证的最优值**。没有通过采样来强制 50% 内外均衡，真实占据比例根据闭合冠计算。

**严格的 Occupancy 正确性检查**：

1. 设计冠不水密时默认通过 `vedo.Mesh.fill_holes()` 修补；必要时可将 `repair="sindre_views"` 切换到 SindrePy 多视图水密化，二者均失败则拒绝生成标签。**近表面点只在原始设计冠表面采样，人工补出的冠底盖只用于内外 SDF 判断，不计入表面噪声采样**
2. SDF 调用 SindrePy `GetSDF.apply_pcu`，可通过 `sdf_backend="cubvh"` 选择 GPU 后端
3. SDF 必须有限；六个明确的远外部点必须全部为正；内部点必须满足 `SDF<0`，外部 `SDF>=0`
4. 默认检查内部和外部至少各占总采样数的 2%；不满足视为数据异常，不写入这份样本
5. 全局 `DEBUG=True` 输出半透明设计冠、蓝色内部点和橙色外部点的离屏截图，以及**每个噪声层的内部点比例**
6. 每个设计冠最大尺度必须在 `[2,30] mm`，否则拒绝构建，避免旧归一化坐标与毫米噪声混用

**生成全新 Sindre LMDB**

```python
import datasets as ds

ds.DEBUG = True  # 调试时打开；正式批处理置为 False
config = ds.SamplingConfig(
    num_jaw_points=8192,
    sdf_backend="pcu",   # CPU；有 cubvh 可切到 CUDA
    repair="vedo",
    seed=1024,
)

report = ds.build_database(
    "data/raw_cases.db",
    "data/toothforge_occ_v1.db",
    config=config,
    map_size_mb=32768,    # Sindre Writer 的参数单位是 MB
    max_cases=5,          # 先验证 5 个真实病例，确认后去掉该参数
    strict=True,
)
print(report)
```

打开 `DEBUG` 后截图输出到 `data/toothforge_occ_v1_debug/`，写入报告保存在 `data/toothforge_occ_v1.report.json`。源数据库保持只读，若输出路径已存在则拒绝覆盖；生成中断的数据库缺少 `build_complete=True`，`MyDataset` 拒绝读作完整训练集。

```python
from torch.utils.data import DataLoader
from datasets import MyDataset

dataset = MyDataset(
    "data/toothforge_occ_v1.db",
    augment=True,         # 每次访问随机裁剪/删牙，仅在同颌点云上执行
    jaw_points=4096,
    tooth_dropout=0.20,
    jaw_noise_mm=0.02,   # 可选扫描点噪声，标签点不再被扰动
)
loader = DataLoader(
    dataset, batch_size=2, num_workers=0,
    collate_fn=MyDataset.collate,
)
batch = next(iter(loader))
print(batch["query_points"].shape)  # (2,32768,3)
print(batch["occupancy"].shape)     # (2,32768)
print(batch["view_images"].shape)   # (2,3,3,256,256)
```

数据库中保留 `target_vertices/target_faces` 以供主生成网络监督。三视图保存为 RGB `uint8`；传入 `image_encoder=DINOv3Encoder()` 时会额外保存 `view_features`，否则训练时需自行用 DINOv3 编码图像再送入 `MCDGConditionEncoder`。占据场目前作为**辅助几何监督和可视化验证数据**，并非已经接入 TRELLIS.2 的 Flow Loss。

本仓库的 `datasets/*.json` 是独立冠 Mesh，不含完整病例的 `current_mesh/opposite_mesh`、`fdi` 等，不能用于直接构建新的条件数据库。需要真实病例 LMDB。附加几何处理依赖 `trimesh`、`vedo`，默认 CPU SDF 需 `point_cloud_utils`，以及 SindrePy 本身的 `lmdb/msgpack` 依赖；这些需在构建数据库的环境中安装。


## 8. 模型、Loss 与训练（本次架构重构）

**本阶段实际可训练的网络**是 `MCDGReconstructionModel`：三视图 DINOv3 tokens + 强目标 FDI + 颈缘线 + 尺寸 + 随机裁剪同颌牙弓经 `MCDGConditionEncoder` 融合，再由 `CrownImplicitDecoder` 进行隐式空间查询。对于 `Q=32768` 的查询点按固定 chunk 逐块 Cross-Attention，输出 `occupancy_logits [B,Q]` 与 `sdf_mm [B,Q]`。它提供可验证的三维**重建监督基线**，不是已经训好的网格生成器。

### Loss

```
Loss = 1.0 × balanced_BCEWithLogits(Occupancy)
     + 0.2 × SmoothL1(clamp(predicted_SDF,±1mm), clamp(target_SDF,±1mm))
```

- `sindre.ai.losses.balance_bce_loss` 负责内部/外部正负点平衡；没有再造同一份 BCE、Dice 或 Focal Loss
- 四组 Occupancy 采样区域各自计算损失后等权平均，避免多出的空间随机点淹没边界训练
- `torch.nn.functional.smooth_l1_loss` 回归真实毫米 SDF，默认 `beta=0.1mm`、截断带宽 `1.0mm`；此处仅监督有限带宽的几何，不是精确临床间隙损失
- 同时统计 Occupancy Accuracy/IoU；暂未实现 CAD 厚度、颈缘线对齐、邻接碰撞、对颌咬合的可微分损失
- 旧私有项目的 Chamfer/Edge/Normal/Laplacian Loss 适用于**已有预测 Mesh** 的后续阶段；当前隐式查询头没有输出 Mesh，因此没有错误套用这些 Loss

### 训练入口与 tqdm

```python
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from datasets import build_database, SamplingConfig
from train import train, TrainingConfig

# 数据预处理阶段只做一次；先用 max_cases=5 实际抽样验证。
with DINOv3Encoder() as dino:
    build_database(
        "data/raw_cases.db", "data/toothforge_occ_v1.db",
        image_encoder=dino,
        config=SamplingConfig(num_jaw_points=8192),
        max_cases=5,
    )

result = train(
    "data/toothforge_occ_v1.db",
    config=TrainingConfig(
        epochs=100, batch_size=2,
        train_queries=4096, val_queries=8192,
        query_chunk_size=512,
        checkpoint_dir="checkpoints/toothforge_implicit",
    ),
)
print(result)
```

训练采用 `tqdm.auto.tqdm` 为每个 epoch 的 train 和 val 显示实时进度以及 `loss/occ/sdf/IoU/lr`；同时有可选 bf16 自动混合精度、AdamW、余弦学习率、梯度累积与裁剪、断点恢复。复用 SindrePy 的 `set_global_seeds`、`save_checkpoint`、`load_checkpoint`，将验证集最优权重写入 `best.pt`，最近一轮写入 `last.pt`。

正式实验应先按**患者/订单**划分 train/val 两个 LMDB，然后向 `train(..., val_db="...")` 传入独立验证库；省略 `val_db` 时只是同数据库按样本索引固定随机划分，不保证患者完全不重复。验证集的牙弓采样采用固定 index seed，且不启用随机裁剪，保证多次验证可对比。

**数据前提**：新 LMDB 需要调用 `build_database(image_encoder=dino)` 预计算 DINOv3 `view_features`。仅有 `view_images` 的数据库不能直接送入条件编码器，训练会明确报错，而不是把像素伪装为 DINO patch 特征。

### TRELLIS.2 第二阶段与开源复用

`TrellisFlowAdapter` 已经删除自行复制的 dense/sparse 模块内循环。仅进行多条件投影、添加 FDI 强全局条件 token，再调用官方 `flow_model(x,t,cond)`。后者保留官方位置编码、稀疏布局、调制和 dtype 逻辑。

`FlowMatchingLoss` 按官方 `trellis2/trainers/flow_matching/flow_matching.py` 同一公式计算：

```text
x_t = (1-t) × x_0 + (sigma_min+(1-sigma_min)×t) × epsilon
v_target = (1-sigma_min) × epsilon - x_0
L_flow = MSE(v_pred, v_target)
```

默认 `sigma_min=1e-5`、调用 Flow 的时间参数为 `t*1000`。对稀疏 SLat，必须用 TRELLIS.2 原生 SparseTensor 及其真实编码坐标；不允许直接把 Occupancy 点、FDI 或目标 mesh 顶点假装成 Flow 的 `x_0`。

**目前尚未完成**将原始设计冠通过官方 TRELLIS.2 Structure/Shape VAE 编码成 `x_0` 的预处理、双阶段 Flow 训练数据集、Shape VAE 网格解码与 CAD 后处理。因此第二阶段公式和条件适配器已实现，不代表完整 TRELLIS.2 生成训练已跑通。此限制在接口和单元测试里明确区分。

### 代码复用依据

| 来源 | 本项目复用 | 未直接复制的部分 |
| --- | --- | --- |
| SindrePy | `PointTransformerV3`、`DINOv3Encoder`、`FourierEmbedder`、`MLP`、`ResidualCrossAttentionBlock`、`balance_bce_loss`、checkpoint 和种子工具 | 不重复维护网络底层、损失或检查点格式 |
| 私有 `Denta_3d_project_private/ShapeNet_Atten` | 牙位 Embedding、线/尺寸 Fourier 特征、PTv3 + Query CrossAttention 的经验 | 旧版 `argparse`、不存在的接口参数、将 Occupancy 当 SDF 的名称混乱 |
| Microsoft TRELLIS.2 | 官方 Structure/Shape Flow 的 `forward` 与 Flow Matching 定义 | 不复制官方 dense/sparse Transformer block 内部实现 |
| PyTorch/tqdm | AdamW、CosineAnnealingLR、bf16、SmoothL1、原生 DataLoader、进度条 | 不自行编写优化器、调度器或进度条 |

新增测试 `tests/test_training.py` 检查条件输出、Chunk Query、梯度反传、Occupancy/SDF Loss、真实 Flow 公式及调用官方前向方法的协议。实际 VTK/CUDA/PTv3/官方 TRELLIS.2 完整环境尚需在你的工作站运行：
```powershell
uv lock
uv sync
uv pip install --python .venv\Scripts\python.exe --no-deps --editable ..\SindrePy
uv pip install --python .venv\Scripts\python.exe pytest
uv run pytest -q
```
这次 `pyproject.toml` 已新增 tqdm 依赖；`uv.lock` 需要通过 `uv lock` 重新生成。
