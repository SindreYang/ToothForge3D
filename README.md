# ToothForge3D

基于 SindrePy 和 O-Voxel / FDG 的**专用开放牙冠生成**研究工程。目标是给定真实牙颌、FDI、颈缘线，自动生成 CAD 可编辑的开放牙冠；**MD、BL、H 为可选设计约束**，用户不知道时由模型先估计，生成后再测量真实尺寸。

> 当前代码已包含轻量网络、成对数据增强、训练与推理接口；**尚未在真实 4090、O-Voxel CUDA 与独立病例上完成端到端实测**。不要将代码实现等同于已验证的牙冠生成精度。

## 目录职责

| 文件 | 职责 |
| --- | --- |
| `src/model.py` | **唯一网络结构文件**：SindrePy MCDG 条件编码、轻量 coarse-to-fine FDG 预测、MD/BL/H 物理尺度条件 |
| `src/train.py` | **唯一训练文件**：train/val、tqdm、BF16、Profiler、分组验证、最佳/最近权重 |
| `src/infer.py` | **唯一推理入口**：从病例 JSON 或原始 Sindre LMDB 的 `path` 读取、预测开放冠、世界坐标保存 Mesh |
| `src/datasets.py` | 仅 FDG v2 的 `FDGDataset`、训练查询采样与在线邻牙增强（不再提供 Occupancy 构建接口） |
| `src/utils.py` | 原始病例直转 O-Voxel FDG、固定颈缘线的目标冠尺寸增强、独立 LMDB 写入和边界小幅修正 |
| `src/losses.py` | **唯一 Loss 文件**：结构 BCE+Dice、offset SmoothL1、intersection BCE、MD/BL/H 实际尺寸与边界到颈缘线双向软约束 |
| `process/process.py` | 真实病例字段、FDI、局部坐标与无设计冠的推理条件构建 |
| `src/views.py` | 真实牙颌颊/舌/咬合三视图与 SindrePy DINOv3 编码；不输入设计冠 Mesh |

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

**邻牙与咬合安全门槛**：增强候选还会经过单开放边界环、连通性与自交检查；邻牙/对颌使用 0.05mm 近接触阈值作为保守拒绝条件。默认 `collision_check=True`，尺寸增强前使用真实病例的同颌牙位标签与已配准对颌扫描构建局部环境 Mesh。复用 SindrePy `AdjacentCollisionDetector(solid=False)` 和 MeshLib 检查每个增强后的牙冠是否新增邻牙/对颌三角面相交；每份增强最多尝试12次，冲突或变形产生自交时拒绝并重新采样。原始设计冠如果已存在相交/接触，则保留原始样本但**停止为该病例生成尺寸增强**，因为开放 Mesh 无法测量穿透深度，也就无法证明形变没有加重穿透。无对颌扫描或原始病例环境缺失则报错；仅接受原始病例 LMDB，不再提供 Occupancy/SDF 迁移接口。测试/实验可显式设 `collision_check=False`，**但未经检查的样本不能视为几何合格**。

`build_database` 返回的报告包含 `rejected_collision`、`skipped_augmentation` 和 `warnings`，因此碰撞筛选后每病例实际训练样本可能少于4份。该检测**只确认三角面是否相交/接触，不计算实体穿透深度、咬合力或真实动态咬合**；还需检查对颌是否在同一毫米坐标系完成配准。

在线 `FDGDataset` 每次读取均重新采样同颌邻牙、以小概率删除整颗邻牙、对同颌 XYZ 加 0.02mm 轻噪声，以及以小概率遮蔽一张缓存视图；持久 DataLoader worker 的随机数状态持续前进。**不会在线单独抖动 `size` 或颈缘线**。需要增加尺寸增强版本时，重新预处理原始冠并生成新 FDG 数据库。

## 原始坐标先验与牙弓局部坐标规范（待实现）

以下是已确认的原始数据特征及下一版统一坐标方案，不代表当前代码已经实现。

### 已知数据先验

- 原始数据采用毫米单位。每份病例的世界 +Z 轴大致指向目标牙冠冠顶，与目标牙的近似插入轴、所在牙颌的咬合轴相近，但存在角度偏差；这里的冠向与实际插入运动的方向符号应明确区分。
- 原始 X、Y 轴没有固定的近远中（MD）、颊舌（BL）解剖含义，绕 Z 轴的旋转需要归一化。
- 训练库可能具有邻牙分割与 FDI 标签，但实际推理不能假设邻牙已经分割或标注牙位。目标 FDI 与颈缘线已知，不等于其他牙齿的 FDI 已知。
- 设计冠只能用于生成监督标签、测量目标尺寸和离线评价；禁止用其形态或插入轴确定正式训练输入的局部坐标系。

### 无邻牙分割时的统一方向计算

第一版保留原始 +Z = (0, 0, 1) 作为冠向先验，只估计 XY 平面内的方向并平移原点。实际推理若不满足该 Z 轴先验，须先进行外部姿态对齐或提供可靠冠向，不得直接套用。

1. 真实闭合颈缘线等弧长采样为 128 点，取采样均值 O 为局部原点。
2. 将同颌扫描投影至垂直于 Z 的平面，基于咬合侧几何、牙列轮廓或稳健的空间采样提取牙弓代表点，减少牙龈、扫描底座及离群点影响。不能简单使用全部 Mesh 顶点拟合，否则采样密度和牙龈面积可能主导结果。
3. 对代表点拟合平滑牙弓中心曲线（可采用稳健样条或其他经过真实病例验证的方法）。以颈缘中心的投影在曲线上定位目标位置，计算局部切向作为 MD 轴候选。若最近位置存在歧义，结合目标 FDI、局部几何和曲线连续性判定，而非只依赖全局 PCA。
4. 根据可辨识的牙弓内侧/外侧关系确定 +Y 指向颊侧（前牙为唇侧），再以 X = Y × Z 构造近远中轴，重新正交化并保证右手坐标系。X 表示近远中轴，不承诺所有左右侧、上下颌的 +X 都指向近中。可用目标 FDI 结合可靠的牙弓中线推断近中符号；仅凭 FDI 或无向切线不能消除 180° 歧义。
5. 邻牙分割标签存在时可以利用牙齿质心辅助曲线拟合、校验方向，但不得作为基础算法的必需条件。无标签时直接使用几何曲线方案。
6. 半牙弓优先使用可见牙弓曲线的局部切向；残缺或曲率不足时降低方向置信度。仅有预备体/颈缘线时，可用颈缘线 PCA 提供无向轴候选，但无法保证 MD/BL 身份或颊舌正负方向，尤其近圆形颈缘线。低置信度时允许用户提供局部轴或补充扫描，不得将任意方向标记为解剖方向已验证。

### 训练、预处理与推理一致性

- 训练与推理共用同一个不依赖设计冠、无需邻牙 FDI 标签的坐标构建接口。先实现方向估计及置信度，再考虑基于预备体和颈缘线对 Z 轴进行有限角度修正；原始 Z 轴偏差分布需实测，不预设固定误差范围。
- 统一保存 origin_world、local_frame、world_to_local_transform、frame_source 和方向置信度。扫描、对颌、颈缘线、目标冠及三视图相机使用同一变换；世界坐标恢复必须使用该变换的逆矩阵，不能重新计算未经重采样的颈缘线均值。
- MD、BL 尺寸沿统一局部轴测量，H 为颈缘原点到冠顶的局部 Z 高度；三视图按统一的颊/舌/咬合方向生成。不能可靠确定方向时，不应宣称 MD/BL 已经具有准确的解剖意义。
- 坐标系变化后必须重新生成三视图缓存、点云条件、FDG 标签和尺寸监督；变更坐标规范版本，重新构建训练库并训练对应权重。
- 无邻牙标签的完整推理链路还需要同步调整点云编码、场景构建及碰撞检测：不能把未知标签当成已知背景或伪造 FDI；可用显式未知标签/掩码及几何环境检测。当前实现仍读取 current_mesh.vertex_labels，并不具备完整的无分割输入支持。
- 验收应覆盖完整、半颌、残缺及无分割牙弓，对同一病例绕原始 Z 旋转前后检查局部坐标、三视图、尺寸、FDG 的一致性，并检查世界坐标往返误差与实际冠体叠加效果。

## 一次性数据预处理

原始病例应具有 `current_mesh`、`opposite_mesh`、`design_mesh`、FDI、真实颈缘线和**仅由输入扫描/预备体确定的 `local_frame`**。允许原始开放冠；已封底的冠尝试用 SindrePy MeshLib 沿颈缘线裁剪后再转换。对冠体异常、边界太远、FDG 重复/越界坐标直接报错。**当前实现若没有外部 local_frame，可能利用设计冠估计 Z 轴及同侧邻牙确定 XY 方向；仅设计冠方向来源会被拒绝，但 design_crown+same_jaw 来源仍被允许，存在训练与推理条件不一致的风险。** 在上述新坐标方案实现前，应提供仅由输入扫描确定的 local_frame。

```python
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from src.utils import CrownFDG, FDGConfig

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


## LMDB 数据库结构与读取

本项目读写两类 SindrePy LMDB：**原始病例库**是预处理输入，保存扫描和病例标注；**FDG 训练库**是 CrownFDG.build_database 的输出，保存训练条件、设计冠监督和 O-Voxel 标签。两者都不是 SQLite 文件，使用 sindre.lmdb.Reader 按样本序号读取，不要用 sqlite3 或直接反序列化文件。

### 打开原始病例库

~~~python
from sindre.lmdb import Reader

with Reader("data/raw_cases.db") as reader:
    print("病例数:", len(reader))
    print("schema_version:", reader.get_meta("schema_version"))
    case = reader[0]  # 有效序号为 0 到 len(reader)-1
    print("顶层键:", sorted(case.keys()))
    print("current_mesh 键:", sorted(case["current_mesh"].keys()))
~~~

原始库的 metadata 由上游数据生成器决定，schema_version 可能缺失；先用 get_meta 检查，不要假定所有来源都有相同 metadata。当前全牙样本的顶层字段包括 order_name、is_upper、current_mesh、opposite_mesh、design_mesh；order_name 是来源病例/订单标识，分享前必须先确认已去标识。病例记录的主要字段如下：

| 键 | 内容与用途 |
| --- | --- |
| current_mesh | 必需。当前牙颌扫描字典；含 vertices [V,3]、faces [F,3] 和逐顶点 vertex_labels [V]。顶点是毫米世界坐标；fdi 可选，存在时作为标签索引到 FDI 值的映射表。标签用于区分牙位、背景和预备体。 |
| opposite_mesh | 可选的对颌扫描字典，至少含 vertices、faces。用于三视图背景、可视化和增强碰撞检查；不是模型的同颌点云输入。 |
| design_mesh | 训练预处理必需，含设计冠 vertices、faces、fdi；只作为目标几何和监督。可含 margin_line、size_mm。推理时不得把此字段作为条件输入。 |
| target_fdi | 可选目标牙 FDI。提取脚本优先读它；缺失时读取 design_mesh.fdi。推理时应显式提供目标牙位。 |
| margin_line | 真实颈缘线 [M,3]，毫米世界坐标，优先级高于 design_mesh.margin_line。缺失时训练流程只会在设计 Mesh 恰有一个有效开放边界环时尝试提取。 |
| local_frame | 可选 [3,3] 右手正交坐标基，列依次为近远中、颊舌、冠向，且由输入扫描/预备体确定。推理时必需；训练可按坐标轴来源校验规则推断。 |
| size_mm | 可选 [MD, BL, H] 毫米尺寸。训练时优先使用它，否则可由设计冠测量；推理时只有用户明确指定尺寸才填写。 |
| view_features | 可选预计算三视图特征 [3,T,C]。若缺失，预处理/推理需提供 DINOv3 encoder 计算。改变牙位、颈缘线或坐标系后，旧缓存不可复用。 |
| order_name、is_upper | 来源库可带有的订单标识及上颌标记；不是网络输入。order_name 可能含可识别信息。 |
| design_info、score | 来源库可能附带的设计说明或评分；当前训练/推理条件构建不读取它们。 |

Mesh 的 vertices、颈缘线和矩阵原点必须处于一致的毫米世界坐标系；faces 是从 0 开始的三角面顶点索引。读取原始训练库时，设计冠是标签，不能作为推理条件。

### 原始库筛选器写入的 metadata

process.extract_simple.extract_simple 输出每个 FDI 最多前 5 例的子库，并尽量复制源 schema_version。它还写入 samples_per_tooth、tooth_counts、source_count 和 filtered_count。process.extract_6.extract_6 输出 FDI 16/26/36/46 子库，写入 filter_fdi、source_count 和 filtered_count，也会复制源 schema（若有）。这两个输出仍是**原始病例库**，可继续交给预处理；不是 FDG 训练库。

### FDG 训练库 metadata 和样本键

训练库顶层 metadata 由预处理器写入：

| metadata 键 | 作用 |
| --- | --- |
| schema_version | 固定为 toothforge3d_fdg_v2，供读取端识别库类型和版本。 |
| build_complete | 写入开始为 false，完整结束后为 true；训练读取端拒绝不完整的库。 |
| coordinate_frame | 固定为 margin_only_v2；FDG 网格尺度由颈缘线确定，不依赖目标尺寸。旧版 margin_size_v1 不能直接用于当前训练。 |
| fdg_config | FDGConfig 的分辨率及 O-Voxel 网格参数。 |
| augmentation | 配对尺寸增强设置、碰撞筛选开关和后端说明；碰撞检测不代表测得实体穿透深度。 |
| written | 成功写入的样本条数。 |
| source_groups | 与样本索引逐项对应的原始病例组号；同一病例的尺寸变体必须落入同一训练/验证分区。 |

每条 FDG 样本的主要键：

| 键 | 内容与用途 |
| --- | --- |
| jaw、target_position | 上/下颌类别及 SindrePy 的目标牙位索引（1–16）。 |
| margin_line、size | 局部坐标下的颈缘线，以及监督尺寸 [MD, BL, H] 毫米。增强样本的尺寸对应增强后的冠体。 |
| jaw_points、jaw_labels、jaw_present | 同颌 PTv3 点云、逐点牙位标签（0 为背景、1–16 为牙位）及是否存在有效点云。 |
| view_features | 预计算 DINOv3 三视图特征，形状为 [3,T,C]。 |
| target_vertices、target_faces | 局部毫米坐标下的开放目标冠 Mesh，供监督、几何验证和尺寸测量；属于 GT，不可作为推理输入。 |
| fdg_coords | O-Voxel 活跃体素坐标 [N,3]，每个轴范围为 [0, fdg_resolution)。 |
| fdg_dual_offsets | 与活跃体素一一对应的双顶点偏移 [N,3]。 |
| fdg_intersected | 与活跃体素一一对应的交线标签 [N,3]。 |
| fdg_resolution | FDG 网格边长；必须与训练配置一致。 |
| fdg_center_mm、fdg_scale_mm、fdg_margin_line_mm | FDG 网格与毫米坐标之间的中心、尺度和颈缘线恢复信息。 |
| source_group、variant_index | 原病例组号和配对变体序号；variant_index=0 是原始设计冠，其余为尺寸增强冠。 |
| origin_world、local_frame、world_to_local_transform、frame_source | 可选坐标回写信息：局部原点、局部轴、世界到局部变换及坐标轴来源。 |
| source_fdi | 可选源病例牙位记录。 |

样本记录中不保存原始 raw_order；原始病例分组由 source_group 和 metadata 的 source_groups 表示。训练入口 FDGDataset 会检查版本、完成标记、坐标规范、分辨率以及必须的键；推理入口只接受原始病例库或无 GT 的病例 JSON，明确拒绝此训练标签库。

也可用同一个 Reader 检查 FDG 库：

~~~python
from sindre.lmdb import Reader

with Reader("data/toothforge_fdg_v2.db") as reader:
    print("样本数:", len(reader))
    for key in (
        "schema_version", "build_complete", "coordinate_frame",
        "fdg_config", "augmentation", "written", "source_groups",
    ):
        print(key, reader.get_meta(key))
    sample = reader[0]
    print("样本键:", sorted(sample.keys()))
~~~

**真实长宽高来源要明确**：训练集可以使用已完成设计冠的 MD/BL/H 作为**监督标签**，但不能把它们当作所有样本的必备模型输入。训练默认 65% 的病例隐藏输入尺寸，独立的尺寸回归头始终接受目标尺寸监督；推理默认不需要用户填写，只有明确指定尺寸时才使用用户条件。

## 训练

```python
from src.train import train, FDGTrainingConfig
from src.model import LightFDGConfig
from src.losses import FDGLossConfig

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

损失只放在 `src/losses.py`，并在 FP32 计算：`L = 1.0 L_coarse + 1.0 L_fine + 0.5 L_offset + 0.5 L_intersection + 0.2 L_margin + 0.2 L_size + 0.4 L_size_prior`。其中 `L_size` 对从 O-Voxel 解码的真实 Mesh 测量 MD/BL/H，沿颈缘线局部坐标直接监督输出几何，而不仅是尺寸 Embedding。类别不平衡通过有限正例权重和 Dice 稳定处理；Margin Loss 允许约 0.15mm 柔性容差。**不使用视图 Loss**，三视图只在输入预处理时编码。

训练 tqdm 显示 coarse/fine/offset/召回率及间隔性实际尺寸误差（mm）。训练采用 AdamW、BF16、梯度累积/裁剪、SindrePy best/last checkpoint 和可选 torch.profiler。CPU 数学测试可以使用 `margin_every=0`，但完整 FDG 网格训练与推理依赖官方 O-Voxel CUDA 环境。

## 推理

`src/infer.py` 接收**真实牙颌病例路径**和最佳权重，病例不能依赖目标设计冠 Mesh。要求输入：

- `target_fdi`：修复目标牙位；
- `margin_line`：输入扫描坐标下的真实颈缘线（毫米）；
- `size_mm=(MD,BL,H)`：**可选**，用户指定的冠体长宽高（毫米）；不指定则由模型自动估计；
- `local_frame`：近远中、颊舌、冠向构成的 3×3 右手正交矩阵；
- `checkpoint_path`：训练生成的 best.pt 或 last.pt。

必要条件既可保存在病例 JSON 中，也可通过函数传参覆盖。病例没有缓存的三视图特征时，由 SindrePy DINOv3 在推理准备阶段计算。**真正推理不读取目标冠坐标或 FDG 监督**。

```python
import numpy as np
from src.infer import infer

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
