"""牙冠专用网络。

集中管理 MCDG 多条件编码与轻量 FDG 结构/几何预测。
三视图只输入离线 DINOv3 tokens；FDI、真实颈缘线与目标长宽高均为强条件。
训练正负采样只决定监督查询位置，推理不读取真实设计冠 FDG 坐标。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sindre.ai.attention_utils.cross_attention import ResidualCrossAttentionBlock
from sindre.ai.attention_utils.multihead_attention import ResidualAttentionBlock
from sindre.ai.embedder import FourierEmbedder
from sindre.ai.layers import MLP


class MCDGConditionEncoder(nn.Module):
    """提取三视图、FDI、颈缘线、MD/BL/H 与可选同颌点云的条件。

    牙位遵循 SindrePy convert_fdi2idx 的同颌 1～16 编码。

    Args:
        image_dim: 输入离线 DINOv3 token 通道数。
        hidden_dim: 条件 token 通道数。
        point_dim: SindrePy PTv3 的输出通道数。
        num_heads: 条件注意力头数。
        frequencies: Fourier 位置编码频率数。
        margin_points: 颈缘线等弧长采样点数。
        view_tokens: 每个视角保留的特征 token 数。
        jaw_tokens: 同颌点云聚合 token 数。
        point_encoder: SindrePy PTv3 或兼容点云编码器。
    """

    def __init__(
        self,
        image_dim: int = 768,
        hidden_dim: int = 512,
        point_dim: int = 64,
        num_heads: int = 8,
        frequencies: int = 6,
        margin_points: int = 128,
        view_tokens: int = 16,
        jaw_tokens: int = 32,
        point_encoder: nn.Module | None = None,
    ):
        super().__init__()
        if min(image_dim, hidden_dim, point_dim, num_heads, frequencies,
               margin_points, view_tokens, jaw_tokens) <= 0:
            raise ValueError("维度、频率数及 token 数必须为正数")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")

        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.margin_points = margin_points
        self.point_encoder = point_encoder

        # 三视图共享图像投影和重采样权重，但每个视角保留独立的局部 tokens。
        self.view_projection = MLP(width=image_dim, output_width=hidden_dim)
        self.view_embedding = nn.Embedding(3, hidden_dim)
        self.view_queries = nn.Parameter(torch.randn(1, view_tokens, hidden_dim) * 0.02)
        self.view_resampler = ResidualCrossAttentionBlock(hidden_dim, num_heads)
        self.view_fusion = ResidualAttentionBlock(
            n_ctx=3 * view_tokens, width=hidden_dim, heads=num_heads,
            norm_method="LayerNorm", atten_method="SDPA",
        )

        # 目标牙位是强条件：除了参与 tokens，还直接以残差注入全局条件。
        self.jaw_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(17, hidden_dim)
        self.tooth_type_embedding = nn.Embedding(4, hidden_dim)
        self.side_embedding = nn.Embedding(2, hidden_dim)

        self.size_fourier = FourierEmbedder(
            num_freqs=frequencies, input_dim=3, include_input=True, include_pi=True
        )
        self.size_encoder = MLP(width=self.size_fourier.out_dim, output_width=hidden_dim)
        self.margin_fourier = FourierEmbedder(
            num_freqs=frequencies, input_dim=3, include_input=True, include_pi=True
        )
        self.margin_projection = MLP(
            width=self.margin_fourier.out_dim + 4, output_width=hidden_dim
        )
        self.margin_fusion = ResidualAttentionBlock(
            n_ctx=margin_points, width=hidden_dim, heads=num_heads,
            norm_method="LayerNorm", atten_method="SDPA",
        )

        # 整颌 PTv3 输出逐点特征，逐点标签在 PTv3 后注入。
        self.point_projection = MLP(width=point_dim, output_width=hidden_dim)
        self.jaw_position_embedding = nn.Embedding(17, hidden_dim)
        # 0=未标注，1..31=沿牙弓的有向位置差(-15..+15)。
        # 11/21 和 31/41 是牙弓中线相邻两牙，不能用原始 1..16 索引相减。
        self.relative_position_embedding = nn.Embedding(32, hidden_dim)
        self.jaw_queries = nn.Parameter(torch.randn(1, jaw_tokens, hidden_dim) * 0.02)
        self.jaw_resampler = ResidualCrossAttentionBlock(hidden_dim, num_heads)
        self.global_projection = MLP(width=5 * hidden_dim, output_width=hidden_dim)

    def forward(
        self,
        view_features: Tensor,
        jaw: Tensor,
        target_position: Tensor,
        margin_line: Tensor,
        size: Tensor,
        jaw_points: Tensor | None = None,
        jaw_labels: Tensor | None = None,
        jaw_present: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """将真实牙颌条件编码为可融合的 tokens 和全局特征。

        Args:
            view_features: [B,3,T,C]，固定颊侧、舌侧、咬合视图顺序。
            jaw: [B]，0 上颌、1 下颌。
            target_position: [B]，按同颌 1～16 编码的目标牙位。
            margin_line: [B,M,3]，当前局部毫米坐标的颈缘线。
            size: [B,3]，分别表示近远中宽、颊舌厚、高度，毫米。
            jaw_points: 可选同颌 [B,N,6] 点云及法线。
            jaw_labels: 可选逐点牙位标签，0 背景、1～16 同颌牙。
            jaw_present: 可选 [B] 的邻牙是否存在掩码。

        Returns:
            具有 tokens、mask、global_condition 的多条件特征字典。

        Raises:
            ValueError: 输入维度、FDI 或真实尺寸非法。
        """
        if view_features.ndim != 4 or view_features.shape[1] != 3 or view_features.shape[2] < 1:
            raise ValueError("view_features 必须是 [B,3,T,C] 且 T>0")
        batch_size = view_features.shape[0]
        if view_features.shape[-1] != self.image_dim or batch_size < 1:
            raise ValueError("图像特征维度或 batch 大小不正确")
        if jaw.shape != (batch_size,) or target_position.shape != (batch_size,):
            raise ValueError("jaw 与 target_position 必须是 [B]")
        if margin_line.shape != (batch_size, self.margin_points, 3):
            raise ValueError(f"margin_line 必须是 [B,{self.margin_points},3]")
        if size.shape != (batch_size, 3):
            raise ValueError("size 必须是 [B,3] (近远中宽、颊舌厚、高度)")
        if not torch.isfinite(size).all() or (size <= 0).any():
            raise ValueError("size 必须为正的有限毫米值")
        if not torch.isfinite(margin_line).all() or not torch.isfinite(view_features).all():
            raise ValueError("视图和颈缘线不允许包含 NaN/Inf")
        if ((jaw < 0) | (jaw > 1)).any() or ((target_position < 1) | (target_position > 16)).any():
            raise ValueError("jaw 必须为 0/1，target_position 必须为 1~16")

        jaw = jaw.long()
        target_position = target_position.long()
        view_count, token_count = view_features.shape[1:3]
        view_ids = torch.arange(view_count, device=view_features.device)
        projected = self.view_projection(view_features)
        projected = projected + self.view_embedding(view_ids)[None, :, None, :]
        projected = projected.reshape(batch_size * view_count, token_count, self.hidden_dim)
        queries = self.view_queries.expand(batch_size * view_count, -1, -1)
        view_tokens = self.view_resampler(queries, projected)
        views = view_tokens.reshape(batch_size, -1, self.hidden_dim)
        views = self.view_fusion(views)

        ordinal = (target_position - 1) % 8 + 1
        tooth_type = torch.where(
            ordinal <= 2, 0,
            torch.where(ordinal == 3, 1, torch.where(ordinal <= 5, 2, 3)),
        )
        # 上颌位置 9~16 为左侧；下颌位置 1~8 才是左侧。
        is_left = (target_position > 8) ^ (jaw == 1)
        target_token = (
            self.jaw_embedding(jaw)
            + self.position_embedding(target_position)
            + self.tooth_type_embedding(tooth_type)
            + self.side_embedding(is_left.long())
        )
        size_token = self.size_encoder(self.size_fourier(size / 10.0))

        previous = torch.roll(margin_line, 1, dims=1)
        following = torch.roll(margin_line, -1, dims=1)
        tangent = F.normalize(following - previous, dim=-1, eps=1e-8)
        segment = torch.linalg.vector_norm(following - margin_line, dim=-1)
        arc = (segment.cumsum(1) - segment) / segment.sum(1, keepdim=True).clamp_min(1e-6)
        margin_features = torch.cat(
            (self.margin_fourier(margin_line / 10.0), tangent, arc.unsqueeze(-1)), dim=-1
        )
        margin_tokens = self.margin_fusion(self.margin_projection(margin_features))

        jaw_count = self.jaw_queries.shape[1]
        jaw_context = view_features.new_zeros(batch_size, jaw_count, self.hidden_dim)
        jaw_token_mask = torch.zeros(
            batch_size, jaw_count, dtype=torch.bool, device=view_features.device
        )
        if jaw_points is not None:
            if jaw_points.ndim != 3 or jaw_points.shape[0] != batch_size or jaw_points.shape[-1] != 6:
                raise ValueError("jaw_points 必须是 [B,N,6] (XYZ + Normal)")
            if jaw_labels is None or jaw_labels.shape != jaw_points.shape[:2]:
                raise ValueError("jaw_labels 必须是与牙弓点云对应的 [B,N]")
            if jaw_present is None:
                jaw_present = torch.ones(batch_size, device=jaw_points.device, dtype=torch.bool)
            if jaw_present.shape != (batch_size,):
                raise ValueError("jaw_present 必须是 [B]")
            present = jaw_present.bool()
            if present.any():
                if self.point_encoder is None:
                    raise ValueError("有实际牙弓时必须传入 SindrePy PTv3 point_encoder")
                if jaw_points.shape[1] < 1 or not torch.isfinite(jaw_points[present]).all():
                    raise ValueError("有效牙弓点云必须非空且数值有限")
                labels = jaw_labels[present].long()
                if ((labels < 0) | (labels > 16)).any():
                    raise ValueError("jaw_labels 仅允许 0~16")
                points = jaw_points[present]
                # PTv3 默认 grid_size=0.01，使用 10mm 归一化减少无意义的稀疏体素跨度。
                # 仅缩放 XYZ，法向保持单位长度；颈缘线和实际尺寸仍以毫米传递。
                normalized = points.clone()
                normalized[..., :3] = normalized[..., :3] / 10.0
                encoded = self.point_encoder(normalized, normalized[..., :3])
                features = getattr(encoded, "feat", encoded)
                if features.ndim == 2:
                    if features.shape[0] != points.shape[0] * points.shape[1]:
                        raise ValueError("PTv3 必须输出逐点特征，不能使用 cls_mode=True")
                    features = features.reshape(points.shape[0], points.shape[1], -1)
                if features.shape[:2] != points.shape[:2]:
                    raise ValueError("PTv3 必须返回与输入点一一对应的特征")
                if features.shape[-1] != self.point_projection.width:
                    raise ValueError("point_dim 与 PTv3 输出通道数不匹配")
                projected_points = self.point_projection(features)
                projected_points = projected_points + self.jaw_position_embedding(labels)
                # 每象限 1(中切牙)..8(第三磨牙)，合成为牙弓连续的 0..15。
                # 序号 1..8 -> 7..0；9..16 -> 8..15，两侧中切牙紧邻。
                tooth_arch = torch.where(labels <= 8, 8 - labels, labels - 1)
                target = target_position[present, None]
                target_arch = torch.where(target <= 8, 8 - target, target - 1)
                relative = torch.where(
                    labels > 0,
                    tooth_arch - target_arch + 16,
                    torch.zeros_like(labels),
                ).clamp(0, 31)
                projected_points = projected_points + self.relative_position_embedding(relative)
                queries = self.jaw_queries.expand(points.shape[0], -1, -1)
                jaw_context[present] = self.jaw_resampler(queries, projected_points)
                jaw_token_mask[present] = True
        elif jaw_labels is not None or jaw_present is not None:
            raise ValueError("jaw_labels / jaw_present 不允许脱离 jaw_points 单独传入")

        weight = jaw_token_mask.unsqueeze(-1)
        jaw_global = (jaw_context * weight).sum(1)
        jaw_global = jaw_global / weight.sum(1).clamp_min(1)
        global_condition = self.global_projection(torch.cat(
            (views.mean(1), target_token, size_token, margin_tokens.mean(1), jaw_global),
            dim=-1,
        )) + target_token

        tokens = torch.cat(
            (views, target_token[:, None], size_token[:, None], margin_tokens, jaw_context),
            dim=1,
        )
        fixed_mask = torch.ones(
            batch_size, views.shape[1] + 2 + margin_tokens.shape[1],
            dtype=torch.bool, device=view_features.device,
        )
        return {
            "tokens": tokens,
            "mask": torch.cat((fixed_mask, jaw_token_mask), dim=1),
            "global_condition": global_condition,
        }


@dataclass(frozen=True)
class LightFDGConfig:
    """轻量 FDG 网络和空间分辨率参数。

    Attributes:
        resolution: 细 FDG 网格每轴分辨率。
        coarse_resolution: 粗结构网格每轴分辨率。
        hidden_dim: 条件编码维度。
        base_channels: 粗结构卷积通道数。
        query_hidden_dim: 细体素查询头通道数。
        max_coarse_cells: 推理最多保留的粗体素数。
        max_output_cells: 推理最多保留的细 FDG 体素数。
        structure_threshold: 粗体素概率筛选阈值。
        fine_threshold: 细体素概率筛选阈值。
        padding_ratio: 颈缘线与尺寸确定的局部空间留白比例。
    """

    resolution: int = 128
    coarse_resolution: int = 32
    hidden_dim: int = 128
    base_channels: int = 24
    query_hidden_dim: int = 128
    max_coarse_cells: int = 512
    max_output_cells: int = 16000
    structure_threshold: float = 0.50
    fine_threshold: float = 0.50
    padding_ratio: float = 0.08

    def __post_init__(self):
        if (self.coarse_resolution < 4 or self.resolution < self.coarse_resolution
                or self.resolution % self.coarse_resolution):
            raise ValueError("FDG resolution 必须是 coarse_resolution 的整数倍")
        if self.hidden_dim <= 0 or self.base_channels < 8 or self.query_hidden_dim < 8:
            raise ValueError("网络通道数非法")
        if min(self.max_coarse_cells, self.max_output_cells) <= 0:
            raise ValueError("最大体素数量必须为正")
        if not 0 < self.structure_threshold < 1 or not 0 < self.fine_threshold < 1:
            raise ValueError("分类阈值必须在 (0,1)")
        if not 0 < self.padding_ratio < 0.4:
            raise ValueError("padding_ratio 必须在 (0,0.4)")


class LightFDGModel(nn.Module):
    """根据牙位、颈缘线、长宽高和邻牙预测开放冠 FDG。

    所有空间位置都显式使用 `size[MD,BL,H]` 的独立物理尺度。
    推理结构来自预测结果，不消费真实冠坐标或真实冠尺寸。

    Args:
        condition_encoder: SindrePy 组件构成的多条件编码器。
        config: 轻量 FDG 网络与网格分辨率配置。
    """

    def __init__(
        self,
        condition_encoder: MCDGConditionEncoder,
        config: LightFDGConfig | None = None,
    ):
        super().__init__()
        self.config = config or LightFDGConfig()
        if condition_encoder.hidden_dim != self.config.hidden_dim:
            raise ValueError("MCDG hidden_dim 与 FDG hidden_dim 必须匹配")
        self.condition_encoder = condition_encoder
        ch = self.config.base_channels
        size = self.config.coarse_resolution
        t = torch.linspace(-1 + 1 / size, 1 - 1 / size, size)
        xx, yy, zz = torch.meshgrid(t, t, t, indexing="ij")
        self.register_buffer(
            "grid", torch.stack((xx, yy, zz))[None], persistent=False,
        )
        # 浅层空间残差主干。不是 TRELLIS 官方 SparseUnetVaeDecoder；
        # 使用 PyTorch 原生高效 3D conv，避免在单冠任务重装重型 sparse 框架。
        self.grid_stem = nn.Conv3d(6, ch, 3, padding=1)
        self.film = nn.Linear(self.config.hidden_dim, ch * 2)
        self.backbone = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(4 if ch % 4 == 0 else 1, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(4 if ch % 4 == 0 else 1, ch),
            nn.SiLU(),
        )
        self.coarse_head = nn.Conv3d(ch, 1, kernel_size=1)
        self.local_head = nn.Sequential(
            nn.Linear(ch + self.config.hidden_dim + 9, self.config.query_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.query_hidden_dim, self.config.query_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.query_hidden_dim, 7),
        )
        # 输出 0=是否有 FDG 体素，1..3=双顶点偏移，4..6=相交标志
        # 不重复实现 FDG mesh extraction。

    def encode(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """提取条件和依赖真实 MD/BL/H 的体素特征。

        Args:
            batch: 含 view_features、jaw、target_position、margin_line、
                size [B,3] 和可选邻牙点云的条件批次。

        Returns:
            粗结构 logits、空间体素特征、全局条件及毫米尺寸规范化系数。

        Raises:
            ValueError: 缺少预计算的三视图 DINOv3 特征。
        """
        keys = (
            "view_features", "jaw", "target_position", "margin_line", "size",
            "jaw_points", "jaw_labels", "jaw_present",
        )
        if "view_features" not in batch:
            raise ValueError("训练需要预处理 DINOv3 view_features；不在训练步骤渲染")
        conditions = self.condition_encoder(**{
            key: batch[key] for key in keys if key in batch
        })
        global_feature = conditions["global_condition"]
        # 只用输入的颈缘线和指定 MD/BL/H 计算规范空间，绝不读取设计冠。
        size = batch["size"].float()
        margin = batch["margin_line"].float()
        margin_span = (margin.amax(dim=1) - margin.amin(dim=1)).amax(dim=-1)
        extent = torch.maximum(size.amax(dim=-1), margin_span)
        scale_mm = (2 * extent / (1 - 2 * self.config.padding_ratio))
        vol = self.grid.expand(global_feature.shape[0], -1, -1, -1, -1)
        # 局部物理位置 / 目标尺寸，直接给结构网络；三个独立尺寸均有空间含义。
        relative_xyz = vol * (scale_mm[:, None, None, None, None] * 0.5)
        relative_xyz = relative_xyz / size[:, :, None, None, None].clamp_min(1e-3)
        volume = self.grid_stem(torch.cat((vol, relative_xyz), dim=1))
        gamma, beta = self.film(global_feature).chunk(2, dim=-1)
        volume = volume * (1 + 0.1 * torch.tanh(gamma)[..., None, None, None])
        volume = volume + beta[..., None, None, None]
        volume = volume + self.backbone(volume)
        coarse_logits = self.coarse_head(volume)[:, 0]
        return {
            "volume": volume, "global": global_feature,
            "coarse_logits": coarse_logits, "size": size, "scale_mm": scale_mm,
        }

    def predict_at_coords(
        self, cache: dict[str, Tensor], coords: Tensor
    ) -> dict[str, Tensor]:
        """在任意指定细体素位置预测 FDG 几何与存在概率。

        Args:
            cache: encode() 返回的粗结构特征与尺寸规范化数据。
            coords: [B,N,3] 的整数体素位置，不要求来自 GT。

        Returns:
            fine_logits、dual_offsets 和 intersected_logits。
        """
        c = self.config
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords 必须为 [B,N,3]")
        if coords.shape[0] != cache["volume"].shape[0]:
            raise ValueError("coords 与 batch 大小不一致")
        # 训练坐标已在 FDGDataset 做过严格校验，避免每步 GPU bool 同步。
        # 推理从 generate_coords 构建，坐标范围天然可控。
        if not self.training and bool(((coords < 0) | (coords >= c.resolution)).any()):
            raise ValueError("坐标超出 FDG 分辨率")
        coords = coords.long()
        stride = c.resolution // c.coarse_resolution
        parent = coords // stride
        batch = torch.arange(
            len(coords), device=coords.device,
        )[:, None]
        feat = cache["volume"][
            batch, :, parent[..., 0], parent[..., 1], parent[..., 2],
        ]
        xyz = (coords.float() + 0.5) / c.resolution * 2 - 1
        local = ((coords % stride).float() + 0.5) / stride * 2 - 1
        global_feature = cache["global"][:, None].expand(-1, coords.shape[1], -1)
        position_mm = xyz * cache["scale_mm"][:, None, None] * 0.5
        scaled_position = position_mm / cache["size"][:, None, :].clamp_min(1e-3)
        raw = self.local_head(torch.cat(
            (feat, global_feature, xyz, local, scaled_position), dim=-1,
        ))
        return {
            "fine_logits": raw[..., 0],
            # 对齐 TRELLIS.2 FlexiDualGridVaeDecoder voxel_margin=0.5：
            # [-0.5,1.5] 可容纳 QEF 在锐边/开放边界的少量体素外延。
            "dual_offsets": torch.sigmoid(raw[..., 1:4]) * 2.0 - 0.5,
            "intersected_logits": raw[..., 4:7],
        }

    def forward(
        self, batch: dict[str, Tensor], coords: Tensor,
    ) -> dict[str, Tensor]:
        """训练用前向；真实坐标只能用于决定监督查询位置。

        Args:
            batch: 仅含真实病例条件、不能把设计 Mesh 作为模型输入。
            coords: 采样的细体素查询坐标 [B,N,3]。

        Returns:
            粗结构、细结构以及 FDG 几何的预测字典。
        """
        cache = self.encode(batch)
        return {**cache, **self.predict_at_coords(cache, coords)}

    @torch.no_grad()
    def generate_coords(
        self, cache: dict[str, Tensor],
    ) -> list[Tensor]:
        """仅根据网络预测的粗/细体素 logits 构建推理坐标。

        Args:
            cache: encode() 返回的空间结构特征及尺寸条件。

        Returns:
            每个病例的预测稀疏体素坐标，可能为空。
        """
        c = self.config
        probs = cache["coarse_logits"].sigmoid().flatten(1)
        result = []
        stride = c.resolution // c.coarse_resolution
        children = torch.stack(torch.meshgrid(
            *[torch.arange(stride, device=probs.device)] * 3,
            indexing="ij",
        ), dim=-1).reshape(-1, 3)
        for i in range(len(probs)):
            selected = torch.flatnonzero(probs[i] >= c.structure_threshold)
            if not len(selected):
                # 防止随机初始化时完全空体素；以最大概率位置做诊断性回退。
                selected = probs[i].topk(1).indices
            if len(selected) > c.max_coarse_cells:
                selected = selected[probs[i, selected].topk(c.max_coarse_cells).indices]
            parent = torch.stack((
                selected // (c.coarse_resolution * c.coarse_resolution),
                selected // c.coarse_resolution % c.coarse_resolution,
                selected % c.coarse_resolution,
            ), dim=-1)
            coords = (parent[:, None] * stride + children[None]).reshape(-1, 3)
            # 子体素预测按块执行，仅在挑选时不用梯度；性能和内存可控。
            score_list = []
            for start in range(0, len(coords), 8192):
                part = self.predict_at_coords(
                    {"volume":cache["volume"][i:i+1],
                     "global":cache["global"][i:i+1],
                     "size":cache["size"][i:i+1],
                     "scale_mm":cache["scale_mm"][i:i+1]},
                    coords[None, start:start+8192],
                )["fine_logits"][0]
                score_list.append(part)
            score = torch.cat(score_list).sigmoid()
            fine = torch.flatnonzero(score >= c.fine_threshold)
            if len(fine) > c.max_output_cells:
                fine = fine[score[fine].topk(c.max_output_cells).indices]
            # 无有效细结构时显式返回空，调用方不能把空网格伪装为成功。
            result.append(coords[fine])
        return result

    def predict(
        self, batch: dict[str, Tensor],
    ) -> list[dict[str, Tensor]]:
        """无 GT 推理，生成每例 FDG 体素位置与几何属性。

        Args:
            batch: 仅包含真实病例条件，不包含 FDG 监督坐标。

        Returns:
            每个病例的 FDG coords、双顶点偏移、相交 logits 和有效标记。
        """
        cache = self.encode(batch)
        predicted = self.generate_coords(cache)
        result = []
        for i, coords in enumerate(predicted):
            if not len(coords):
                result.append({"coords":coords, "valid":False})
                continue
            one = self.predict_at_coords(
                {"volume":cache["volume"][i:i+1],
                 "global":cache["global"][i:i+1],
                     "size":cache["size"][i:i+1],
                     "scale_mm":cache["scale_mm"][i:i+1]},
                coords[None],
            )
            result.append({
                "coords": coords,
                "dual_offsets": one["dual_offsets"][0],
                "intersected_logits": one["intersected_logits"][0],
                "valid":True,
            })
        return result


def create_light_fdg_model(
    *,
    image_dim: int,
    margin_points: int = 128,
    config: LightFDGConfig | None = None,
    point_encoder: nn.Module | None = None,
) -> LightFDGModel:
    """创建复用 SindrePy PTv3 的轻量冠体网络。

    Args:
        image_dim: DINOv3 特征最后一维通道数。
        margin_points: 颈缘线固定采样点数。
        config: FDG 结构及轻量化配置。
        point_encoder: 可选外部点云编码器；未指定时创建轻量 PTv3。

    Returns:
        用于训练或推理的 LightFDGModel。
    """
    config = config or LightFDGConfig()
    if point_encoder is None:
        from sindre.ai.vision3d.pointcloud_utils.point_transformerV3 import (
            PointTransformerV3,
        )
        point_encoder = PointTransformerV3(
            in_channels=6,
            stride=(2, 2),
            enc_depths=(1, 1, 1),
            enc_channels=(32, 64, 96),
            enc_num_head=(2, 4, 6),
            enc_patch_size=(128, 128, 128),
            dec_depths=(1, 1),
            dec_channels=(32, 64),
            dec_num_head=(2, 4),
            dec_patch_size=(128, 128),
            drop_path=0.05, cls_mode=False,
            enable_flash=False, shuffle_orders=True,
        )
    encoder = MCDGConditionEncoder(
        image_dim=image_dim,
        hidden_dim=config.hidden_dim, point_dim=32,
        num_heads=4, margin_points=margin_points,
        view_tokens=4, jaw_tokens=8,
        point_encoder=point_encoder,
    )
    return LightFDGModel(encoder, config)
