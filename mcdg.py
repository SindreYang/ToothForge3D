"""ToothForge3D：基于 SindrePy 的多条件编码器和 TRELLIS.2 Flow 适配器。

仅负责条件编码和 Flow 速度预测；不把尚未实现的网格生成、颈缘线拟合
或训练管线伪装成可直接调用的完整牙冠生成器。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sindre.ai.attention_utils.cross_attention import ResidualCrossAttentionBlock
from sindre.ai.attention_utils.multihead_attention import ResidualAttentionBlock
from sindre.ai.embedder import FourierEmbedder
from sindre.ai.layers import MLP


class MCDGConditionEncoder(nn.Module):
    """三视图 + 目标牙位 + 颈缘线 + 尺寸 + 可选同颌牙弓条件。

    牙位与 SindrePy convert_fdi2idx 保持一致：
    上颌 11~18 -> 1~8、21~28 -> 9~16；
    下颌 31~38 -> 1~8、41~48 -> 9~16。
    不使用对颌；牙弓几何按整颌点云一次编码，不要求逐颗分割。
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
        """返回 tokens、有效 token mask 及全局条件。

        view_features: [B,3,T,C]，顺序固定为颊、舌、咬合。
        jaw_points: [B,N,6] (XYZ + Normal)，可省略；不包含对颌。
        jaw_labels: [B,N]，0 为牙龈/未知、1~16 为同颌牙位索引。
        jaw_present: [B]，允许一个 batch 内部分病例没有任何同颌牙。
        所有几何均在同一坐标系中，尺寸与坐标单位为 mm。
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


class CrownImplicitDecoder(nn.Module):
    """旧私有冠生成项目的隐式查询路径，复用 SindrePy Fourier/CrossAttention。

    输入查询点与颈缘线、同颌点云处于同一局部 mm 空间。
    输出原始 Occupancy logits 和实际毫米 SDF；输出不是完整 mesh。
    对查询点分块避免 [B,32768,hidden_dim] 的高显存占用。
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        num_heads: int = 8,
        query_frequencies: int = 6,
        query_chunk_size: int = 1024,
    ):
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads or query_chunk_size <= 0:
            raise ValueError("hidden_dim 必须可整除 num_heads，chunk_size 必须正数")
        self.query_chunk_size = query_chunk_size
        self.query_fourier = FourierEmbedder(
            num_freqs=query_frequencies, input_dim=3,
            include_input=True, include_pi=True,
        )
        self.query_projection = MLP(
            width=self.query_fourier.out_dim, output_width=hidden_dim,
        )
        self.attention = ResidualCrossAttentionBlock(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 2)

    def forward(
        self, query_points: Tensor, conditions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if query_points.ndim != 3 or query_points.shape[-1] != 3 or not query_points.shape[1]:
            raise ValueError("query_points 应为非空的 [B,Q,3]")
        if not torch.isfinite(query_points).all():
            raise ValueError("查询点含有 NaN/Inf")
        tokens, mask = conditions["tokens"], conditions["mask"]
        global_condition = conditions["global_condition"]
        batch, count = query_points.shape[:2]
        if tokens.shape[:2] != mask.shape or tokens.shape[0] != batch:
            raise ValueError("conditions 的 tokens/mask 批次和长度不一致")
        if global_condition.shape != (batch, tokens.shape[-1]):
            raise ValueError("global_condition 维度不匹配")
        if tokens.device != query_points.device:
            raise ValueError("queries 和 condition tokens 必须在同一设备")
        predictions = []
        for i in range(batch):
            context = torch.cat(
                [tokens[i:i + 1, mask[i]], global_condition[i:i + 1, None]],
                dim=1,
            )
            chunks = []
            for j in range(0, count, self.query_chunk_size):
                query = query_points[i:i + 1, j:j + self.query_chunk_size]
                query_embedding = self.query_projection(
                    self.query_fourier(query / 10.0),
                )
                feature = self.attention(query_embedding, context)
                chunks.append(self.output_projection(self.norm(feature)))
            predictions.append(torch.cat(chunks, dim=1))
        combined = torch.cat(predictions, dim=0)
        return {
            "occupancy_logits": combined[..., 0],
            "sdf_mm": combined[..., 1],
        }


class MCDGReconstructionModel(nn.Module):
    """已可计算 BCE+SDF 的一阶段重建基线；与 TRELLIS Flow 两阶段分开。"""

    def __init__(
        self,
        condition_encoder: MCDGConditionEncoder,
        query_decoder: CrownImplicitDecoder,
    ):
        super().__init__()
        if condition_encoder.hidden_dim != query_decoder.output_projection.in_features:
            raise ValueError("条件编码器与查询解码器 hidden_dim 不一致")
        self.condition_encoder = condition_encoder
        self.query_decoder = query_decoder

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if "view_features" not in batch:
            raise ValueError(
                "训练必须提供预计算 view_features；请用 SindrePy DINOv3Encoder "
                "在 build_database(image_encoder=...) 时提取，不要假定 RGB 是 patch tokens"
            )
        keys = (
            "view_features", "jaw", "target_position", "margin_line", "size",
            "jaw_points", "jaw_labels", "jaw_present",
        )
        conditions = self.condition_encoder(
            **{key: batch[key] for key in keys if key in batch}
        )
        return self.query_decoder(batch["query_points"], conditions)


class TrellisFlowAdapter(nn.Module):
    """只转换多模态条件；Flow 前向直接委托官方 TRELLIS.2。

    之前复制了官方 dense/sparse Flow 内部 block 循环，会随着官方实现更新
    出现维护风险。改为调用 flow_model(x,t,cond)，保留其 RoPE、稀疏布局、
    AdaLN 和 dtype 处理。目标 FDI 的强全局条件作为附加 cross-attn token。
    """

    def __init__(self, flow_model: nn.Module, condition_dim: int):
        super().__init__()
        if not hasattr(flow_model, "cond_channels"):
            raise TypeError("需传入官方 TRELLIS.2 Flow Model（缺 cond_channels）")
        self.flow_model = flow_model
        self.condition_projection = MLP(
            width=condition_dim, output_width=flow_model.cond_channels,
        )
        self.global_projection = MLP(
            width=condition_dim, output_width=flow_model.cond_channels,
        )

    def forward(
        self, x: Tensor | object, timestep: Tensor,
        conditions: dict[str, Tensor],
    ) -> Tensor | object:
        tokens = self.condition_projection(conditions["tokens"])
        global_token = self.global_projection(conditions["global_condition"])[:, None]
        mask = conditions["mask"].bool()
        if mask.shape != tokens.shape[:2] or global_token.shape[0] != tokens.shape[0]:
            raise ValueError("条件 tokens / mask / global_condition 不一致")
        if timestep.ndim != 1 or timestep.shape[0] != tokens.shape[0]:
            raise ValueError("timestep 必须为 [B]，单位沿用 TRELLIS.2 的 t*1000")
        context = torch.cat([tokens, global_token], dim=1)
        mask = torch.cat(
            [mask, torch.ones_like(mask[:, :1])], dim=1,
        )
        if hasattr(x, "coords") and hasattr(x, "feats"):
            # 官方 SLatFlowModel.forward 原生接受 list[Tensor] 变长条件。
            if bool(mask.all()):
                return self.flow_model(x, timestep, context)
            return self.flow_model(
                x, timestep,
                [context[i, mask[i]] for i in range(context.shape[0])],
            )
        if not isinstance(x, Tensor) or x.ndim != 5:
            raise ValueError("Dense Flow 输入必须是 [B,C,R,R,R]")
        if x.shape[0] != tokens.shape[0]:
            raise ValueError("Dense Flow 批次与条件不匹配")
        if bool(mask.all()):
            return self.flow_model(x, timestep, context)
        # 官方 Dense Flow 只接受定长 context。仅当 mask 不等长时逐样本调用。
        return torch.cat([
            self.flow_model(
                x[i:i + 1], timestep[i:i + 1],
                context[i:i + 1, mask[i]],
            )
            for i in range(x.shape[0])
        ], dim=0)
