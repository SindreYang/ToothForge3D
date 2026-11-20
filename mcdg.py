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
        self.relative_position_embedding = nn.Embedding(17, hidden_dim)
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
            if self.point_encoder is None:
                raise ValueError("有牙弓点云时必须传入 SindrePy PTv3 point_encoder")
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
                if jaw_points.shape[1] < 1 or not torch.isfinite(jaw_points[present]).all():
                    raise ValueError("有效牙弓点云必须非空且数值有限")
                labels = jaw_labels[present].long()
                if ((labels < 0) | (labels > 16)).any():
                    raise ValueError("jaw_labels 仅允许 0~16")
                points = jaw_points[present]
                # SindrePy PTv3 forward(x,p) 返回 Point.feat，正常为 [B*N,D]。
                encoded = self.point_encoder(points, points[..., :3])
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
                relative = (labels - target_position[present, None]).abs().clamp(max=16)
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


class TrellisFlowAdapter(nn.Module):
    """把 MCDG 条件接入官方 TRELLIS.2 稠密或稀疏 Flow block。"""

    def __init__(self, flow_model: nn.Module, condition_dim: int):
        super().__init__()
        self.flow_model = flow_model
        self.condition_projection = MLP(
            width=condition_dim, output_width=flow_model.cond_channels
        )
        self.global_projection = MLP(
            width=condition_dim, output_width=flow_model.model_channels
        )

    def _dense(
        self, x: Tensor, timestep: Tensor, tokens: Tensor, global_condition: Tensor
    ) -> Tensor:
        from trellis2.modules.utils import manual_cast

        flow = self.flow_model
        batch_size, channels, resolution = x.shape[0], x.shape[1], x.shape[-1]
        if x.shape[2:] != (resolution, resolution, resolution):
            raise ValueError("稠密 Flow 的三个空间维度必须相同")
        h = x.reshape(batch_size, channels, -1).permute(0, 2, 1).contiguous()
        h = flow.input_layer(h)
        if flow.pe_mode == "ape":
            h = h + flow.pos_emb[None]
        mod = flow.t_embedder(timestep) + global_condition
        if flow.share_mod:
            mod = flow.adaLN_modulation(mod)
        mod = manual_cast(mod, flow.dtype)
        h = manual_cast(h, flow.dtype)
        tokens = manual_cast(tokens, flow.dtype)
        for block in flow.blocks:
            h = block(h, mod, tokens, getattr(flow, "rope_phases", None))
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = flow.out_layer(h)
        return h.permute(0, 2, 1).reshape(
            batch_size, flow.out_channels, resolution, resolution, resolution
        ).contiguous()

    def _sparse(
        self, x: object, timestep: Tensor, tokens: Tensor, mask: Tensor,
        global_condition: Tensor,
    ) -> object:
        from trellis2.modules import sparse as sp
        from trellis2.modules.utils import manual_cast

        flow = self.flow_model
        context = sp.VarLenTensor.from_tensor_list(
            [tokens[index][mask[index]] for index in range(tokens.shape[0])]
        )
        h = manual_cast(flow.input_layer(x), flow.dtype)
        mod = flow.t_embedder(timestep) + global_condition
        if flow.share_mod:
            mod = flow.adaLN_modulation(mod)
        mod = manual_cast(mod, flow.dtype)
        context = manual_cast(context, flow.dtype)
        if flow.pe_mode == "ape":
            position = flow.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(position, flow.dtype)
        for block in flow.blocks:
            h = block(h, mod, context)
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return flow.out_layer(h)

    def forward(self, x: Tensor | object, timestep: Tensor, conditions: dict[str, Tensor]) -> Tensor | object:
        tokens = self.condition_projection(conditions["tokens"])
        mask = conditions["mask"].bool()
        global_condition = self.global_projection(conditions["global_condition"])
        if hasattr(x, "coords") and hasattr(x, "feats"):
            return self._sparse(x, timestep, tokens, mask, global_condition)
        if not isinstance(x, Tensor) or x.ndim != 5:
            raise ValueError("稠密 TRELLIS.2 Flow 输入必须是 [B, C, R, R, R]")
        if mask.all():
            return self._dense(x, timestep, tokens, global_condition)
        return torch.cat([
            self._dense(
                x[index:index + 1], timestep[index:index + 1],
                tokens[index][mask[index]].unsqueeze(0),
                global_condition[index:index + 1],
            )
            for index in range(x.shape[0])
        ], dim=0)
