"""MCDG 条件编码与 TRELLIS.2 Flow 适配。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sindre.ai.attention_utils.cross_attention import ResidualCrossAttentionBlock
from sindre.ai.attention_utils.multihead_attention import ResidualAttentionBlock
from sindre.ai.embedder import FourierEmbedder
from sindre.ai.layers import MLP


class MCDGConditionEncoder(nn.Module):
    """用 SindrePy 编码三视图、牙位、尺寸、颈缘线和可选邻牙点云。"""

    def __init__(
        self,
        image_dim: int = 768,
        hidden_dim: int = 512,
        point_dim: int = 64,
        num_heads: int = 8,
        frequencies: int = 6,
        margin_points: int = 128,
        view_tokens: int = 16,
        neighbor_tokens_per_tooth: int = 16,
        point_encoder: nn.Module | None = None,
    ):
        super().__init__()
        if min(image_dim, hidden_dim, point_dim, num_heads, frequencies, margin_points,
               view_tokens, neighbor_tokens_per_tooth) <= 0:
            raise ValueError("网络维度、频率数和 token 数必须大于 0")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")

        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.margin_points = margin_points
        self.neighbor_tokens_per_tooth = neighbor_tokens_per_tooth
        self.point_encoder = point_encoder

        self.view_projection = MLP(width=image_dim, output_width=hidden_dim)
        self.view_embedding = nn.Embedding(3, hidden_dim)
        self.view_queries = nn.Parameter(torch.randn(1, view_tokens, hidden_dim) * 0.02)
        self.view_fusion = ResidualCrossAttentionBlock(hidden_dim, num_heads)

        self.jaw_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(17, hidden_dim)
        self.tooth_type_embedding = nn.Embedding(4, hidden_dim)
        self.side_embedding = nn.Embedding(2, hidden_dim)

        self.size_fourier = FourierEmbedder(
            num_freqs=frequencies, input_dim=3, include_input=False, include_pi=True
        )
        self.size_encoder = MLP(width=self.size_fourier.out_dim, output_width=hidden_dim)
        self.margin_fourier = FourierEmbedder(
            num_freqs=frequencies, input_dim=3, include_input=False, include_pi=True
        )
        self.margin_projection = MLP(
            width=self.margin_fourier.out_dim + 4, output_width=hidden_dim
        )
        self.margin_fusion = ResidualAttentionBlock(
            n_ctx=margin_points, width=hidden_dim, heads=num_heads,
            norm_method="LayerNorm", atten_method="SDPA",
        )

        self.point_projection = MLP(width=point_dim, output_width=hidden_dim)
        self.neighbor_position_embedding = nn.Embedding(17, hidden_dim)
        self.neighbor_queries = nn.Parameter(
            torch.randn(1, neighbor_tokens_per_tooth, hidden_dim) * 0.02
        )
        self.neighbor_fusion = ResidualCrossAttentionBlock(hidden_dim, num_heads)
        self.global_projection = MLP(width=hidden_dim * 5, output_width=hidden_dim)

    def forward(
        self,
        view_features: Tensor,
        jaw: Tensor,
        target_position: Tensor,
        margin_line: Tensor,
        size: Tensor,
        teeth_points: Tensor | None = None,
        teeth_positions: Tensor | None = None,
        teeth_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if view_features.ndim != 4 or view_features.shape[1] != 3:
            raise ValueError("view_features 必须是 [B, 3, T, C]，顺序为颊、舌、咬合")
        batch_size = view_features.shape[0]
        if view_features.shape[-1] != self.image_dim:
            raise ValueError(f"DINOv3 特征维度必须为 {self.image_dim}")
        if jaw.shape != (batch_size,) or target_position.shape != (batch_size,):
            raise ValueError("jaw 和 target_position 必须是 [B]")
        if margin_line.shape != (batch_size, self.margin_points, 3):
            raise ValueError(f"margin_line 必须是 [B, {self.margin_points}, 3]")
        if size.shape != (batch_size, 3) or not torch.isfinite(size).all() or torch.any(size <= 0):
            raise ValueError("size 必须是有限的正数 [B, 3] 毫米")
        if torch.any((jaw < 0) | (jaw > 1)):
            raise ValueError("jaw 仅支持 0（上颌）或 1（下颌）")
        if torch.any((target_position < 1) | (target_position > 16)):
            raise ValueError("target_position 必须在 1～16 之间")

        view_ids = torch.arange(3, device=view_features.device)
        views = self.view_projection(view_features) + self.view_embedding(view_ids)[None, :, None]
        views = self.view_fusion(
            self.view_queries.expand(batch_size, -1, -1), views.flatten(1, 2)
        )

        jaw = jaw.long()
        target_position = target_position.long()
        arch_position = (target_position - 1) % 8 + 1
        tooth_type = torch.where(
            arch_position <= 2,
            0,
            torch.where(arch_position == 3, 1, torch.where(arch_position <= 5, 2, 3)),
        )
        target = (
            self.jaw_embedding(jaw)
            + self.position_embedding(target_position)
            + self.tooth_type_embedding(tooth_type)
            + self.side_embedding((target_position > 8).long())
        )
        size_token = self.size_encoder(self.size_fourier(size / 10.0))

        following = torch.roll(margin_line, shifts=-1, dims=1)
        previous = torch.roll(margin_line, shifts=1, dims=1)
        tangent = F.normalize(following - previous, dim=-1)
        segment_length = torch.linalg.vector_norm(following - margin_line, dim=-1)
        arc_length = segment_length.cumsum(dim=1) - segment_length
        arc_length = arc_length / segment_length.sum(dim=1, keepdim=True).clamp_min(1e-6)
        margin_features = torch.cat(
            (self.margin_fourier(margin_line / 10.0), tangent, arc_length.unsqueeze(-1)), dim=-1
        )
        margin_tokens = self.margin_fusion(self.margin_projection(margin_features))

        neighbor_tokens = view_features.new_zeros(batch_size, 0, self.hidden_dim)
        neighbor_mask = torch.zeros(
            batch_size, 0, dtype=torch.bool, device=view_features.device
        )
        if teeth_points is not None:
            if self.point_encoder is None:
                raise ValueError("teeth_points 需要 SindrePy PointTransformerV3 point_encoder")
            if teeth_points.ndim != 4 or teeth_points.shape[0] != batch_size or teeth_points.shape[-1] != 6:
                raise ValueError("teeth_points 必须是 [B, M, N, 6]（XYZ + Normal）")
            if teeth_positions is None or teeth_mask is None:
                raise ValueError("teeth_points 需要同时提供 teeth_positions 和 teeth_mask")
            if teeth_positions.shape != teeth_points.shape[:2] or teeth_mask.shape != teeth_points.shape[:2]:
                raise ValueError("teeth_positions 和 teeth_mask 必须是 [B, M]")

            teeth_mask = teeth_mask.bool()
            teeth_positions = teeth_positions.long()
            if torch.any(teeth_mask & ((teeth_positions < 1) | (teeth_positions > 16))):
                raise ValueError("有效同颌牙位必须在 1～16 之间")
            tooth_count, point_count = teeth_mask.shape[1], teeth_points.shape[2]
            neighbor_mask = teeth_mask.unsqueeze(-1).expand(
                -1, -1, self.neighbor_tokens_per_tooth
            ).reshape(batch_size, -1)
            neighbor_tokens = view_features.new_zeros(
                batch_size, tooth_count, self.neighbor_tokens_per_tooth, self.hidden_dim
            )
            valid_teeth = teeth_mask.nonzero(as_tuple=True)
            if valid_teeth[0].numel():
                points = teeth_points[valid_teeth]
                encoded = self.point_encoder(points, points[..., :3])
                point_features = getattr(encoded, "feat", encoded)
                if point_features.ndim == 2:
                    point_features = point_features.reshape(len(points), point_count, -1)
                if point_features.shape[:2] != points.shape[:2]:
                    raise ValueError("PointTransformerV3 必须返回每个输入点对应的 .feat")
                point_features = self.point_projection(point_features)
                point_features = point_features + self.neighbor_position_embedding(
                    teeth_positions[valid_teeth]
                )[:, None]
                queries = self.neighbor_queries.expand(len(points), -1, -1)
                neighbor_tokens[valid_teeth] = self.neighbor_fusion(queries, point_features)
            neighbor_tokens = neighbor_tokens.flatten(1, 2)

        neighbor_weights = neighbor_mask.unsqueeze(-1).to(neighbor_tokens.dtype)
        neighbor_global = (neighbor_tokens * neighbor_weights).sum(dim=1)
        neighbor_global = neighbor_global / neighbor_weights.sum(dim=1).clamp_min(1.0)
        global_condition = self.global_projection(torch.cat(
            (target, size_token, views.mean(dim=1), margin_tokens.mean(dim=1), neighbor_global),
            dim=-1,
        ))

        tokens = torch.cat(
            (views, target[:, None], size_token[:, None], margin_tokens, neighbor_tokens), dim=1
        )
        fixed_mask = torch.ones(
            batch_size,
            views.shape[1] + 2 + margin_tokens.shape[1],
            dtype=torch.bool,
            device=view_features.device,
        )
        return {
            "tokens": tokens,
            "mask": torch.cat((fixed_mask, neighbor_mask), dim=1),
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
