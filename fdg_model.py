"""牙冠专用轻量 FDG 网络：条件 -> 粗空间结构 -> 细体素 -> FDG。

训练时允许在 GT 体素上回归几何，但结构分支必须独立监督；
推理的体素坐标只来自预测的粗结构/细结构，不读取目标冠 FDG coords。
不重复实现 O-Voxel CUDA 网格重建或 TRELLIS.2 巨型 VAE/Flow。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mcdg import MCDGConditionEncoder


@dataclass(frozen=True)
class LightFDGConfig:
    resolution: int = 128
    coarse_resolution: int = 32
    hidden_dim: int = 128
    base_channels: int = 24
    query_hidden_dim: int = 128
    max_coarse_cells: int = 512
    max_output_cells: int = 16000
    structure_threshold: float = 0.50
    fine_threshold: float = 0.50

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


class LightFDGModel(nn.Module):
    """以低分辨率 3D 卷积预测支架，再以小型坐标头预测稀疏 FDG。

    整个模型不从 GT fdg_coords 构造输入支架；训练传入 coords 只决定
    需要监督哪些查询位置。推理坐标必须走 generate_coords()。
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
        self.grid_stem = nn.Conv3d(3, ch, 3, padding=1)
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
            nn.Linear(ch + self.config.hidden_dim + 6, self.config.query_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.query_hidden_dim, self.config.query_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.query_hidden_dim, 7),
        )
        # 输出 0=是否有 FDG 体素，1..3=双顶点偏移，4..6=相交标志
        # 不重复实现 FDG mesh extraction。

    def encode(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
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
        vol = self.grid.expand(global_feature.shape[0], -1, -1, -1, -1)
        volume = self.grid_stem(vol)
        gamma, beta = self.film(global_feature).chunk(2, dim=-1)
        volume = volume * (1 + 0.1 * torch.tanh(gamma)[..., None, None, None])
        volume = volume + beta[..., None, None, None]
        volume = volume + self.backbone(volume)
        coarse_logits = self.coarse_head(volume)[:, 0]
        return {
            "volume": volume, "global": global_feature,
            "coarse_logits": coarse_logits,
        }

    def predict_at_coords(
        self, cache: dict[str, Tensor], coords: Tensor
    ) -> dict[str, Tensor]:
        """coords [B,N,3] 是查询位置，不要求来自 Ground Truth。"""
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
        raw = self.local_head(torch.cat(
            (feat, global_feature, xyz, local), dim=-1,
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
        cache = self.encode(batch)
        return {**cache, **self.predict_at_coords(cache, coords)}

    @torch.no_grad()
    def generate_coords(
        self, cache: dict[str, Tensor],
    ) -> list[Tensor]:
        """仅用网络预测生成真实的稀疏坐标；从不读取 FDG GT。"""
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
                     "global":cache["global"][i:i+1]},
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
        """推理返回独立病例的预测 FDG 坐标和特征，供 CrownFDG.get_mesh。"""
        cache = self.encode(batch)
        predicted = self.generate_coords(cache)
        result = []
        for i, coords in enumerate(predicted):
            if not len(coords):
                result.append({"coords":coords, "valid":False})
                continue
            one = self.predict_at_coords(
                {"volume":cache["volume"][i:i+1],
                 "global":cache["global"][i:i+1]},
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
    """直接复用 SindrePy PTv3，使用轻量配置而非复制 PTv3 实现。"""
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
