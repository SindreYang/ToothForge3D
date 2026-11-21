"""牙冠几何训练目标：SindrePy Occupancy + 毫米 SDF，及官方 Flow Matching。

不在本项目重复实现 BCE/focal/dice/chamfer；占据分类直接使用
sindre.ai.losses.balance_bce_loss。Flow 损失严格采用 TRELLIS.2 的
x_t=(1-t)*x_0+(sigma_min+(1-sigma_min)*t)*noise,
v=(1-sigma_min)*noise-x_0 公式。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sindre.ai.losses import balance_bce_loss


@dataclass(frozen=True)
class CrownLossConfig:
    occupancy_weight: float = 1.0
    sdf_weight: float = 0.2
    sdf_band_mm: float = 1.0
    sdf_beta_mm: float = 0.1
    balance_regions: bool = True

    def __post_init__(self):
        if self.occupancy_weight < 0 or self.sdf_weight < 0:
            raise ValueError("Loss 权重不能小于 0")
        if self.occupancy_weight + self.sdf_weight <= 0:
            raise ValueError("至少一个 Loss 权重必须为正数")
        if self.sdf_band_mm <= 0 or self.sdf_beta_mm <= 0:
            raise ValueError("毫米 SDF 截断范围与 SmoothL1 beta 必须 > 0")


class CrownReconstructionLoss(nn.Module):
    """对四档采样区间分别算 Occupancy 与 SDF，再对区间等权平均。

    由于四组采样数量不同，不直接在所有点上算 BCE，否则 uniform
    分布更容易支配近表面边界。分类采用 SindrePy 正负点平衡 BCE。
    """

    def __init__(self, config: CrownLossConfig | None = None):
        super().__init__()
        self.config = config or CrownLossConfig()

    def forward(
        self, prediction: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        logits = prediction["occupancy_logits"]
        predicted_sdf = prediction["sdf_mm"]
        labels = batch["occupancy"].float()
        sdf = batch["sdf_mm"].float()
        region = batch["query_region"].long()
        if logits.shape != labels.shape or predicted_sdf.shape != sdf.shape:
            raise ValueError("预测与 Occupancy/SDF 标签形状不一致")
        if region.shape != labels.shape or logits.ndim != 2 or logits.numel() == 0:
            raise ValueError("query_region 和标签必须为非空 [B,Q]")
        if not torch.isfinite(predicted_sdf).all() or not torch.isfinite(sdf).all():
            raise ValueError("SDF 含 NaN/Inf")
        if not torch.isfinite(logits).all():
            raise ValueError("Occupancy logits 含 NaN/Inf")
        if ((labels != 0) & (labels != 1)).any() or ((region < 0) | (region > 3)).any():
            raise ValueError("Occupancy 必须为 0/1，query_region 必须在 0~3")

        occ_parts = []
        sdf_parts = []
        masks = (
            [(region == group) for group in range(4)]
            if self.config.balance_regions else
            [torch.ones_like(region, dtype=torch.bool)]
        )
        for mask in masks:
            if not bool(mask.any()):
                continue
            # 复用 SindrePy 的平衡 BCEWithLogits，不重写已有损失函数。
            occ_parts.append(balance_bce_loss(logits[mask], labels[mask]))
            sdf_parts.append(F.smooth_l1_loss(
                predicted_sdf[mask].clamp(
                    -self.config.sdf_band_mm, self.config.sdf_band_mm,
                ),
                sdf[mask].clamp(
                    -self.config.sdf_band_mm, self.config.sdf_band_mm,
                ),
                beta=self.config.sdf_beta_mm,
            ))
        if not occ_parts:
            raise ValueError("没有有效的 Occupancy 区域")
        occupancy_loss = torch.stack(occ_parts).mean()
        sdf_loss = torch.stack(sdf_parts).mean()
        total = (
            self.config.occupancy_weight * occupancy_loss
            + self.config.sdf_weight * sdf_loss
        )
        with torch.no_grad():
            predicted_label = (logits >= 0).to(labels.dtype)
            accuracy = (predicted_label == labels).float().mean()
            intersect = (predicted_label * labels).sum()
            union = ((predicted_label + labels) > 0).float().sum()
            iou = intersect / union.clamp_min(1)
        return {
            "loss": total,
            "occupancy": occupancy_loss,
            "sdf": sdf_loss,
            "accuracy": accuracy,
            "iou": iou,
        }


class FlowMatchingLoss(nn.Module):
    """TRELLIS.2 标准速度场回归目标，仅消费官方 VAE 的**真实 latent**。

    不允许使用 query_points、occupancy 或 mesh.vertices 假充 x_0。
    当需要 SLat sparse flow 时接收真实的 sp.SparseTensor；输入必须具有
    feats/coords/replace，与官方稀疏张量契约一致。
    """

    def __init__(self, sigma_min: float = 1e-5):
        super().__init__()
        if not 0 <= sigma_min < 1:
            raise ValueError("sigma_min 必须在 [0,1)")
        self.sigma_min = sigma_min

    def forward(
        self, adapter: nn.Module, clean_latent: Tensor | object,
        conditions: dict[str, Tensor], *,
        t: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        sparse = (
            hasattr(clean_latent, "feats")
            and hasattr(clean_latent, "coords")
            and hasattr(clean_latent, "replace")
        )
        if sparse:
            data = clean_latent.feats
            batch_size = clean_latent.shape[0]
        else:
            if not isinstance(clean_latent, Tensor) or clean_latent.ndim != 5:
                raise ValueError(
                    "Flow 的 x_0 必须是官方解码器 latent：[B,C,R,R,R] "
                    "或官方 SparseTensor，而不是 Occupancy 点云"
                )
            data = clean_latent
            batch_size = data.shape[0]
        if t is None:
            t = torch.rand(batch_size, device=data.device, dtype=torch.float32)
        if t.shape != (batch_size,) or not torch.isfinite(t).all():
            raise ValueError("t 必须为有限的 [B]")
        if ((t < 0) | (t > 1)).any():
            raise ValueError("Flow t 必须在 0~1")
        if noise is None:
            noise = torch.randn_like(data)
        if noise.shape != data.shape:
            raise ValueError("noise 和 x_0.feats 的形状必须相同")

        # TRELLIS.2 官方 sparse tensor 重载了与 [B] 时间系数的广播运算。
        if sparse:
            noise_tensor = clean_latent.replace(noise)
            noisy_latent = (
                (1 - t).view(-1, 1) * clean_latent
                + (self.sigma_min + (1 - self.sigma_min) * t).view(-1, 1) * noise_tensor
            )
            prediction = adapter(noisy_latent, t * 1000, conditions)
            velocity = (1 - self.sigma_min) * noise - data
            predicted = prediction.feats
        else:
            shape = (batch_size,) + (1,) * (data.ndim - 1)
            t_view = t.reshape(shape)
            x_t = (
                (1 - t_view) * data
                + (self.sigma_min + (1 - self.sigma_min) * t_view) * noise
            )
            prediction = adapter(x_t, t * 1000, conditions)
            predicted = prediction
            velocity = (1 - self.sigma_min) * noise - data

        if predicted.shape != velocity.shape:
            raise ValueError("Flow 预测的 velocity 形状和真实 latent 不一致")
        mse = F.mse_loss(predicted.float(), velocity.float())
        return {"loss": mse, "flow_mse": mse}
