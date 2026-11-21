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


class CrownMeshLoss(nn.Module):
    """可选的 *Mesh 已生成后* 几何损失，直接复用 PyTorch3D。

    沿用旧私有口腔网格生成项目的 Chamfer、法向、Laplacian 和 Edge
    正则化组合，不对 Occupancy decoder 的原始 logits 假称已有 Mesh。
    仅当 Shape VAE / Mesh decoder 真正输出 PyTorch3D Meshes 时使用。
    """

    def __init__(
        self, *,
        chamfer_weight: float = 1.0,
        normal_weight: float = 0.01,
        laplacian_weight: float = 0.1,
        edge_weight: float = 0.01,
        num_surface_points: int = 4096,
    ):
        super().__init__()
        if min(chamfer_weight, normal_weight, laplacian_weight, edge_weight) < 0:
            raise ValueError("Mesh Loss 权重不能为负")
        if num_surface_points < 1:
            raise ValueError("num_surface_points 必须为正数")
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.laplacian_weight = laplacian_weight
        self.edge_weight = edge_weight
        self.num_surface_points = num_surface_points

    def forward(self, predicted_mesh, target_mesh) -> dict[str, Tensor]:
        # Lazy import: 阶段一不用安装 PyTorch3D，避免 Windows / CUDA 编译成本。
        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.loss import (
            chamfer_distance, mesh_normal_consistency,
            mesh_laplacian_smoothing, mesh_edge_loss,
        )

        pred_points = sample_points_from_meshes(
            predicted_mesh, self.num_surface_points,
        )
        target_points = sample_points_from_meshes(
            target_mesh, self.num_surface_points,
        )
        chamfer, _ = chamfer_distance(pred_points, target_points)
        normal = mesh_normal_consistency(predicted_mesh)
        laplacian = mesh_laplacian_smoothing(
            predicted_mesh, method="uniform",
        )
        edge = mesh_edge_loss(predicted_mesh)
        total = (
            self.chamfer_weight * chamfer
            + self.normal_weight * normal
            + self.laplacian_weight * laplacian
            + self.edge_weight * edge
        )
        return {
            "loss": total,
            "chamfer": chamfer,
            "normal": normal,
            "laplacian": laplacian,
            "edge": edge,
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



@dataclass(frozen=True)
class FDGLossConfig:
    """FDG 训练的唯一 Loss 配置，所有系数非负且有上限。"""

    coarse_weight: float = 1.0
    fine_weight: float = 1.0
    offset_weight: float = 0.5
    intersection_weight: float = 0.5
    margin_weight: float = 0.2
    max_positive_weight: float = 20.0
    offset_beta: float = 0.1
    margin_tolerance_mm: float = 0.15
    margin_samples_per_segment: int = 4

    def __post_init__(self):
        positive = (
            self.coarse_weight, self.fine_weight, self.offset_weight,
            self.intersection_weight, self.margin_weight,
        )
        if min(positive) < 0 or not any(positive):
            raise ValueError("FDG Loss 权重必须非负且至少一个 >0")
        if (self.max_positive_weight < 1 or self.offset_beta <= 0
                or self.margin_tolerance_mm < 0
                or self.margin_samples_per_segment < 1):
            raise ValueError("FDG Loss 平衡权重、beta 或 margin 参数无效")


class CrownFDGLoss(nn.Module):
    """粗/细结构分类 + FDG 双顶点 + 相交标志 + 可选开放边界颈缘线。

    结构 logits 同时监督粗网格和细网格，防止 GT FDG 坐标泄漏成唯一输入。
    几何部分仅在真实正体素监督；全损失统一 fp32，避免 bf16 几何数值不稳。
    """

    def __init__(self, config: FDGLossConfig | None = None):
        super().__init__()
        self.config = config or FDGLossConfig()

    @staticmethod
    def _safe_bce(logits: Tensor, target: Tensor, max_weight: float) -> Tensor:
        with torch.no_grad():
            positives = target.sum()
            negatives = target.numel() - positives
            factor = (negatives / positives.clamp_min(1)).clamp(1, max_weight)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=factor,
        )

    @staticmethod
    def _dice(logits: Tensor, target: Tensor) -> Tensor:
        prob = torch.sigmoid(logits)
        # 对整体正类有效；空结构由上游校验并拒绝。
        numerator = (2 * prob * target).sum() + 1
        denom = prob.sum() + target.sum() + 1
        return 1 - numerator / denom

    @staticmethod
    def _point_to_segments(points: Tensor, edges: Tensor) -> Tensor:
        """有限线段欧氏距离，支持对预测 Mesh 顶点求导。"""
        a, b = edges[:, 0], edges[:, 1]
        delta = b - a
        t = ((points[:, None] - a[None]) * delta[None]).sum(-1)
        t = t / delta.square().sum(-1).clamp_min(1e-12)[None]
        projected = a[None] + t.clamp(0, 1)[..., None] * delta[None]
        return torch.linalg.vector_norm(
            points[:, None] - projected, dim=-1,
        ).min(dim=1).values

    def get_margin_loss(
        self, vertices: Tensor, faces: Tensor, margin_line: Tensor,
    ) -> dict[str, Tensor]:
        """开放边界与目标颈缘线双向柔性贴合，单位 mm。

        如果预测 Mesh 没有开放边界，必须报错而非返回0掩盖模型缺陷。
        """
        from utils import CrownFDG

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices 应为 [N,3]")
        line = torch.as_tensor(
            margin_line, device=vertices.device, dtype=torch.float32,
        )
        vertices = vertices.float()
        if line.ndim != 2 or line.shape[1] != 3 or len(line) < 3:
            raise ValueError("margin_line 应为 [M,3]")
        if not torch.isfinite(vertices).all() or not torch.isfinite(line).all():
            raise ValueError("颈缘线或预测 Mesh 包含 NaN/Inf")
        edges = CrownFDG.get_boundary_edges(faces, len(vertices))
        if not len(edges):
            raise ValueError("预测 Mesh 无开放边界；不能错误地将其视为已贴合颈缘线")
        boundary = torch.unique(edges.reshape(-1))
        a = vertices[edges]
        b = torch.stack((line, line.roll(-1, dims=0)), dim=1)
        fractions = torch.arange(
            self.config.margin_samples_per_segment,
            device=line.device, dtype=torch.float32,
        ) / self.config.margin_samples_per_segment
        sampled = (
            line[:, None] * (1 - fractions[None, :, None])
            + line.roll(-1, 0)[:, None] * fractions[None, :, None]
        ).reshape(-1, 3)
        pred_to_line = self._point_to_segments(vertices[boundary], b)
        line_to_pred = self._point_to_segments(sampled, a)
        tolerance = self.config.margin_tolerance_mm
        err_a = F.relu(pred_to_line - tolerance)
        err_b = F.relu(line_to_pred - tolerance)
        loss = 0.5 * (
            F.smooth_l1_loss(err_a, torch.zeros_like(err_a), beta=0.1)
            + F.smooth_l1_loss(err_b, torch.zeros_like(err_b), beta=0.1)
        )
        return {
            "loss": loss,
            "mean_distance_mm": pred_to_line.detach().mean(),
            "max_distance_mm": pred_to_line.detach().max(),
            "boundary_vertices": torch.as_tensor(len(boundary), device=vertices.device),
        }

    def forward(
        self,
        prediction: dict[str, Tensor],
        batch: dict[str, Tensor],
        *,
        margin_meshes: list[tuple[Tensor, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        required = ("coarse_logits", "fine_logits", "dual_offsets", "intersected_logits")
        if any(key not in prediction for key in required):
            raise ValueError(f"FDG 预测缺少字段: {required}")
        coarse = prediction["coarse_logits"].float()
        fine = prediction["fine_logits"].float()
        offset = prediction["dual_offsets"].float()
        intersects = prediction["intersected_logits"].float()
        coarse_gt = batch["coarse_target"].to(coarse.device).float()
        fine_gt = batch["fine_active"].to(fine.device).float()
        offset_gt = batch["fdg_target_offsets"].to(offset.device).float()
        intersects_gt = batch["fdg_target_intersected"].to(intersects.device).float()
        if (coarse.shape != coarse_gt.shape or fine.shape != fine_gt.shape
                or offset.shape != offset_gt.shape or intersects.shape != intersects_gt.shape
                or offset.shape != (*fine.shape, 3)):
            raise ValueError("FDG 结构、体素坐标及几何 GT 形状不一致")
        # 合并张量检查，避免每项 torch.bool 导致多次 GPU 同步。
        tensors = (
            coarse, fine, offset, intersects, coarse_gt, fine_gt,
            offset_gt, intersects_gt,
        )
        if not bool(torch.stack([torch.isfinite(x).all() for x in tensors]).all()):
            raise ValueError("FDG 预测或监督含 NaN/Inf")
        if bool((((coarse_gt != 0) & (coarse_gt != 1)).any()
                 | ((fine_gt != 0) & (fine_gt != 1)).any())):
            raise ValueError("粗/细体素占据必须是 0/1")
        selected = fine_gt.bool()
        if bool((~selected.any()) | (coarse_gt.sum() == 0)):
            raise ValueError("FDG 样本必须具有有效的真实正体素监督")

        c = self.config
        coarse_loss = (
            self._safe_bce(coarse, coarse_gt, c.max_positive_weight)
            + self._dice(coarse, coarse_gt)
        )
        fine_loss = (
            self._safe_bce(fine, fine_gt, c.max_positive_weight)
            + self._dice(fine, fine_gt)
        )
        offset_loss = F.smooth_l1_loss(
            offset[selected], offset_gt[selected], beta=c.offset_beta,
        )
        intersection_loss = self._safe_bce(
            intersects[selected], intersects_gt[selected], c.max_positive_weight,
        )
        total = (
            c.coarse_weight * coarse_loss
            + c.fine_weight * fine_loss
            + c.offset_weight * offset_loss
            + c.intersection_weight * intersection_loss
        )
        terms = {
            "coarse": coarse_loss,
            "fine": fine_loss,
            "offset": offset_loss,
            "intersection": intersection_loss,
        }
        if margin_meshes is not None:
            if len(margin_meshes) != coarse.shape[0]:
                raise ValueError("margin_meshes 数量与 batch 不一致")
            margins = [
                self.get_margin_loss(
                    verts, faces, batch["margin_line"][i],
                ) for i, (verts, faces) in enumerate(margin_meshes)
            ]
            margin = torch.stack([item["loss"] for item in margins]).mean()
            terms["margin"] = margin
            terms["margin_mm"] = torch.stack([
                item["mean_distance_mm"] for item in margins
            ]).mean()
            total = total + c.margin_weight * margin
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("FDG Loss 失稳：NaN/Inf")
        with torch.no_grad():
            coarse_pred = coarse >= 0
            fine_pred = fine >= 0
            terms["coarse_recall"] = (
                (coarse_pred & coarse_gt.bool()).sum() / coarse_gt.sum().clamp_min(1)
            )
            terms["fine_precision"] = (
                (fine_pred & fine_gt.bool()).sum() / fine_pred.sum().clamp_min(1)
            )
            terms["offset_mae"] = (offset[selected] - offset_gt[selected]).abs().mean()
        terms["loss"] = total
        return terms
