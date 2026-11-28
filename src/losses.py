"""牙冠 FDG 训练的全部损失函数。

仅计算粗/细体素结构、FDG 顶点、曲面相交、MD/BL/H 及开放颈缘边界监督。
不重复通用 BCE/Dice 类，也不保留旧 SDF/Flow 阶段的未使用训练损失。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FDGLossConfig:
    """统一 FDG 结构、几何和颈缘线 Loss 超参数。

    Attributes:
        coarse_weight: 粗结构 BCE + Dice 权重。
        fine_weight: 细体素 BCE + Dice 权重。
        offset_weight: 双顶点 SmoothL1 权重。
        intersection_weight: 表面相交 BCE 权重。
        margin_weight: 开放边界与颈缘线双向距离权重。
        size_weight: Mesh 实际近远中宽/颊舌厚/冠高与目标的回归权重。
        size_prior_weight: 仅使用病例条件自动估计 MD/BL/H 的辅助监督权重。
        size_beta_mm: 尺寸 SmoothL1 的毫米过渡阈值。
        max_positive_weight: 类别不平衡权重上限。
        offset_beta: SmoothL1 的过渡阈值。
        margin_tolerance_mm: 颈缘线距离的免惩罚容差。
        margin_samples_per_segment: 每条颈缘线段采样数。
    """

    coarse_weight: float = 1.0
    fine_weight: float = 1.0
    offset_weight: float = 0.5
    intersection_weight: float = 0.5
    margin_weight: float = 0.2
    size_weight: float = 0.2
    size_prior_weight: float = 0.4
    size_beta_mm: float = 0.2
    max_positive_weight: float = 20.0
    offset_beta: float = 0.1
    margin_tolerance_mm: float = 0.15
    margin_samples_per_segment: int = 4

    def __post_init__(self):
        positive = (
            self.coarse_weight, self.fine_weight, self.offset_weight,
            self.intersection_weight, self.margin_weight, self.size_weight,
            self.size_prior_weight,
        )
        if min(positive) < 0 or not any(positive):
            raise ValueError("FDG Loss 权重必须非负且至少一个 >0")
        if (self.max_positive_weight < 1 or self.offset_beta <= 0
                or self.size_beta_mm <= 0 or self.margin_tolerance_mm < 0
                or self.margin_samples_per_segment < 1):
            raise ValueError("FDG Loss 平衡权重、beta 或 margin 参数无效")


class CrownFDGLoss(nn.Module):
    """统一计算空间结构、FDG 几何以及柔性边界监督。

    将粗网格和细网格的结构监督独立计算；仅对真实有效 FDG 体素
    回归偏移与相交标志。所有数值敏感的计算强制 FP32。

    Args:
        config: 损失权重及稳定性参数。
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
        """计算预测开放边界与目标颈缘线的双向软距离。

        Args:
            vertices: FDG 解码后的局部毫米顶点 [V,3]。
            faces: 对应三角面 [F,3]。
            margin_line: 同一局部坐标系的颈缘线 [M,3]。

        Returns:
            Loss、平均/最大距离及开放边界顶点数。

        Raises:
            ValueError: Mesh 无开放边界、拓扑异常或颈缘线非法。
        """
        from .utils import CrownFDG

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
            "mean_distance_mm": 0.5 * (pred_to_line.detach().mean() + line_to_pred.detach().mean()),
            "max_distance_mm": torch.maximum(pred_to_line.detach().max(), line_to_pred.detach().max()),
            "boundary_vertices": torch.as_tensor(len(boundary), device=vertices.device),
        }

    def forward(
        self,
        prediction: dict[str, Tensor],
        batch: dict[str, Tensor],
        *,
        margin_meshes: list[tuple[Tensor, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """合并训练步骤的全部可用监督，不计算视图 Loss。

        Args:
            prediction: 模型粗/细体素与 FDG 预测。
            batch: 对齐的真实 coarse/fine/offset/intersection 标签。
            margin_meshes: 可选经原生 O-Voxel 解码的批量 Mesh，用于间隔
                计算颈缘线 Loss；缺省不计算此项。

        Returns:
            总 Loss、分类/几何分量、粗结构召回率等指标。

        Raises:
            ValueError: 张量形状、真值标签或正体素数量非法。
            FloatingPointError: 数值检查发现非有限的 Loss。
        """
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
        if "predicted_size" in prediction:
            size_gt = batch["size"].to(coarse.device).float()
            estimated = prediction["predicted_size"].float()
            if (estimated.shape != size_gt.shape or size_gt.shape != (coarse.shape[0], 3)
                    or not bool(torch.isfinite(estimated).all())):
                raise ValueError("尺寸预测必须为有限的 [B,3]")
            prior_mask = batch.get("size_prior_mask")
            if prior_mask is None:
                prior_mask = torch.ones(
                    coarse.shape[0], device=coarse.device, dtype=torch.bool,
                )
            prior_mask = prior_mask.to(coarse.device).bool()
            if prior_mask.shape != (coarse.shape[0],):
                raise ValueError("size_prior_mask 必须是 [B]")
            # 只在原始设计冠上学习自然尺寸先验，不能以人工随机尺寸变体
            # 来训练“不给尺寸时模型自己推测”的头。
            errors = F.smooth_l1_loss(
                estimated, size_gt, beta=c.size_beta_mm,
                reduction="none",
            ).mean(dim=-1)
            weight = prior_mask.float()
            prior_loss = (errors * weight).sum() / weight.sum().clamp_min(1)
            prior_mae = (
                (estimated.detach() - size_gt).abs().mean(dim=-1) * weight
            ).sum() / weight.sum().clamp_min(1)
            terms["size_prior"] = prior_loss
            terms["size_prior_mae_mm"] = prior_mae
            total = total + c.size_prior_weight * prior_loss
        if margin_meshes is not None:
            if len(margin_meshes) != coarse.shape[0]:
                raise ValueError("margin_meshes 数量与 batch 不一致")
            if ("size" not in batch
                    or batch["size"].shape != (len(margin_meshes), 3)):
                raise ValueError("Mesh 尺寸监督需要 [B,3] 的真实目标 MD/BL/H")
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
            # 这里将实际预测冠尺寸直接约束到目标 MD/BL/H，不使用
            # 单纯从 FDI 或形状平均中估计尺寸，避免网络忽略大小条件。
            predicted_sizes = []
            for verts, _ in margin_meshes:
                if verts.ndim != 2 or verts.shape[1] != 3 or not len(verts):
                    raise ValueError("FDG 解码 Mesh 不能为空，尺寸监督需要 [V,3] 顶点")
                verts = verts.float()
                predicted_sizes.append(torch.stack((
                    verts[:, 0].amax() - verts[:, 0].amin(),
                    verts[:, 1].amax() - verts[:, 1].amin(),
                    verts[:, 2].amax(),  # 局部原点在颈缘均值，不是整体 z span。
                )))
            size_prediction = torch.stack(predicted_sizes)
            target_size = batch["size"].to(size_prediction.device).float()
            size_loss = F.smooth_l1_loss(
                size_prediction, target_size, beta=c.size_beta_mm,
            )
            terms["size"] = size_loss
            terms["size_mae_mm"] = (
                size_prediction.detach() - target_size.detach()
            ).abs().mean()
            total = total + c.size_weight * size_loss
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
