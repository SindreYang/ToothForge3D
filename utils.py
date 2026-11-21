"""牙冠开放曲面 FDG 工具：复用 TRELLIS.2 的 O-Voxel，不复制 FDG 算法。

与当前 SDF/Occupancy 数据库并行：转换真实开放冠 -> FDG 标签；
在预测 Mesh 上计算柔性颈缘线距离，推理后可以轻微移动开放边界。
统一使用局部毫米坐标；FDG 内部使用归一化 [-0.5, 0.5] 坐标。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import logging

import numpy as np
import torch
from torch import Tensor

DEBUG = False
log = logging.getLogger("ToothForge3D.FDG")


@dataclass(frozen=True)
class FDGConfig:
    """原生 O-Voxel FDG 转换和颈缘线柔性修正参数。

    Attributes:
        resolution: FDG 网格分辨率。
        padding_ratio: 使用颈缘线和目标尺寸确定的空间留白比例。
        face_weight: 官方 QEF 网格转换的面片权重。
        boundary_weight: 官方 QEF 开放边界权重。
        regularization_weight: 官方 QEF 位置正则化权重。
        margin_tolerance_mm: 允许的边界贴合偏移。
        margin_weight: 留作可视化/后处理配置；训练 Loss 使用 losses.py。
        margin_samples_per_segment: 边界软贴合曲线加密数量。
        max_refine_distance_mm: 后处理允许参与微调的最大距离。
        max_refine_step_mm: 每次边界微调的最大毫米位移。
        refine_strength: 后处理的移动强度系数。
    """

    resolution: int = 128
    padding_ratio: float = 0.08
    face_weight: float = 1.0
    boundary_weight: float = 0.2
    regularization_weight: float = 0.01
    margin_tolerance_mm: float = 0.15
    margin_weight: float = 0.20
    margin_samples_per_segment: int = 4
    max_refine_distance_mm: float = 1.0
    max_refine_step_mm: float = 0.20
    refine_strength: float = 0.40

    def __post_init__(self):
        if self.resolution < 16:
            raise ValueError("FDG resolution 必须 >=16")
        if not 0 < self.padding_ratio < 0.4:
            raise ValueError("padding_ratio 必须在 (0,0.4)")
        if min(self.face_weight, self.boundary_weight,
               self.regularization_weight, self.margin_tolerance_mm,
               self.max_refine_distance_mm, self.max_refine_step_mm) < 0:
            raise ValueError("FDG 权重和距离不能为负")
        if self.margin_samples_per_segment < 1:
            raise ValueError("margin_samples_per_segment 必须 >=1")
        if not 0 <= self.refine_strength <= 1 or self.margin_weight < 0:
            raise ValueError("refine_strength 须在 [0,1]，margin_weight 不能为负")


class CrownFDG:
    """只包装官方 FDG 转换接口，颈缘线保留为独立的开放边界条件。

    训练时网络应预测与 coords 一一对应的 dual_offsets 和
    intersected_logits；当前 ToothForge3D 的 SDF 查询头并非该网络。
    """

    AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    def __init__(self, config: FDGConfig | None = None):
        self.config = config or FDGConfig()

    @staticmethod
    def get_boundary_edges(faces: Tensor, num_vertices: int) -> Tensor:
        """保留拓扑边的两个端点；训练距离以连续线段而非离散顶点计算。"""
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.numel() == 0:
            raise ValueError("faces 必须是非空 [F,3]")
        if num_vertices < 3 or (faces < 0).any() or (faces >= num_vertices).any():
            raise ValueError("faces 中存在非法顶点索引")
        faces = faces.long()
        edges = torch.sort(torch.cat(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
            dim=0,
        ), dim=-1).values
        unique, counts = torch.unique(edges, dim=0, return_counts=True)
        if (counts > 2).any():
            raise ValueError("Mesh 含非流形边，不能可靠识别颈缘线")
        return unique[counts == 1]

    @staticmethod
    def get_boundary_indices(faces: Tensor, num_vertices: int) -> Tensor:
        """边界顶点索引，可用于推理后微调（不用于计算离散点到点的训练 Loss）。"""
        return torch.unique(CrownFDG.get_boundary_edges(faces, num_vertices).reshape(-1))

    @staticmethod
    def get_frame(vertices: np.ndarray, margin_line: np.ndarray) -> tuple[np.ndarray, float]:
        """从冠体与颈缘线的共同包围盒计算中心和最大跨度（mm）。"""
        v = np.asarray(vertices, dtype=np.float32)
        line = np.asarray(margin_line, dtype=np.float32)
        if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3
                or line.ndim != 2 or line.shape[1] != 3 or len(line) < 3):
            raise ValueError("vertices / margin_line 必须为 [N,3] / [M,3]")
        if not np.isfinite(v).all() or not np.isfinite(line).all():
            raise ValueError("开放冠或颈缘线包含 NaN/Inf")
        combined = np.concatenate((v, line), axis=0)
        extent = np.ptp(combined, axis=0)
        span = float(extent.max())
        if not np.isfinite(span) or span < 1e-5:
            raise ValueError("开放冠几何尺度过小")
        center = ((combined.min(0) + combined.max(0)) / 2).astype(np.float32)
        return center, span

    def get_normalization(
        self, vertices: np.ndarray, margin_line: np.ndarray
    ) -> tuple[np.ndarray, float]:
        center, span = self.get_frame(vertices, margin_line)
        scale_mm = span / (1 - 2 * self.config.padding_ratio)
        return center, float(scale_mm)

    def get_condition_frame(
        self, margin_line: np.ndarray, size_mm: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """只根据推理时已知的颈缘线和尺寸生成 FDG 坐标框。

        绝不读取设计冠的目标顶点以确定坐标框，否则推理时无法重建。
        取以颈缘中心为核心的立方空间，半宽足够容纳牙冠上下任一方向。
        """
        line = np.asarray(margin_line, dtype=np.float32)
        size = np.asarray(size_mm, dtype=np.float32)
        if (line.ndim != 2 or line.shape[-1] != 3 or len(line) < 3
                or size.shape != (3,)
                or not np.isfinite(line).all() or not np.isfinite(size).all()
                or (size <= 0).any()):
            raise ValueError("FDG 坐标框必须由有效的 margin_line [M,3] 与 size_mm [3] 确定")
        center = line.mean(axis=0).astype(np.float32)
        extent = max(float(size.max()), float(np.ptp(line, axis=0).max()))
        scale_mm = 2 * extent / (1 - 2 * self.config.padding_ratio)
        return center, scale_mm

    def get_open_crown(
        self, vertices: np.ndarray, faces: np.ndarray, margin_line: np.ndarray, *,
        cut_closed: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """保证输入 FDG 的是开放牙冠；闭合冠优先复用 SindrePy MeshLib 裁剪。

        一般沿颈缘裁剪时保留较大部分是冠体，但并非所有病例都如此。
        裁剪后检查真实边界到颈缘的平均距离；过远则拒绝，需人工检查。
        """
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int32)
        margin = np.asarray(margin_line, dtype=np.float32)
        from losses import CrownFDGLoss

        boundary = self.get_boundary_indices(
            torch.from_numpy(f).long(), len(v),
        )
        if len(boundary):
            error = CrownFDGLoss().get_margin_loss(
                torch.from_numpy(v), torch.from_numpy(f).long(),
                torch.from_numpy(margin),
            )["mean_distance_mm"]
            if float(error) > self.config.max_refine_distance_mm:
                raise ValueError(
                    f"原始开放冠边界距离颈缘线 {float(error):.3f} mm 偏大，"
                    "请核实标注是否与 Mesh 同坐标"
                )
            return v, f
        if not cut_closed:
            raise ValueError("输入是闭合冠且 cut_closed=False，无法生成开放 FDG")
        from sindre.utils3d.algorithm import cut_mesh_by_meshlib

        kept_v, kept_f, _, _ = cut_mesh_by_meshlib(
            v, f, margin, get_bigger_part=True, smooth_boundary=False,
        )
        kept_v = np.asarray(kept_v, dtype=np.float32)
        kept_f = np.asarray(kept_f, dtype=np.int32)
        boundary = self.get_boundary_indices(
            torch.from_numpy(kept_f).long(), len(kept_v),
        )
        if not len(boundary):
            raise ValueError("SindrePy MeshLib 裁剪后没有开放边界")
        with torch.no_grad():
            error = CrownFDGLoss().get_margin_loss(
                torch.from_numpy(kept_v), torch.from_numpy(kept_f).long(),
                torch.from_numpy(margin),
            )["mean_distance_mm"]
            if float(error) > self.config.max_refine_distance_mm:
                raise ValueError(
                    f"自动裁剪后的边界平均距离 {float(error):.3f} mm 偏大，"
                    "请检查是否保留了正确的冠体部分"
                )
        return kept_v, kept_f

    def get_fdg(
        self, vertices: np.ndarray, faces: np.ndarray, margin_line: np.ndarray,
        *, cut_closed: bool = True, size_mm: np.ndarray | None = None,
    ) -> dict:
        """复用官方 O-Voxel 将开放冠转换为 FDG 训练标签。

        Args:
            vertices: 局部毫米坐标下的牙冠顶点 [V,3]。
            faces: 对应三角面 [F,3]。
            margin_line: 真实非平面颈缘线 [M,3]，局部毫米坐标。
            cut_closed: 输入封底 Mesh 时，是否尝试 SindrePy MeshLib 裁剪。
            size_mm: [MD,BL,H] 指定目标尺寸；正式 FDG 训练必须提供。

        Returns:
            包含 coords、dual_offsets、intersected 与毫米坐标恢复参数的字典。

        Raises:
            ValueError: 非流形、边界不符、坐标越界或无效训练标签。
        """
        from o_voxel.convert import mesh_to_flexible_dual_grid

        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int32)
        line = np.asarray(margin_line, dtype=np.float32)
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
            raise ValueError("faces 必须为非空 [F,3]")
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError("faces 存在越界索引")
        if (f[:, 0] == f[:, 1]).any() or (f[:, 1] == f[:, 2]).any() or (f[:, 0] == f[:, 2]).any():
            raise ValueError("Mesh 存在退化三角面索引")
        v, f = self.get_open_crown(v, f, line, cut_closed=cut_closed)
        # 若原始输入是闭合冠，裁剪可能引入少量顶点；统一重新计算 FDG 帧。
        center, scale_mm = (
            self.get_condition_frame(line, size_mm)
            if size_mm is not None else self.get_normalization(v, line)
        )

        normalized = (v - center[None]) / scale_mm
        if not np.isfinite(normalized).all() or np.abs(normalized).max() > 0.499:
            raise ValueError(
                "设计冠超出颈缘线+尺寸确定的 FDG 空间，"
                "请检查 size_mm 是否真实可用，禁止依据 GT Mesh 自行扩大空间"
            )
        coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
            torch.from_numpy(normalized).float(),
            torch.from_numpy(f).int(),
            grid_size=self.config.resolution,
            aabb=self.AABB,
            face_weight=self.config.face_weight,
            boundary_weight=self.config.boundary_weight,
            regularization_weight=self.config.regularization_weight,
        )
        coords = coords.detach().cpu().to(torch.int32)
        dual_vertices = dual_vertices.detach().cpu().float()
        intersected = intersected.detach().cpu().bool()
        if (coords.ndim != 2 or coords.shape[1] != 3 or not len(coords)
                or dual_vertices.shape != coords.shape
                or intersected.shape != coords.shape):
            raise ValueError("O-Voxel 返回的 FDG 标签维度不一致")
        if not torch.isfinite(dual_vertices).all():
            raise ValueError("FDG 双顶点含 NaN/Inf")
        if (coords < 0).any() or (coords >= self.config.resolution).any():
            raise ValueError("O-Voxel 返回 FDG 坐标越界")
        if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
            raise ValueError("O-Voxel 返回重复 FDG 坐标，禁止写入含冲突标签的数据库")
        offsets = dual_vertices * self.config.resolution - coords.float()
        if DEBUG:
            log.info("FDG cells=%d, resolution=%d, margin=%d",
                     len(coords), self.config.resolution, len(line))
        return {
            "coords": coords.numpy(),
            "dual_offsets": offsets.numpy(),
            "intersected": intersected.numpy().astype(np.uint8),
            "center_mm": center,
            "scale_mm": scale_mm,
            "margin_line_mm": line.copy(),
            "resolution": self.config.resolution,
        }

    def get_mesh(
        self,
        fdg: dict,
        *,
        dual_offsets: Tensor | None = None,
        intersected_logits: Tensor | None = None,
        split_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """O-Voxel CUDA 解码为开放 Mesh，坐标从 FDG 归一化空间还原成 mm。

        传入 predicted dual_offsets 可使 Mesh 顶点位置保留梯度；
        topology 来自交线 0/1 判定，本身不可微，需独立使用 BCE 监督。
        训练时可不传 intersected_logits，使用 GT 拓扑监督边界位置。
        """
        from o_voxel.convert import flexible_dual_grid_to_mesh

        if not torch.cuda.is_available():
            raise RuntimeError(
                "O-Voxel FDG 网格提取使用 CUDA hashmap；当前环境没有 CUDA。"
                "开放冠的 get_fdg/边界 Loss 可在 CPU 测试，但 get_mesh 需要 CUDA。"
            )
        device = torch.device("cuda")
        coords = torch.as_tensor(fdg["coords"], dtype=torch.int32, device=device)
        offsets = (
            torch.as_tensor(fdg["dual_offsets"], dtype=torch.float32, device=device)
            if dual_offsets is None else dual_offsets.to(device=device, dtype=torch.float32)
        )
        if offsets.shape != coords.shape:
            raise ValueError("预测 dual_offsets 必须为 [N,3]，N 与 FDG coords 相同")
        if intersected_logits is None:
            flags = torch.as_tensor(fdg["intersected"], device=device).bool()
        else:
            if intersected_logits.shape != coords.shape:
                raise ValueError("intersected_logits 必须为 [N,3]")
            flags = intersected_logits.to(device).sigmoid() > 0.5
        if split_weight is not None:
            split_weight = split_weight.to(device=device, dtype=torch.float32)
        vertices, faces = flexible_dual_grid_to_mesh(
            coords, offsets, flags,
            split_weight=split_weight,
            aabb=self.AABB,
            grid_size=int(fdg["resolution"]),
            train=False,
        )
        # 返回毫米坐标；不要在这里 snap 到目标线，先评估自然预测误差。
        vertices = vertices * float(fdg["scale_mm"]) + torch.as_tensor(
            fdg["center_mm"], device=vertices.device, dtype=vertices.dtype
        )[None]
        return vertices, faces.long()

    def refine_margin(
        self, vertices: Tensor, faces: Tensor, margin_line: Tensor
    ) -> Tensor:
        """推理后只轻微调整靠近颈缘线的边界顶点，不修改拓扑与内部顶点。

        不适合修复巨大边界偏差或错误拓扑。
        """
        with torch.no_grad():
            refined = vertices.detach().clone()
            boundary_ids = self.get_boundary_indices(faces, len(refined))
            if not len(boundary_ids):
                raise ValueError("Mesh 没有开放边界，无法轻微修整到颈缘线")
            margin_line = torch.as_tensor(
                margin_line, device=refined.device, dtype=refined.dtype,
            )
            if margin_line.ndim != 2 or margin_line.shape[-1] != 3 or len(margin_line) < 3:
                raise ValueError("margin_line 必须为 [M,3]")
            if not torch.isfinite(margin_line).all():
                raise ValueError("margin_line 含 NaN/Inf")
            targets = self._sample_margin(margin_line)
            distances = torch.cdist(refined[boundary_ids][None], targets[None])[0]
            dist, closest = distances.min(dim=1)
            to_target = targets[closest] - refined[boundary_ids]
            step = to_target * self.config.refine_strength
            step = step * (
                self.config.max_refine_step_mm /
                torch.linalg.vector_norm(step, dim=1, keepdim=True).clamp_min(1e-9)
            ).clamp(max=1.0)
            step[dist > self.config.max_refine_distance_mm] = 0
            refined[boundary_ids] += step
            return refined

    def _sample_margin(self, margin_line: Tensor) -> Tensor:
        # 周期性曲线，不依赖端点重复；用于损失和推理微调，保持 Torch 梯度。
        fractions = torch.arange(
            self.config.margin_samples_per_segment,
            device=margin_line.device, dtype=margin_line.dtype,
        ) / self.config.margin_samples_per_segment
        following = margin_line.roll(-1, dims=0)
        samples = (
            margin_line[:, None, :] * (1 - fractions[None, :, None])
            + following[:, None, :] * fractions[None, :, None]
        )
        return samples.reshape(-1, 3)

    @staticmethod
    def show(
        vertices: np.ndarray, faces: np.ndarray, margin_line: np.ndarray,
        *, screenshot: str | Path,
    ) -> Path:
        """DEBUG：透明开放冠 + 红色颈缘线，方便肉眼检查是否选错冠底。"""
        import vedo

        path = Path(screenshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = vedo.Mesh([vertices, faces]).c("#b5c7d7").alpha(0.65)
        line = np.asarray(margin_line, dtype=np.float32)
        curve = vedo.Line(np.concatenate([line, line[:1]], axis=0)).c("#ef4444").lw(4)
        plotter = vedo.Plotter(offscreen=True, size=(1000, 800), bg="#f7f9fb")
        try:
            plotter.show(mesh, curve, axes=1, interactive=False)
            plotter.screenshot(str(path))
        finally:
            plotter.close()
        return path

    @staticmethod
    def augment_crown_size(
        vertices: np.ndarray,
        faces: np.ndarray,
        margin_line: np.ndarray,
        size_mm: np.ndarray,
        scales: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """对开放冠做保持颈缘线固定的条件尺寸增强。

        仅改变目标冠的冠身形态，牙弓、颈缘线和 DINOv3 输入不变。
        对应的 MD/BL/H 与新 FDG 标签同时更新，保证条件与 GT 成对。

        Args:
            vertices: 局部毫米坐标的开放冠顶点，形状为 [V,3]。
            faces: 目标冠三角面，形状为 [F,3]。
            margin_line: 真实颈缘线，形状为 [M,3]。
            size_mm: 原始设计的近远中、颊舌、高度，形状为 [3]。
            scales: 三个轴独立的目标冠尺寸比例。

        Returns:
            (增强顶点, 对应的新 MD/BL/H 尺寸)，不修改传入数组。

        Raises:
            ValueError: 输入几何、比例或增强后的尺寸非法。
        """
        from scipy.spatial import cKDTree

        verts = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        margin = np.asarray(margin_line, dtype=np.float32)
        size = np.asarray(size_mm, dtype=np.float32)
        factors = np.asarray(scales, dtype=np.float32)
        if (verts.ndim != 2 or verts.shape[1] != 3 or
                margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 3 or
                size.shape != (3,) or factors.shape != (3,) or
                not np.isfinite(verts).all() or not np.isfinite(margin).all() or
                not np.isfinite(size).all() or (size <= 0).any() or
                not np.isfinite(factors).all() or (factors < 0.75).any() or
                (factors > 1.25).any()):
            raise ValueError("尺寸增强要求有效的开放冠、颈缘线和 0.75~1.25 的缩放比例")
        # 将非平面颈缘曲线加密：边缘上的形变为0，越远离边缘作用越强。
        samples = np.concatenate([
            margin[:, None, :] * (1 - t) + np.roll(margin, -1, axis=0)[:, None, :] * t
            for t in np.linspace(0, 1, 8, endpoint=False)
        ], axis=1).reshape(-1, 3)
        distances = cKDTree(samples).query(verts, workers=-1)[0].astype(np.float32)
        distance_weight = np.clip(distances / 2.0, 0.0, 1.0)
        blend = distance_weight ** 2 * (3 - 2 * distance_weight)
        # 开放冠真正的拓扑边界顶点保持不变，避免非平面颈缘漂移。
        boundary = CrownFDG.get_boundary_indices(
            torch.from_numpy(faces.copy()), len(verts),
        ).cpu().numpy()
        if len(boundary) == 0:
            raise ValueError("增强前必须提供开放冠，闭合 Mesh 先沿颈缘线裁剪")
        blend[boundary] = 0.0
        center = margin.mean(axis=0)
        augmented = verts + (verts - center) * (factors - 1)[None, :] * blend[:, None]
        augmented[boundary] = verts[boundary]
        old_extent = np.maximum(np.ptp(verts, axis=0), 1e-3)
        new_extent = np.maximum(np.ptp(augmented, axis=0), 1e-3)
        updated_size = size * (new_extent / old_extent)
        if not np.isfinite(augmented).all() or (updated_size <= 0).any():
            raise ValueError("尺寸增强产生非法顶点/长宽高")
        return augmented.astype(np.float32), updated_size.astype(np.float32)


    def build_database(
        self, input_db: str | Path, output_db: str | Path, *,
        max_cases: int | None = None, map_size_mb: int = 32768,
        image_encoder: object | None = None,
        num_jaw_points: int = 4096, image_size: int = 256,
        augment_variants: int = 4, augment_scale: float = 0.10,
        seed: int = 1024,
    ) -> dict:
        """将真实病例和成对尺寸增强写入新的 Sindre FDG LMDB。

        在预处理阶段完成三视图 DINOv3、开放冠裁剪、FDG 网格编码。
        同一原始病例的尺寸副本使用相同 source_group，防止验证泄漏。

        Args:
            input_db: 原始病例 LMDB 或旧 Occupancy 数据库路径。
            output_db: 全新 FDG LMDB 目标路径，不允许覆盖原数据。
            max_cases: 最多转换的原始病例数；None 表示全部。
            map_size_mb: Sindre LMDB 映射文件容量，单位 MB。
            image_encoder: 可选已加载的 SindrePy DINOv3Encoder。
            num_jaw_points: 预处理的同颌点云采样数。
            image_size: 预处理三视图的像素分辨率。
            augment_variants: 每份原始冠生成的配对尺寸样本数。
            augment_scale: MD/BL/H 三轴独立的比例变化幅度。
            seed: 可复现的尺寸增强随机种子。

        Returns:
            包含源病例数量、写入样本数量、错误信息的报告字典。

        Raises:
            ValueError: 输入不完整、目标路径不合法或增强范围非法。
            RuntimeError: FDG 原生算子未安装或数据不能生成有效开放冠。
        """
        input_path = Path(input_db).expanduser().resolve()
        output_path = Path(output_db).expanduser().resolve()
        if input_path == output_path:
            raise ValueError("原数据库与新 FDG 数据库不能相同")
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if output_path.exists():
            raise FileExistsError(f"禁止覆盖已有 FDG DB: {output_path}")
        if max_cases is not None and max_cases < 1:
            raise ValueError("max_cases 应为正数")
        if map_size_mb < 1 or num_jaw_points < 1 or image_size < 64:
            raise ValueError("map_size_mb / num_jaw_points / image_size 参数非法")
        if augment_variants < 1 or not 0 <= augment_scale <= 0.20:
            raise ValueError("augment_variants>=1, augment_scale 应在 [0,0.20]")
        from sindre.lmdb import Reader, Writer

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"total": 0, "written": 0, "errors": []}
        with Reader(str(input_path)) as reader:
            source_schema = reader.get_meta("schema_version")
            from_occ = source_schema == "toothforge3d_occ_v1"
            if from_occ and reader.get_meta("build_complete") is not True:
                raise ValueError("旧 Occupancy LMDB 尚未完成")
            if source_schema == "toothforge3d_fdg_v1":
                raise ValueError("源数据库已经是 FDG；请勿重复转换")
            writer = Writer(str(output_path), map_size_limit=map_size_mb)
            try:
                writer.put_meta("schema_version", "toothforge3d_fdg_v1")
                writer.put_meta("build_complete", False)
                writer.put_meta("fdg_config", asdict(self.config))
                writer.put_meta("coordinate_frame", "margin_size_v1")
                writer.put_meta("augmentation", {
                    "variants_per_case": augment_variants,
                    "size_scale": augment_scale,
                    "margin_fixed": True,
                })
                source_groups = []
                count = min(len(reader), max_cases) if max_cases else len(reader)
                for index in range(count):
                    report["total"] += 1
                    try:
                        raw = reader[index]
                        if from_occ:
                            sample = raw
                        else:
                            from process import DentalCase
                            sample = DentalCase(raw).get_condition(
                                crop_mode="full", tooth_dropout=0,
                                num_points=num_jaw_points,
                                render_views=False, image_encoder=image_encoder,
                                image_size=image_size,
                            )
                        if (sample.get("margin_line") is None
                                or sample.get("size") is None
                                or sample.get("view_features") is None):
                            raise ValueError(
                                "病例缺少颈缘线、真实可用尺寸或 DINOv3 特征；"
                                "请在预处理传入 image_encoder"
                            )
                        open_vertices, open_faces = self.get_open_crown(
                            sample["target_vertices"],
                            sample["target_faces"],
                            sample["margin_line"],
                        )
                        base_size = np.asarray(sample["size"], dtype=np.float32)
                        rng = np.random.default_rng(seed + index * 7919)
                        for variant in range(augment_variants):
                            if variant == 0:
                                variant_vertices = open_vertices
                                variant_size = base_size
                            else:
                                factors = tuple(rng.uniform(
                                    1 - augment_scale, 1 + augment_scale, size=3
                                ).tolist())
                                variant_vertices, variant_size = self.augment_crown_size(
                                    open_vertices, open_faces, sample["margin_line"],
                                    base_size, factors,
                                )
                            converted = self.get_fdg(
                                variant_vertices, open_faces, sample["margin_line"],
                                cut_closed=False, size_mm=variant_size,
                            )
                            if DEBUG:
                                self.show(
                                    variant_vertices, open_faces, sample["margin_line"],
                                    screenshot=output_path.parent
                                    / f"{output_path.stem}_debug"
                                    / f"{index:06d}_{variant:02d}_open_margin.png",
                                )
                            record = {
                                key: sample[key] for key in (
                                    "jaw", "target_position", "margin_line",
                                    "jaw_points", "jaw_labels", "jaw_present",
                                    "view_features",
                                )
                            }
                            record["size"] = variant_size
                            record["target_vertices"] = variant_vertices
                            record["target_faces"] = open_faces
                            record["source_group"] = index
                            record["variant_index"] = variant
                            if "source_fdi" in sample:
                                record["source_fdi"] = sample["source_fdi"]
                            record.update({
                                f"fdg_{key}": value for key, value in converted.items()
                            })
                            writer.put_sample(record)
                            report["written"] += 1
                            source_groups.append(index)

                    except Exception as exc:
                        report["errors"].append({"index": index, "error": str(exc)})
                        log.exception("FDG 样本 %d 转换失败", index)
                        raise
                writer.put_meta("written", report["written"])
                writer.put_meta("source_groups", source_groups)
                writer.put_meta("build_complete", True)
            finally:
                writer.close()
                output_path.with_suffix(".report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return report


def main(
    input_db: str | Path, output_db: str | Path,
    *, resolution: int = 128, max_cases: int | None = 5,
    image_encoder: object | None = None, augment_variants: int = 4,
):
    """直接从原始病例 LMDB 构建 FDG，预处理时一次性缓存 DINOv3。"""
    return CrownFDG(FDGConfig(resolution=resolution)).build_database(
        input_db, output_db, max_cases=max_cases,
        image_encoder=image_encoder, augment_variants=augment_variants,
    )


if __name__ == "__main__":
    raise SystemExit("请调用 utils.main('旧数据库.db', '新FDG数据库.db')")
