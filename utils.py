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
from torch.nn import functional as F

DEBUG = False
log = logging.getLogger("ToothForge3D.FDG")


@dataclass(frozen=True)
class FDGConfig:
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
    def get_boundary_indices(faces: Tensor, num_vertices: int) -> Tensor:
        """通过面片拓扑寻找未被成对共享的边，保持顶点的梯度路径。

        SindreMesh.get_boundary 适用于 NumPy 网格的交互/查看；
        Loss 需要 Torch Tensor 索引，因此只保留这里的最小拓扑计算。
        """
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
        return torch.unique(unique[counts == 1].reshape(-1))

    @staticmethod
    def get_frame(vertices: np.ndarray, margin_line: np.ndarray) -> tuple[np.ndarray, float]:
        """兼容性入口；实际需通过实例的 get_normalization 获取 padding 参数。"""
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
        boundary = self.get_boundary_indices(
            torch.from_numpy(f).long(), len(v),
        )
        if len(boundary):
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
            mesh = torch.from_numpy(kept_v)
            target = self._sample_margin(torch.from_numpy(margin))
            error = torch.cdist(mesh[boundary][None], target[None])[0].min(dim=1).values
            if float(error.mean()) > self.config.max_refine_distance_mm:
                raise ValueError(
                    f"自动裁剪后的边界平均距离 {float(error.mean()):.3f} mm 偏大，"
                    "请检查是否保留了正确的冠体部分"
                )
        return kept_v, kept_f

    def get_fdg(
        self, vertices: np.ndarray, faces: np.ndarray, margin_line: np.ndarray,
        *, cut_closed: bool = True,
    ) -> dict:
        """开放冠直接转为 FDG，不补底、不重建 SDF。

        对齐 O-Voxel 约定：返回的 dual_vertices 是 [-.5,.5] 空间绝对位置，
        训练标签 dual_offsets = dual_vertices * resolution - coords。
        """
        from o_voxel.convert import mesh_to_flexible_dual_grid

        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int32)
        line = np.asarray(margin_line, dtype=np.float32)
        center, scale_mm = self.get_normalization(v, line)
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
            raise ValueError("faces 必须为非空 [F,3]")
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError("faces 存在越界索引")
        if (f[:, 0] == f[:, 1]).any() or (f[:, 1] == f[:, 2]).any() or (f[:, 0] == f[:, 2]).any():
            raise ValueError("Mesh 存在退化三角面索引")
        v, f = self.get_open_crown(v, f, line, cut_closed=cut_closed)
        # 若原始输入是闭合冠，裁剪可能引入少量顶点；统一重新计算 FDG 帧。
        center, scale_mm = self.get_normalization(v, line)

        normalized = (v - center[None]) / scale_mm
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

    def get_margin_loss(
        self, vertices: Tensor, faces: Tensor, margin_line: Tensor
    ) -> dict[str, Tensor]:
        """开放边界 ↔ 真实颈缘线的双向柔性距离（单位 mm）。

        允许小于 tolerance 的偏移，不要求严格重合。
        """
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError("vertices 应为 [N,3]，单位 mm")
        margin_line = torch.as_tensor(
            margin_line, device=vertices.device, dtype=vertices.dtype
        )
        if margin_line.ndim != 2 or margin_line.shape[-1] != 3 or len(margin_line) < 3:
            raise ValueError("margin_line 应为 [M,3]，单位 mm")
        if not torch.isfinite(vertices).all() or not torch.isfinite(margin_line).all():
            raise ValueError("Mesh 与颈缘线不能包含 NaN/Inf")
        boundary_ids = self.get_boundary_indices(faces, len(vertices))
        if not len(boundary_ids):
            raise ValueError("预测 Mesh 无开放边界；不能错误地将其视为已贴合颈缘线")
        targets = self._sample_margin(margin_line)
        distances = torch.cdist(vertices[boundary_ids][None], targets[None])[0]
        edge_to_line = distances.min(dim=1).values
        line_to_edge = distances.min(dim=0).values
        tolerance = self.config.margin_tolerance_mm
        # 小误差免惩罚，大误差平滑增长：保留后期小幅人工修整空间。
        edge_loss = F.smooth_l1_loss(
            F.relu(edge_to_line - tolerance),
            torch.zeros_like(edge_to_line), beta=0.10,
        )
        coverage_loss = F.smooth_l1_loss(
            F.relu(line_to_edge - tolerance),
            torch.zeros_like(line_to_edge), beta=0.10,
        )
        return {
            "loss": (edge_loss + coverage_loss) * 0.5,
            "mean_distance_mm": edge_to_line.mean().detach(),
            "max_distance_mm": edge_to_line.max().detach(),
            "boundary_vertices": torch.tensor(
                len(boundary_ids), device=vertices.device, dtype=torch.int64,
            ),
        }

    def get_loss(
        self,
        pred_offsets: Tensor,
        pred_intersected_logits: Tensor,
        fdg: dict,
        *,
        mesh_vertices: Tensor | None = None,
        mesh_faces: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """轻量 FDG 网络可直接使用的训练损失。

        offset 回归 + intersected BCE；有开放预测 Mesh 时可选 Margin Loss。
        """
        target_offsets = torch.as_tensor(
            fdg["dual_offsets"], device=pred_offsets.device, dtype=pred_offsets.dtype
        )
        target_flags = torch.as_tensor(
            fdg["intersected"], device=pred_intersected_logits.device,
            dtype=pred_intersected_logits.dtype,
        )
        if (pred_offsets.shape != target_offsets.shape or
                pred_intersected_logits.shape != target_flags.shape):
            raise ValueError("FDG 预测必须与 GT sparse voxel coords 一一对应")
        offset_loss = F.smooth_l1_loss(pred_offsets, target_offsets, beta=0.10)
        intersected_loss = F.binary_cross_entropy_with_logits(
            pred_intersected_logits, target_flags,
        )
        total = offset_loss + intersected_loss
        result = {"offset": offset_loss, "intersected": intersected_loss}
        if (mesh_vertices is None) != (mesh_faces is None):
            raise ValueError("mesh_vertices 与 mesh_faces 应同时传入")
        if mesh_vertices is not None:
            margin_line = torch.as_tensor(
                fdg["margin_line_mm"], device=mesh_vertices.device,
                dtype=mesh_vertices.dtype,
            )
            margin_terms = self.get_margin_loss(
                mesh_vertices, mesh_faces, margin_line,
            )
            result["margin"] = margin_terms["loss"]
            total = total + self.config.margin_weight * margin_terms["loss"]
        result["loss"] = total
        return result

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
                return refined
            margin_line = torch.as_tensor(
                margin_line, device=refined.device, dtype=refined.dtype,
            )
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

    def build_database(
        self, input_db: str | Path, output_db: str | Path, *,
        max_cases: int | None = None, map_size_mb: int = 32768,
    ) -> dict:
        """把已存在的 ToothForge3D LMDB 转为**新的** FDG 数据库。

        旧 DB 不可被覆盖，FDG 数据不再依赖 closed-crown 的 SDF 标签。
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
        if map_size_mb < 1:
            raise ValueError("map_size_mb 必须为正数")
        from sindre.lmdb import Reader, Writer

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"total": 0, "written": 0, "errors": []}
        with Reader(str(input_path)) as reader:
            if reader.get_meta("build_complete") is not True:
                raise ValueError("源 LMDB 仍未构建完成")
            writer = Writer(str(output_path), map_size_limit=map_size_mb)
            try:
                writer.put_meta("schema_version", "toothforge3d_fdg_v1")
                writer.put_meta("build_complete", False)
                writer.put_meta("fdg_config", asdict(self.config))
                count = min(len(reader), max_cases) if max_cases else len(reader)
                for index in range(count):
                    report["total"] += 1
                    try:
                        sample = reader[index]
                        converted = self.get_fdg(
                            sample["target_vertices"],
                            sample["target_faces"],
                            sample["margin_line"],
                        )
                        # 避免把废弃的闭合冠监督复制进 FDG 新 DB。
                        old_keys = {
                            "query_points", "occupancy", "query_region",
                            "sdf_mm", "inside_fraction",
                            "inside_fraction_by_region",
                        }
                        record = {k: v for k, v in sample.items() if k not in old_keys}
                        record.update({
                            f"fdg_{key}": value for key, value in converted.items()
                        })
                        writer.put_sample(record)
                        report["written"] += 1
                    except Exception as exc:
                        report["errors"].append({"index": index, "error": str(exc)})
                        log.exception("FDG 样本 %d 转换失败", index)
                        raise
                writer.put_meta("written", report["written"])
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
):
    """手动传参运行，正式生成全库将 max_cases=None。"""
    return CrownFDG(FDGConfig(resolution=resolution)).build_database(
        input_db, output_db, max_cases=max_cases,
    )


if __name__ == "__main__":
    raise SystemExit("请调用 utils.main('旧数据库.db', '新FDG数据库.db')")
