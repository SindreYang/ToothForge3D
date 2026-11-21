"""ToothForge3D 新版 MyDataset + 占据场采样 + Sindre LMDB 重建。

DEBUG 为全局开关：True 时输出 Occupancy 内/外点与设计冠的可视化 PNG。
输入 DB 是原始病例，输出 DB 是新结构；绝不在原始 DB 上覆写。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from process import DentalCase

DEBUG = False
log = logging.getLogger("ToothForge3D.datasets")


@dataclass(frozen=True)
class SamplingConfig:
    """严格采用毫米制；所有 Occupancy 查询点在设计冠局部坐标中。

    近表面点沿面法向高斯抖动，另有较小切向噪声；
    空间点在设计冠局部 AABB 外扩区域均匀采样。
    0=外部、1=内部，不把 0/1 Occupancy 错命名成 SDF。
    """

    near_count_005: int = 8192
    near_count_015: int = 4096
    near_count_035: int = 4096
    uniform_count: int = 16384
    noise_mm: tuple[float, float, float] = (0.05, 0.15, 0.35)
    tangential_ratio: float = 0.15
    uniform_padding_mm: float = 0.8
    num_jaw_points: int = 8192
    image_size: int = 256
    min_inside_fraction: float = 0.02
    min_outside_fraction: float = 0.02
    repair: str = "vedo"  # vedo / sindre_views / none
    sdf_backend: str = "pcu"  # pcu (CPU) / cubvh (CUDA)
    seed: int = 1024

    def __post_init__(self):
        if min(self.near_count_005, self.near_count_015, self.near_count_035, self.uniform_count) < 1:
            raise ValueError("四组 Occupancy 采样点数均须 >=1")
        if self.num_jaw_points < 1 or self.image_size < 64:
            raise ValueError("牙弓点数必须 >=1，渲染分辨率必须 >=64")
        if any(x <= 0 or not np.isfinite(x) for x in self.noise_mm):
            raise ValueError("噪声尺度必须是正的毫米值")
        if not 0 <= self.tangential_ratio <= 1 or self.uniform_padding_mm < 0:
            raise ValueError("切向噪声比例必须在 0~1，空间扩边不能为负")
        if not 0 <= self.min_inside_fraction < 0.5 or not 0 <= self.min_outside_fraction < 0.5:
            raise ValueError("最小占据比例必须在 [0,0.5)")
        if self.repair not in ("vedo", "sindre_views", "none"):
            raise ValueError("repair 必须为 vedo、sindre_views 或 none")
        if self.sdf_backend not in ("pcu", "cubvh"):
            raise ValueError("sdf_backend 必须为 pcu 或 cubvh")

    @property
    def num_queries(self) -> int:
        return (self.near_count_005 + self.near_count_015
                + self.near_count_035 + self.uniform_count)


class CrownOccupancySampler:
    """从真实设计冠构建经过水密性检查的空间 Occupancy 监督。"""

    def __init__(self, config: SamplingConfig | None = None):
        self.config = config or SamplingConfig()

    def get_watertight(self, vertices: np.ndarray, faces: np.ndarray):
        """优先修补现有冠底边界；不能确认水密时停止，不生成错误标签。"""
        import trimesh

        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        if not mesh.is_watertight:
            if self.config.repair == "none":
                raise ValueError("设计冠非水密，Occupancy 不能直接判定内外")
            if self.config.repair == "vedo":
                import vedo

                closed = vedo.Mesh([v, f]).clean().fill_holes().triangulate()
                mesh = trimesh.Trimesh(
                    vertices=np.asarray(closed.vertices),
                    faces=np.asarray(closed.cells), process=False,
                )
            else:
                from sindre.utils3d.watertight import remesh_watertight_by_views

                watertight = remesh_watertight_by_views(
                    grid_resolution=256, restore=True,
                )
                out_vertices, out_faces = watertight(v, f)
                mesh = trimesh.Trimesh(
                    vertices=out_vertices, faces=out_faces, process=False,
                )
        if not mesh.is_watertight:
            raise ValueError("设计冠水密化失败；不能生成可靠 Occupancy 标签")
        mesh.fix_normals()
        if (not mesh.is_winding_consistent or not np.isfinite(mesh.volume)
                or abs(float(mesh.volume)) < 1e-8):
            raise ValueError("设计冠法向、体积或拓扑异常")
        if mesh.volume < 0:
            mesh.invert()
        return mesh

    def sample(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        seed: int | None = None,
    ) -> dict:
        """空间点与 Occupancy 必须一起生成，禁止只扰动点不重新算标签。"""
        mesh = self.get_watertight(vertices, faces)
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        # 近表面采样必须来自原始设计冠表面，不能把填孔生成的人工底盖
        # 当成真实牙冠几何监督；填孔网格只用于 SDF 内外判定。
        surface_v = np.asarray(vertices, dtype=np.float64)
        surface_f = np.asarray(faces, dtype=np.int64)
        a, b, c = (
            surface_v[surface_f[:, 0]],
            surface_v[surface_f[:, 1]],
            surface_v[surface_f[:, 2]],
        )
        cross = np.cross(b - a, c - a)
        double_area = np.linalg.norm(cross, axis=1)
        if double_area.sum() < 1e-10:
            raise ValueError("设计冠三角形总面积为 0")
        probabilities = double_area / double_area.sum()

        point_groups = []
        region_groups = []
        counts = (self.config.near_count_005, self.config.near_count_015,
                  self.config.near_count_035)
        for group, (count, sigma) in enumerate(zip(counts, self.config.noise_mm)):
            face_idx = rng.choice(len(surface_f), size=count, p=probabilities)
            # 面积均匀的三角面重心坐标采样。
            r1 = np.sqrt(rng.random((count, 1)))
            r2 = rng.random((count, 1))
            surface = ((1 - r1) * a[face_idx]
                       + r1 * (1 - r2) * b[face_idx]
                       + r1 * r2 * c[face_idx])
            normals = cross[face_idx] / np.maximum(double_area[face_idx, None], 1e-12)
            normal_noise = rng.normal(0, sigma, (count, 1)) * normals
            tangent_noise = rng.normal(
                0, sigma * self.config.tangential_ratio, (count, 3),
            )
            # 去掉沿法向的切向分量；允许噪声同时进入内侧与外侧。
            tangent_noise -= (tangent_noise * normals).sum(axis=1, keepdims=True) * normals
            point_groups.append(surface + normal_noise + tangent_noise)
            region_groups.append(np.full(count, group, dtype=np.uint8))

        mins = v.min(axis=0) - self.config.uniform_padding_mm
        maxs = v.max(axis=0) + self.config.uniform_padding_mm
        point_groups.append(rng.uniform(
            mins, maxs, (self.config.uniform_count, 3),
        ))
        region_groups.append(np.full(self.config.uniform_count, 3, dtype=np.uint8))
        query = np.concatenate(point_groups).astype(np.float32)
        region = np.concatenate(region_groups)

        # 复用 SindrePy GetSDF，而不是维护另一份几何查询实现。
        # SindrePy 定义 SDF < 0 为内部；SDF == 0 的表面按外部处理。
        from sindre.ai.vision3d.implicit_utils.utils import GetSDF

        sdf_query = GetSDF()
        if self.config.sdf_backend == "pcu":
            signed = sdf_query.apply_pcu(v, f, query)
        else:
            signed = sdf_query.apply_cubvh(
                v.astype(np.float32), f.astype(np.int32), query
            )
        signed = np.asarray(
            signed.detach().cpu().numpy() if hasattr(signed, "detach") else signed,
            dtype=np.float32,
        ).reshape(-1)
        if signed.shape != (len(query),) or not np.isfinite(signed).all():
            raise ValueError("SDF 查询长度错误或含 NaN/Inf")

        # 检查确定在包围盒外的六个远点，捕获错误的符号约定与反转法向。
        centre = (mins + maxs) / 2
        radius = float((maxs - mins).max()) * 3
        far = np.concatenate([
            centre[None, :] + np.eye(3) * radius,
            centre[None, :] - np.eye(3) * radius,
        ]).astype(np.float32)
        if self.config.sdf_backend == "pcu":
            far_sdf = sdf_query.apply_pcu(v, f, far)
        else:
            far_sdf = sdf_query.apply_cubvh(
                v.astype(np.float32), f.astype(np.int32), far
            )
        far_sdf = np.asarray(
            far_sdf.detach().cpu().numpy() if hasattr(far_sdf, "detach") else far_sdf
        ).reshape(-1)
        if not np.isfinite(far_sdf).all() or np.any(far_sdf <= 0):
            raise ValueError("已知网格外部点 SDF 未全部为正，Occupancy 方向不可靠")

        occupancy = (signed < 0).astype(np.uint8)
        fraction = float(occupancy.mean())
        if not (self.config.min_inside_fraction <= fraction
                <= 1 - self.config.min_outside_fraction):
            raise ValueError(
                f"Occupancy 内外比例异常 inside={fraction:.4f}，禁止保存错误标签"
            )
        group_inside = np.array([
            occupancy[region == group].mean() for group in range(4)
        ], dtype=np.float32)
        return {
            "query_points": query,
            "occupancy": occupancy,
            "query_region": region,
            "sdf_mm": signed,
            "inside_fraction": fraction,
            "inside_fraction_by_region": group_inside,
            "watertight_vertices": v.astype(np.float32),
            "watertight_faces": f.astype(np.int32),
        }

    def show(self, record: dict, path: str | Path, *, max_points: int = 12000):
        """DEBUG 时渲染蓝色内部点、橙色外部点、半透明水密设计冠。"""
        import vedo

        points = np.asarray(record["query_points"], dtype=np.float32)
        inside = np.asarray(record["occupancy"], dtype=np.uint8).astype(bool)
        if len(points) != len(inside):
            raise ValueError("query_points / occupancy 数量不匹配")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        step = max(1, len(points) // max_points)
        crown = vedo.Mesh([
            record["watertight_vertices"], record["watertight_faces"]
        ]).c("#bbc6d3").alpha(0.25)
        outer = vedo.Points(points[~inside][::step], r=3).c("#f59e0b")
        inner = vedo.Points(points[inside][::step], r=3).c("#2563eb")
        plotter = vedo.Plotter(offscreen=True, size=(1100, 850), bg="#f8fafc")
        try:
            plotter.show(crown, outer, inner, axes=1, interactive=False)
            plotter.screenshot(str(path))
        finally:
            plotter.close()
        log.info("DEBUG Occupancy %s inside=%.3f", path, float(inside.mean()))
        return path


class MyDataset(Dataset):
    """读取新 LMDB。随机裁剪仅作用于整颌点云，不再改动已经标注的查询点。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        augment: bool = True,
        jaw_points: int = 4096,
        tooth_dropout: float = 0.20,
        jaw_noise_mm: float = 0.02,
        query_count: int | None = None,
        include_images: bool = True,
        include_mesh: bool = True,
        seed: int = 1024,
    ):
        from sindre.lmdb import Reader

        if jaw_points < 1 or not 0 <= tooth_dropout <= 1 or jaw_noise_mm < 0:
            raise ValueError("jaw_points/tooth_dropout/jaw_noise_mm 参数非法")
        if query_count is not None and query_count < 4:
            raise ValueError("query_count 至少为 4，需覆盖四组采样区域")
        self.reader = Reader(str(db_path))
        # 未完成构建的 LMDB 不允许进入训练，防止把半截写入当作全量数据。
        completed = self.reader.get_meta("build_complete")
        if completed is not True:
            self.reader.close()
            raise ValueError("LMDB 未完成构建或版本不兼容：缺少 build_complete=True")
        if self.reader.get_meta("schema_version") != "toothforge3d_occ_v1":
            self.reader.close()
            raise ValueError("LMDB schema_version 与 ToothForge3D Occupancy V1 不一致")
        self.augment = augment
        self.jaw_points = jaw_points
        self.tooth_dropout = tooth_dropout
        self.jaw_noise_mm = jaw_noise_mm
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.query_count = query_count
        self.include_images = include_images
        self.include_mesh = include_mesh

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index: int):
        record = self.reader[index]
        points = np.asarray(record["jaw_points"], dtype=np.float32)
        labels = np.asarray(record["jaw_labels"], dtype=np.int64)
        worker = get_worker_info()
        rng = (
            np.random.default_rng(self.seed + index * 9973)
            if not self.augment else
            np.random.default_rng(worker.seed + index * 9973)
            if worker is not None else self.rng
        )
        if self.augment and len(points):
            mode = rng.choice(("full", "large", "medium", "local", "none"),
                              p=(0.15, 0.20, 0.25, 0.30, 0.10))
            if mode == "none":
                points, labels = points[:0], labels[:0]
            elif mode != "full":
                ranges = {"large": (0.65, 1.0), "medium": (0.35, 0.75),
                          "local": (0.10, 0.45)}
                distances = np.linalg.norm(points[:, :3], axis=1)
                low, high = ranges[mode]
                radius = np.quantile(distances, rng.uniform(low, high))
                keep = distances <= radius
                points, labels = points[keep], labels[keep]
            if len(points) and self.tooth_dropout:
                for tooth in np.unique(labels[labels > 0]):
                    if rng.random() < self.tooth_dropout:
                        keep = labels != tooth
                        points, labels = points[keep], labels[keep]

        present = len(points) > 0
        if present:
            chosen = rng.choice(
                len(points), self.jaw_points, replace=len(points) < self.jaw_points
            )
            points, labels = points[chosen].copy(), labels[chosen].copy()
            if self.augment and self.jaw_noise_mm:
                points[:, :3] += rng.normal(
                    0, self.jaw_noise_mm, (self.jaw_points, 3),
                ).astype(np.float32)
        else:
            points = np.zeros((self.jaw_points, 6), dtype=np.float32)
            labels = np.zeros((self.jaw_points,), dtype=np.int64)

        # 按四种真实采样区域等比例取 query，减少 32k 点高显存的训练负担。
        # query_points、occupancy、region 和 SDF 永远使用同一组索引。
        query_points = np.asarray(record["query_points"], dtype=np.float32)
        query_labels = np.asarray(record["occupancy"], dtype=np.float32)
        query_region = np.asarray(record["query_region"], dtype=np.int64)
        query_sdf = np.asarray(record["sdf_mm"], dtype=np.float32)
        if (query_points.ndim != 2 or query_points.shape[1] != 3
                or query_labels.shape != (len(query_points),)
                or query_region.shape != (len(query_points),)
                or query_sdf.shape != (len(query_points),)):
            raise ValueError("LMDB Occupancy 查询点与标签长度不一致")
        if self.query_count is not None and self.query_count < len(query_points):
            quotas = [self.query_count // 4] * 4
            for i in range(self.query_count % 4):
                quotas[i] += 1
            pieces = []
            for group, quota in enumerate(quotas):
                ids = np.flatnonzero(query_region == group)
                if len(ids) == 0:
                    raise ValueError(f"LMDB 缺少 query_region={group}，不能分层采样")
                pieces.append(rng.choice(ids, size=quota, replace=len(ids) < quota))
            ids = np.concatenate(pieces)
            rng.shuffle(ids)
            query_points = query_points[ids]
            query_labels = query_labels[ids]
            query_region = query_region[ids]
            query_sdf = query_sdf[ids]

        # target_faces / vertices 变长，由 collate 保留为 list。
        output = {
            "jaw": torch.tensor(record["jaw"], dtype=torch.long),
            "target_position": torch.tensor(record["target_position"], dtype=torch.long),
            "margin_line": torch.as_tensor(np.asarray(record["margin_line"]), dtype=torch.float32),
            "size": torch.as_tensor(np.asarray(record["size"]), dtype=torch.float32),
            "jaw_points": torch.as_tensor(points, dtype=torch.float32),
            "jaw_labels": torch.as_tensor(labels, dtype=torch.long),
            "jaw_present": torch.tensor(present, dtype=torch.bool),
            "query_points": torch.as_tensor(query_points, dtype=torch.float32),
            "occupancy": torch.as_tensor(query_labels, dtype=torch.float32),
            "query_region": torch.as_tensor(query_region, dtype=torch.long),
            "sdf_mm": torch.as_tensor(query_sdf, dtype=torch.float32),
            "inside_fraction_by_region": torch.as_tensor(
                np.asarray(record["inside_fraction_by_region"]), dtype=torch.float32
            ),
        }
        if self.include_images:
            output["view_images"] = torch.as_tensor(
                np.asarray(record["view_images"]).copy()
            ).permute(0, 3, 1, 2).float() / 255
        if self.include_mesh:
            output["target_vertices"] = torch.as_tensor(
                np.asarray(record["target_vertices"]), dtype=torch.float32
            )
            output["target_faces"] = torch.as_tensor(
                np.asarray(record["target_faces"]), dtype=torch.long
            )
        if "view_features" in record:
            output["view_features"] = torch.as_tensor(
                np.asarray(record["view_features"]), dtype=torch.float32
            )
        if DEBUG:
            log.info(
                "样本=%s FDI=%s 内部点比例=%.3f 同颌点=%d",
                index, record["source_fdi"], float(output["occupancy"].mean()), len(points),
            )
        return output

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """固定形状张量堆叠，目标三角网格以列表形式保留。"""
        if not batch:
            raise ValueError("空 batch")
        result = {}
        for key in batch[0]:
            if key in ("target_vertices", "target_faces"):
                result[key] = [item[key] for item in batch]
            else:
                result[key] = torch.stack([item[key] for item in batch], dim=0)
        return result

    def close(self) -> None:
        self.reader.close()



class FDGDataset(Dataset):
    """新开放冠 FDG LMDB：结构/几何数据共享相同的空间坐标。

    训练只有条件图像的 DINO tokens，禁止加载/渲染 RGB 目标冠图。
    每份样本采固定数量正/负细体素；真实完整 FDG coords 仅作为监督
    或间隔性重建指标，并不会传给网络的推理结构分支。
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        resolution: int = 128,
        coarse_resolution: int = 32,
        query_count: int = 2048,
        jaw_points: int = 2048,
        augment: bool = True,
        seed: int = 1024,
    ):
        from sindre.lmdb import Reader

        if (resolution < coarse_resolution or coarse_resolution < 4
                or resolution % coarse_resolution or query_count < 4 or jaw_points < 1):
            raise ValueError("FDG 分辨率/查询数/牙颌点数参数无效")
        self.reader = Reader(str(db_path))
        if (self.reader.get_meta("schema_version") != "toothforge3d_fdg_v1"
                or self.reader.get_meta("build_complete") is not True
                or self.reader.get_meta("coordinate_frame") != "margin_size_v1"):
            self.reader.close()
            raise ValueError(
                "FDG LMDB 必须是已完整构建的 margin_size_v1 坐标规范；"
                "请用最新版 utils.CrownFDG.build_database 重建"
            )
        self.resolution = resolution
        self.coarse_resolution = coarse_resolution
        self.query_count = query_count
        self.jaw_points = jaw_points
        self.augment = augment
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.reader[index]
        worker = get_worker_info()
        rng = (
            np.random.default_rng(worker.seed + index * 7919)
            if self.augment and worker is not None else
            self.rng if self.augment else
            np.random.default_rng(self.seed + index * 7919)
        )
        resolution = self.resolution
        stride = resolution // self.coarse_resolution
        if int(row["fdg_resolution"]) != resolution:
            raise ValueError("FDG 数据库分辨率与训练配置不一致")
        coords = np.asarray(row["fdg_coords"], dtype=np.int64)
        offsets = np.asarray(row["fdg_dual_offsets"], dtype=np.float32)
        intersects = np.asarray(row["fdg_intersected"], dtype=np.float32)
        if (coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 1
                or offsets.shape != coords.shape or intersects.shape != coords.shape
                or (coords < 0).any() or (coords >= resolution).any()
                or not np.isfinite(offsets).all()
                or not np.logical_or(intersects == 0, intersects == 1).all()):
            raise ValueError("FDG voxel coords / offsets / intersected 标签无效")

        # 重复/越界坐标已在 FDG 数据构建阶段严格校验；
        # 不在每个训练 Epoch 重复做 O(NlogN) 唯一性排序。
        parents = coords // stride
        coarse = np.zeros((self.coarse_resolution,) * 3, dtype=np.float32)
        coarse[parents[:, 0], parents[:, 1], parents[:, 2]] = 1
        # 正例至多一半；负例混合局部扰动和全局均匀，避免全靠背景学会零输出。
        n_pos = self.query_count // 2
        positive_idx = rng.choice(len(coords), size=n_pos, replace=len(coords) < n_pos)
        positive = coords[positive_idx]
        # 按一维唯一索引过滤负例；使用 NumPy 向量化而非 Python 每点循环。
        keys = coords[:, 0] * resolution ** 2 + coords[:, 1] * resolution + coords[:, 2]
        neg = []
        need = self.query_count - n_pos
        candidates_per_try = max(2 * need, 512)
        for _ in range(12):
            random_neighbors = coords[rng.integers(len(coords), size=candidates_per_try // 2)]
            neighbors = random_neighbors + rng.integers(
                -4, 5, size=(len(random_neighbors), 3)
            )
            uniform = rng.integers(
                0, resolution,
                size=(candidates_per_try - len(random_neighbors), 3),
            )
            candidates = np.concatenate((neighbors, uniform))
            inside = ((candidates >= 0) & (candidates < resolution)).all(axis=1)
            candidates = candidates[inside]
            cand_keys = (
                candidates[:, 0] * resolution ** 2
                + candidates[:, 1] * resolution + candidates[:, 2]
            )
            available = candidates[~np.isin(cand_keys, keys)]
            if len(available):
                neg.append(available[:need])
                need -= min(need, len(available))
            if need <= 0:
                break
        if need > 0:
            raise ValueError("未能采足 FDG 负体素；需要检查 coords 是否异常")
        negative = np.concatenate(neg)[:self.query_count - n_pos]
        fine_coords = np.concatenate((positive, negative)).astype(np.int64)
        fine_active = np.concatenate((
            np.ones(n_pos, dtype=np.float32),
            np.zeros(len(negative), dtype=np.float32),
        ))
        target_offsets = np.concatenate((
            offsets[positive_idx], np.zeros((len(negative), 3), np.float32),
        ))
        target_intersects = np.concatenate((
            intersects[positive_idx], np.zeros((len(negative), 3), np.float32),
        ))
        perm = rng.permutation(len(fine_coords))
        # 同颌仅是附加条件，与目标设计冠 FDG 坐标无关。
        jaw = np.asarray(row["jaw_points"], dtype=np.float32)
        labels = np.asarray(row["jaw_labels"], dtype=np.int64)
        if len(jaw):
            ids = rng.choice(len(jaw), size=self.jaw_points, replace=len(jaw) < self.jaw_points)
            jaw, labels = jaw[ids].copy(), labels[ids].copy()
            if self.augment:
                jaw[:, :3] += rng.normal(0, 0.02, size=(len(jaw), 3)).astype(np.float32)
        else:
            jaw = np.zeros((self.jaw_points, 6), np.float32)
            labels = np.zeros((self.jaw_points,), np.int64)
        if "view_features" not in row or row["view_features"] is None:
            raise ValueError("FDG 数据库缺少预计算 DINOv3 view_features")
        return {
            "jaw": torch.tensor(row["jaw"], dtype=torch.long),
            "target_position": torch.tensor(row["target_position"], dtype=torch.long),
            "margin_line": torch.as_tensor(np.asarray(row["margin_line"]), dtype=torch.float32),
            "size": torch.as_tensor(np.asarray(row["size"]), dtype=torch.float32),
            "jaw_points": torch.from_numpy(jaw),
            "jaw_labels": torch.as_tensor(labels, dtype=torch.long),
            "jaw_present": torch.tensor(bool(np.asarray(row["jaw_points"]).shape[0])),
            "view_features": torch.as_tensor(np.asarray(row["view_features"]), dtype=torch.float32),
            "coarse_target": torch.from_numpy(coarse),
            "fine_coords": torch.as_tensor(fine_coords[perm].copy(), dtype=torch.long),
            "fine_active": torch.as_tensor(fine_active[perm].copy()),
            "fdg_target_offsets": torch.from_numpy(target_offsets[perm].copy()),
            "fdg_target_intersected": torch.from_numpy(target_intersects[perm].copy()),
            "fdg_coords": torch.as_tensor(coords.copy(), dtype=torch.long),
            "fdg_offsets": torch.as_tensor(offsets.copy()),
            "fdg_intersected": torch.as_tensor(intersects.copy()),
            "fdg_center_mm": torch.as_tensor(np.asarray(row["fdg_center_mm"]), dtype=torch.float32),
            "fdg_scale_mm": torch.tensor(float(row["fdg_scale_mm"]), dtype=torch.float32),
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("FDG batch 不能为空")
        ragged = ("fdg_coords", "fdg_offsets", "fdg_intersected")
        return {
            key: [row[key] for row in batch] if key in ragged
            else torch.stack([row[key] for row in batch])
            for key in batch[0]
        }

    def close(self):
        self.reader.close()


def build_database(
    input_db: str | Path,
    output_db: str | Path,
    *,
    config: SamplingConfig | None = None,
    map_size_mb: int = 32768,
    max_cases: int | None = None,
    image_encoder=None,
    strict: bool = True,
) -> dict:
    """从原始 Sindre LMDB 生成**新的** LMDB。

    必须提供包含设计冠、同颌与对颌网格的真实病例；不能用只包含
    vertices/faces 的 datasets/*.json 冒充完整病例。
    """
    config = config or SamplingConfig()
    input_path = Path(input_db).expanduser().resolve()
    output_path = Path(output_db).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("输入和输出数据库不能相同")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"不会覆盖已有的输出数据库: {output_path}")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases 必须为正数")
    if map_size_mb <= 0:
        raise ValueError("map_size_mb 必须为正数（单位 MB）")
    from sindre.lmdb import Reader, Writer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = CrownOccupancySampler(config)
    report = {"total": 0, "written": 0, "errors": []}
    debug_dir = output_path.parent / f"{output_path.stem}_debug"

    with Reader(str(input_path)) as reader:
        count = min(len(reader), max_cases) if max_cases is not None else len(reader)
        writer = Writer(str(output_path), map_size_limit=map_size_mb)
        try:
            writer.put_meta("schema_version", "toothforge3d_occ_v1")
            writer.put_meta("build_complete", False)
            writer.put_meta("sampling_config", asdict(config))
            writer.put_meta("input_database", str(input_path))
            for index in range(count):
                report["total"] += 1
                try:
                    case = DentalCase(reader[index])
                    record = case.get_condition(
                        crop_mode="full", tooth_dropout=0,
                        num_points=config.num_jaw_points,
                        render_views=True, image_encoder=image_encoder,
                        image_size=config.image_size,
                    )
                    if not record["geometry_ready"]:
                        raise ValueError("当前病例没有可靠的颈缘线或牙冠尺寸")
                    crown_extent = float(np.ptp(record["target_vertices"], axis=0).max())
                    # 旧版 Occupancy 曾在牙弓归一化 [-1,1] 空间采样。
                    # 当前毫米制噪声必须拒绝明显不是 mm 的几何尺度。
                    if not np.isfinite(crown_extent) or not 2.0 <= crown_extent <= 30.0:
                        raise ValueError(
                            f"牙冠最大尺寸 {crown_extent:.3f} 不在预期 mm 区间 [2,30]；"
                            "请先核实原始数据单位/归一化"
                        )
                    occ = sampler.sample(
                        record["target_vertices"], record["target_faces"],
                        seed=config.seed + index * 1009,
                    )
                    if DEBUG:
                        path = debug_dir / f"{index:06d}_fdi{case.target_fdi}.png"
                        sampler.show(occ, path)
                        log.info(
                            "DEBUG 样本 %d: inside=%.2f%%，每层内部率=%s，采样量=%s",
                            index, occ["inside_fraction"] * 100,
                            np.round(occ["inside_fraction_by_region"], 3).tolist(),
                            np.bincount(occ["query_region"], minlength=4).tolist(),
                        )
                    row = {
                        "raw_index": index,
                        "source_fdi": record["source_fdi"],
                        "jaw": record["jaw"],
                        "target_position": record["target_position"],
                        "margin_line": record["margin_line"].astype(np.float32),
                        "size": record["size"].astype(np.float32),
                        "jaw_points": record["jaw_points"].astype(np.float32),
                        "jaw_labels": record["jaw_labels"].astype(np.int16),
                        "view_images": record["view_images"].astype(np.uint8),
                        "query_points": occ["query_points"],
                        "occupancy": occ["occupancy"],
                        "query_region": occ["query_region"],
                        "sdf_mm": occ["sdf_mm"],
                        "inside_fraction": occ["inside_fraction"],
                        "inside_fraction_by_region": occ["inside_fraction_by_region"],
                        "target_vertices": record["target_vertices"].astype(np.float32),
                        "target_faces": record["target_faces"].astype(np.int32),
                        "origin_world": record["origin_world"].astype(np.float32),
                        "local_frame": record["local_frame"].astype(np.float32),
                        "frame_verified": bool(record["frame_verified"]),
                    }
                    if record["view_features"] is not None:
                        row["view_features"] = np.asarray(
                            record["view_features"], dtype=np.float32
                        )
                    writer.put_sample(row)
                    report["written"] += 1
                    if index % 50 == 0:
                        log.info("Occupancy 新库进度 %d/%d，写入 %d", index + 1, count, report["written"])
                except Exception as exc:
                    log.exception("病例 %d 预处理失败: %s", index, exc)
                    report["errors"].append({"index": index, "error": str(exc)})
                    if strict:
                        raise
            writer.put_meta("processed", report["total"])
            writer.put_meta("written", report["written"])
            writer.put_meta("failed", len(report["errors"]))
            # 即使 strict=False 有少量坏样本被跳过，也明确记录失败数量；
            # 全部处理完成前绝不能写 True。
            writer.put_meta("build_complete", True)
        finally:
            writer.close()
            report_path = output_path.with_suffix(".report.json")
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return report


def main(
    input_db: str | Path,
    output_db: str | Path,
    *,
    max_cases: int | None = None,
    map_size_mb: int = 32768,
    strict: bool = True,
):
    """直接用 main 参数控制，不引入 argparse。开启 DEBUG=True 可输出占据点截图。"""
    report = build_database(
        input_db, output_db,
        max_cases=max_cases,
        map_size_mb=map_size_mb,
        strict=strict,
    )
    print(f"新 LMDB: {output_db}, 写入={report['written']}, 失败={len(report['errors'])}")
    return report


if __name__ == "__main__":
    # 改为实际原始 LMDB 路径再执行，避免误用独立 Mesh JSON 文件。
    raise SystemExit("请调用 datasets.main('原始病例.db', '新训练集.db')")
