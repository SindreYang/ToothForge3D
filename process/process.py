"""全冠设计数据预处理：同颌随机裁剪、逐点牙位、可选颈缘线。

输入为 current_mesh、opposite_mesh、design_mesh；FDI 确定目标颌。
设计冠只作为监督，不参与方向估计及渲染。颊舌侧渲染上下颌，咬合面只拍目标颌；
对颌不输入 PTv3 点云条件。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from sindre.lmdb import Reader
from sindre.utils3d.core.dental.utils import convert_fdi2idx
from sindre.utils3d.core.sample import sample_pcd_farthest
from sindre.utils3d.sindremesh import SindreMesh
from sindre.utils3d.sindrepcd import SindrePCD

from src.views import CrownViews, get_world_to_local_transform, transform_points
from src.utils import CrownFDG, FDGConfig, _validate_training_frame
from dataclasses import asdict
import logging
import hashlib

log = logging.getLogger("ToothForge3D.preprocess")

DEBUG = False
DEBUG_SAMPLE_INDEX = 0

# 与 SindrePy 的 FDI→1~16 规则保持一致。注意下颌 31→1，41→9。
UPPER_FDI = tuple(range(11, 19)) + tuple(range(21, 29))
LOWER_FDI = tuple(range(31, 39)) + tuple(range(41, 49))


def normalize_record(value: Any) -> Any:
    """递归解包 LMDB 返回的零维 NumPy 标量，同时保留网格数组。"""
    if isinstance(value, dict):
        return {key: normalize_record(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_case_data(
    data: dict[str, Any] | str | Path,
    *,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """读取 JSON 病例或 SindrePy LMDB 中指定序号的病例。"""
    if isinstance(data, dict):
        if sample_index is not None:
            raise ValueError("传入病例字典时不能指定 sample_index")
        record = data
    else:
        path = Path(data)
        if path.suffix.lower() in {".db", ".lmdb"}:
            if sample_index is None:
                raise ValueError("读取 LMDB 必须指定 sample_index")
            if isinstance(sample_index, bool) or not isinstance(
                sample_index, (int, np.integer)
            ):
                raise TypeError("sample_index 必须是整数")
            with Reader(str(path)) as reader:
                if sample_index < 0 or sample_index >= len(reader):
                    raise IndexError(
                        f"sample_index={sample_index} 超出数据库范围 [0, {len(reader)})"
                    )
                record = reader[int(sample_index)]
        else:
            if sample_index is not None:
                raise ValueError("JSON 病例不能指定 sample_index")
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
    if not isinstance(record, dict):
        raise ValueError("病例记录必须是字典")
    return normalize_record(record)


class DentalCase:
    """提取训练或无目标冠推理所需的真实牙颌条件。

    current_mesh 标签支持真实 FDI 或 fdi 表索引。推理模式只接收
    target_fdi、真实颈缘线、MD/BL/H、local_frame 及可见的牙颌扫描；
    不读取设计冠顶点，防止将目标几何泄漏进输入。
    """

    def __init__(self, data: dict[str, Any], *, inference: bool = False):
        """构建训练/推理病例条件，推理时不读取目标设计冠。

        Args:
            data: 包含 current_mesh 的真实病例。
            inference: 是否为无目标设计冠的推理模式。

        Raises:
            ValueError: 关键网格、FDI、颈缘线、尺寸或方向信息不完整。
        """
        data = normalize_record(data)
        if not isinstance(data, dict) or "current_mesh" not in data:
            raise ValueError("病例必须含有 current_mesh")
        self.inference = inference
        if inference:
            if "design_mesh" in data:
                raise ValueError("推理病例不得携带 design_mesh")
            for key in ("target_fdi", "margin_line"):
                if data.get(key) is None:
                    raise ValueError(f"推理病例缺少 {key}")
            raw_fdi = data["target_fdi"]
            design_mesh = {}
        else:
            design_mesh = data.get("design_mesh")
            if not isinstance(design_mesh, dict):
                raise ValueError("训练病例必须提供规范的 design_mesh")
            raw_fdi = data.get("target_fdi")
            if raw_fdi is None:
                raw_fdi = design_mesh.get("fdi")
            if design_mesh.get("fdi") is not None and raw_fdi is not None and int(raw_fdi) != int(design_mesh["fdi"]):
                raise ValueError("target_fdi 与 design_mesh.fdi 不一致")
            if raw_fdi is None:
                raise ValueError("design_mesh 必须包含真实 FDI 牙位")
        self.target_fdi = int(raw_fdi)
        if self.target_fdi in UPPER_FDI:
            self.jaw = 0
        elif self.target_fdi in LOWER_FDI:
            self.jaw = 1
        else:
            raise ValueError(f"设计记录的 FDI 无效或不是恒牙：{self.target_fdi}")
        if "is_upper" in data and data["is_upper"] is not None:
            if not isinstance(data["is_upper"], (bool, np.bool_)):
                raise ValueError("is_upper 存在时必须是布尔值")
            if bool(data["is_upper"]) != (self.jaw == 0):
                raise ValueError("is_upper 与目标牙位所属上下颌不一致")
        self.data = data
        self.opposite_jaw = 1 - self.jaw  # 仅记录对颌类别，不将 opposite_mesh 输入网络。
        self.valid_fdi = UPPER_FDI if self.jaw == 0 else LOWER_FDI

        # SindrePy 已有 FDI 转换函数，这里只处理被验证过的真实 FDI。
        self.target_position = int(convert_fdi2idx(np.array([self.target_fdi]))[0])
        self.current_mesh = data["current_mesh"]
        self.design_mesh = design_mesh
        self.vertices = np.asarray(self.current_mesh["vertices"], dtype=np.float32)
        self.faces = np.asarray(self.current_mesh["faces"], dtype=np.int64)
        self.design_vertices = (
            np.empty((0, 3), dtype=np.float32) if inference
            else np.asarray(self.design_mesh["vertices"], dtype=np.float32)
        )
        self.design_faces = (
            np.empty((0, 3), dtype=np.int64) if inference
            else np.asarray(self.design_mesh["faces"], dtype=np.int64)
        )
        meshes = [("current_mesh", self.vertices, self.faces)]
        if not inference:
            meshes.append(("design_mesh", self.design_vertices, self.design_faces))
        for name, vertices, faces in meshes:
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
                raise ValueError(f"{name}.vertices 需要 [N,3] 且至少 3 个点")
            if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
                raise ValueError(f"{name}.faces 需要 [F,3]")
            if not np.isfinite(vertices).all() or (faces < 0).any() or (faces >= len(vertices)).any():
                raise ValueError(f"{name} 包含非法坐标或三角面片")

        self.labels_known = self.current_mesh.get("vertex_labels") is not None
        labels = (np.asarray(self.current_mesh["vertex_labels"], dtype=np.int64).reshape(-1)
                  if self.labels_known else np.full(len(self.vertices), -1, dtype=np.int64))
        if len(labels) != len(self.vertices):
            raise ValueError("vertex_labels 必须与 current_mesh.vertices 一一对应")
        label_table = self.current_mesh.get("fdi")
        if label_table is not None and self.labels_known:
            table = np.asarray(label_table, dtype=np.int64).reshape(-1)
            # 文件有两种实际编码方式：顶点存 FDI 或存 fdi 字典索引。
            # 如 fdi=[0,32,41,2031] 且标签只有 0..3，则属于字典索引。
            if len(table) > 0 and labels.min() >= 0 and labels.max() < len(table):
                labels = table[labels]
        self.fdi_labels = labels
        self._cached_normals: np.ndarray | None = None

        # 当前牙颌可能包含目标牙预备体：例如 2031 = 2000 + 31。
        # 预备体属于实际可观察的几何，可出现在输入图像中，但不能作为目标牙冠标签。
        self.base_label = 2000 + self.target_fdi
        self.base_vertex_ids = np.flatnonzero(labels == self.base_label)

        # 只从 current_mesh 构造拍摄场景；目标设计冠永远不能被追加进去。
        # 对 current_mesh 中误含的目标成品冠，按照目标 FDI 明确排除。
        scene_keep = (
            (labels == 0) | np.isin(labels, self.valid_fdi) | (labels == self.base_label)
        ) & (labels != self.target_fdi)
        render_face_mask = (np.all(scene_keep[self.faces], axis=1) if self.labels_known
                            else np.ones(len(self.faces), dtype=bool))
        self.scene_faces_original = self.faces[render_face_mask]
        if len(self.scene_faces_original):
            scene_ids = np.unique(self.scene_faces_original.reshape(-1))
            index_map = np.full(len(self.vertices), -1, dtype=np.int64)
            index_map[scene_ids] = np.arange(len(scene_ids))
            self.scene_vertices = self.vertices[scene_ids]
            self.scene_faces = index_map[self.scene_faces_original]
        else:
            self.scene_vertices = np.empty((0, 3), dtype=np.float32)
            self.scene_faces = np.empty((0, 3), dtype=np.int64)

        # 对颌只作为侧向两张图的可视化背景；不输入 PTv3 或其他 3D 生成分支。
        opposite = data.get("opposite_mesh")
        self.opposite_vertices = None
        self.opposite_faces = None
        if opposite is not None:
            if not isinstance(opposite, dict) or "vertices" not in opposite or "faces" not in opposite:
                raise ValueError("opposite_mesh 必须包含 vertices 和 faces")
            opposite_vertices = np.asarray(opposite["vertices"], dtype=np.float32)
            opposite_faces = np.asarray(opposite["faces"], dtype=np.int64)
            if (
                opposite_vertices.ndim != 2 or opposite_vertices.shape[1] != 3
                or len(opposite_vertices) < 3
                or opposite_faces.ndim != 2 or opposite_faces.shape[1] != 3
                or len(opposite_faces) == 0
                or not np.isfinite(opposite_vertices).all()
                or (opposite_faces < 0).any()
                or (opposite_faces >= len(opposite_vertices)).any()
            ):
                raise ValueError("opposite_mesh 不是有效的三角网格")
            self.opposite_vertices = opposite_vertices
            self.opposite_faces = opposite_faces


        # 保留同颌牙和牙龈(0)，过滤目标牙冠、对颌以及 2031 等特殊标签。
        # 0 不是牙位；它会作为背景标签随整颌点云进入 PTv3。
        allowed = (labels == 0) | (
            np.isin(labels, self.valid_fdi) & (labels != self.target_fdi)
        )
        self.context_vertex_ids = (np.flatnonzero(allowed) if self.labels_known
                                   else np.arange(len(self.vertices)))
        valid_labels = labels[self.context_vertex_ids].copy()
        self.context_labels = (np.asarray(convert_fdi2idx(valid_labels), dtype=np.int64)
                               if self.labels_known else np.full(len(valid_labels), 17, dtype=np.int64))
        if not np.isin(self.context_labels, range(18)).all():
            raise ValueError("同颌牙/背景/未知标签转换到 0~17 失败")

        self.margin_line = self.get_margin_line()
        self._views_builder = None
        if self.margin_line is not None or data.get("local_frame") is not None:
            self._views_builder = CrownViews(
                self.scene_vertices,
                self.scene_faces,
                self.margin_line,
                jaw_points=self.scene_vertices,
                jaw_labels=self.context_labels,
                target_position=self.target_position,
                local_frame=(None if str(data.get("frame_source", "")).startswith("design_crown") else data.get("local_frame")),
            )
            self.frame = self._views_builder.frame
            self.origin_world = self._views_builder.origin.astype(np.float32)
            self.frame_verified = self._views_builder.anatomical_verified
            self.frame_source = self._views_builder.frame_source
            self.frame_confidence = self._views_builder.frame_confidence
            self.world_to_local_transform = self._views_builder.world_to_local_transform
        else:
            # 封闭 Mesh 无真实颈缘线时不能可靠判别冠向，禁止伪造标准三视图。
            self.frame = np.eye(3, dtype=np.float32)
            self.origin_world = (
                self.margin_line.mean(axis=0) if self.margin_line is not None
                else self.design_vertices.mean(axis=0)
            )
            self.frame_verified = False
            self.frame_source = "unresolved"
            self.frame_confidence = None
            self.world_to_local_transform = get_world_to_local_transform(self.origin_world, self.frame)
        self.size = self.get_size()
        view_features = data.get("view_features") if self.frame_source == "provided" else None
        self.view_features = None if view_features is None else np.asarray(
            view_features, dtype=np.float32
        )
        if self.view_features is not None:
            if (self.view_features.ndim != 3 or self.view_features.shape[0] != 3
                    or self.view_features.shape[1] < 1
                    or not np.isfinite(self.view_features).all()):
                raise ValueError("view_features 必须是 [3,T,C] 的真实三视图编码")

    def get_margin_line(self, count: int = 128) -> np.ndarray | None:
        """优先读取真实边界；仅在设计冠恰好有一个闭合开放边界时自动提取。

        不从任意封闭冠上猜测颈缘线。无法可靠提取时返回 None，
        由上游标注真实颈缘线后再进入条件训练。
        """
        if count < 3:
            raise ValueError("边界采样至少需要 3 个点")
        explicit = self.data.get("margin_line", self.design_mesh.get("margin_line"))
        if explicit is not None:
            points = np.asarray(explicit, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
                raise ValueError("margin_line 必须为 [L,3]")
            if not np.isfinite(points).all():
                raise ValueError("margin_line 包含非有限值")
        else:
            if not len(self.design_faces):
                return None
            mesh = SindreMesh([self.design_vertices, self.design_faces])
            loops = mesh.get_boundary(return_points=True, max_boundary=False)
            if len(loops) != 1:
                return None
            points = np.asarray(loops[0], dtype=np.float64)

        # 闭合曲线等弧长重采样，保持真实非平面边界形态。
        if np.linalg.norm(points[0] - points[-1]) < 1e-6:
            points = points[:-1]
        following = np.roll(points, -1, axis=0)
        lengths = np.linalg.norm(following - points, axis=1)
        if lengths.sum() <= 1e-8:
            return None
        keep = lengths > 1e-8
        points, lengths = points[keep], lengths[keep]
        if len(points) < 3:
            return None
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        closed = np.concatenate([points, points[:1]], axis=0)
        arc = np.linspace(0, cumulative[-1], count, endpoint=False)
        return np.stack(
            [np.interp(arc, cumulative, closed[:, axis]) for axis in range(3)],
            axis=1,
        ).astype(np.float32)

    def get_size(self) -> np.ndarray | None:
        """优先使用显式尺寸，否则在设计冠局部坐标系中测量 MD/BL/H。"""
        value = self.data.get("size_mm", self.design_mesh.get("size_mm"))
        if value is None:
            # 推理时绝不通过整颌可见扫描的包围盒充当目标冠长宽高。
            if self.inference or self._views_builder is None:
                return None
            target = transform_points(self.design_vertices, self.world_to_local_transform)
            return np.array([np.ptp(target[:, 0]), np.ptp(target[:, 1]), target[:, 2].max()], dtype=np.float32)
        size = np.asarray(value, dtype=np.float32)
        if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("size_mm 必须为正的 [3]，顺序近远中/颊舌/高度")
        return size

    def get_condition(
        self,
        *,
        num_points: int = 4096,
        crop_mode: str = "random",
        tooth_dropout: float = 0.25,
        rng: random.Random | None = None,
        render_views: bool = False,
        image_encoder: Any | None = None,
        image_size: int = 256,
    ) -> dict[str, Any]:
        """按病例建立同颌牙弓上下文。每次调用都允许得到不同的裁剪结果。

        有同颌牙时最多输出 num_points 个真实采样点；若没有同颌牙，返回
        [0,6] 空数组并标记 jaw_present=False，不伪造邻牙。
        """
        if num_points < 1 or not (0 <= tooth_dropout <= 1):
            raise ValueError("num_points 必须为正，tooth_dropout 必须在 [0,1] 内")
        modes = ("full", "large", "medium", "local", "none", "random")
        if crop_mode not in modes:
            raise ValueError(f"crop_mode 必须属于 {modes}")
        rng = rng or random.Random()
        if crop_mode == "random":
            crop_mode = rng.choices(modes[:-1], weights=[15, 20, 25, 30, 10], k=1)[0]

        candidate = self.context_vertex_ids.copy()
        position = self.context_labels.copy()
        if crop_mode == "none":
            candidate = candidate[:0]
            position = position[:0]
        elif crop_mode != "full" and len(candidate):
            center = (
                self.margin_line.mean(axis=0)
                if self.margin_line is not None else self.design_vertices.mean(axis=0)
            )
            distances = np.linalg.norm(self.vertices[candidate] - center, axis=1)
            ranges = {"large": (0.65, 1.0), "medium": (0.35, 0.75), "local": (0.10, 0.45)}
            lower, upper = ranges[crop_mode]
            radius = float(np.quantile(distances, rng.uniform(lower, upper)))
            keep = distances <= radius
            candidate, position = candidate[keep], position[keep]

        if tooth_dropout > 0 and len(candidate):
            for tooth in np.unique(position[(position > 0) & (position <= 16)]):
                if rng.random() < tooth_dropout:
                    keep = position != tooth
                    candidate, position = candidate[keep], position[keep]

        if len(candidate):
            # 法向来自原网格，随机裁剪不破坏完整网格的表面法向。
            if self._cached_normals is None:
                sm = SindreMesh([self.vertices, self.faces])
                self._cached_normals = np.asarray(sm.vertex_normals, dtype=np.float32)
            normals = self._cached_normals
            if normals.shape != self.vertices.shape:
                raise ValueError("SindreMesh 未能计算逐顶点法向")
            if len(candidate) > num_points:
                ids = np.asarray(
                    sample_pcd_farthest(self.vertices[candidate], num_points), dtype=np.int64
                )
                if ids.shape != (num_points,) or (ids < 0).any() or (ids >= len(candidate)).any():
                    raise ValueError("SindrePy FPS 返回非法索引")
                candidate, position = candidate[ids], position[ids]
            jaw_points = np.concatenate(
                (self.vertices[candidate], normals[candidate]), axis=1
            ).astype(np.float32)
        else:
            jaw_points = np.empty((0, 6), dtype=np.float32)
            position = np.empty((0,), dtype=np.int64)

        # 同颌点云、颈缘线、监督目标使用 SindrePy 的同一刚体变换。
        # 若提供了视图特征，其相机也必须对应这个 local_frame。
        origin = self.origin_world.astype(np.float32)
        world_to_local = self.world_to_local_transform
        if len(jaw_points):
            cloud = SindrePCD({
                "points": jaw_points[:, :3],
                "point_normals": jaw_points[:, 3:],
            })
            cloud.apply_transform(world_to_local)
            jaw_points = np.concatenate(
                (cloud.points, cloud.point_normals), axis=1
            ).astype(np.float32)
        local_margin = (
            None if self.margin_line is None else
            transform_points(self.margin_line, world_to_local).astype(np.float32)
        )
        if self.inference:
            local_target = None
        else:
            target_mesh = SindreMesh([self.design_vertices, self.design_faces])
            target_mesh.apply_transform(world_to_local)
            local_target = np.asarray(target_mesh.vertices, dtype=np.float32)
        view_images = None
        view_features = self.view_features
        camera_info = None
        view_mask = np.array([self.opposite_vertices is not None, self.opposite_vertices is not None, True], dtype=bool)
        view_scale_mm = np.zeros(3, dtype=np.float32)
        if view_features is not None:
            if self.data.get("view_mask") is not None:
                cached_mask = np.asarray(self.data["view_mask"], dtype=bool)
                if cached_mask.shape != (3,):
                    raise ValueError("缓存 view_mask 必须为 [3]")
                view_mask &= cached_mask
            if self.data.get("view_scale_mm") is not None:
                view_scale_mm = np.asarray(self.data["view_scale_mm"], dtype=np.float32).copy()
                if view_scale_mm.shape != (3,) or not np.isfinite(view_scale_mm).all() or (view_scale_mm < 0).any():
                    raise ValueError("缓存 view_scale_mm 必须为非负有限 [3]")
        if render_views or image_encoder is not None:
            if self._views_builder is None:
                raise ValueError("缺少有效颈缘线或 local_frame，无法拍摄当前牙颌")
            if len(self.scene_faces) == 0:
                raise ValueError("current_mesh 中缺少可拍摄的同颌三角面，不能使用设计冠替代")

            # 优先复用原始数据中 2000+FDI 预备体顶点测距。
            # 若无明确的预备体标签，则尝试 sindre 的曲线裁剪；失败退回颈缘线范围。
            base_vertices = self.vertices[self.base_vertex_ids]
            if not len(base_vertices):
                base_vertices = self._views_builder.get_cut_base(
                    self.scene_vertices, self.scene_faces
                )
            camera_info = self._views_builder.get_camera_setup(
                base_vertices=base_vertices,
                opposite_vertices=self.opposite_vertices,
            )
            view_mask = np.array([self.opposite_vertices is not None, self.opposite_vertices is not None, True], dtype=bool)
            view_scale_mm = np.array([2 * camera_info[name]["parallel_scale"] for name in CrownViews.NAMES], dtype=np.float32)
            view_images = self._views_builder.get_images(
                image_size,
                jaw_vertices=self.scene_vertices,
                jaw_faces=self.scene_faces,
                opposite_vertices=self.opposite_vertices,
                opposite_faces=self.opposite_faces,
                base_vertices=base_vertices,
            )
            if image_encoder is not None:
                view_features = self._views_builder.encode(
                    image_encoder, images=view_images, view_mask=view_mask,
                )
            if not render_views:
                view_images = None

        view_scale_mm[~view_mask] = 0
        if view_features is not None:
            if not view_mask.any():
                raise ValueError("病例没有有效视图")
            view_features = view_features.copy()
            view_features[~view_mask] = 0
        return {
            "jaw": self.jaw,
            "opposite_jaw": self.opposite_jaw,
            "target_position": self.target_position,
            "jaw_points": jaw_points,
            "jaw_labels": position.astype(np.int64),
            "jaw_present": len(jaw_points) > 0,
            "margin_line": local_margin,
            "size": self.size,
            "view_features": view_features,
            "view_mask": view_mask,
            "view_scale_mm": view_scale_mm,
            "view_images": view_images,
            "view_source": "both_jaws_lateral_target_jaw_occlusal" if (
                image_encoder is not None or render_views
            ) else ("provided" if view_features is not None else None),
            "camera_info": camera_info,
            "base_present": len(self.base_vertex_ids) > 0,
            "opposite_present": self.opposite_vertices is not None,
            "target_vertices": local_target,
            "target_faces": self.design_faces,
            "source_fdi": self.target_fdi,
            "origin_world": origin,
            "local_frame": self.frame,
            "world_to_local_transform": world_to_local.astype(np.float32),
            "frame_verified": self.frame_verified,
            "frame_source": self.frame_source,
            "frame_confidence": self.frame_confidence,
            "labels_known": self.labels_known,
            "view_anatomical_verified": self.frame_verified,
            "crop_mode": crop_mode,
            "geometry_ready": self.margin_line is not None and (
                self.size is not None or self.inference
            ),
            "ready_for_training": (
                view_features is not None
                and self.margin_line is not None
                and (self.size is not None or self.inference)
            ),
        }


def process(
    data: dict[str, Any] | str | Path,
    *,
    sample_index: int | None = None,
    num_points: int = 4096,
    augment_index: int = 0,
    seed: int = 1024,
    crop_mode: str = "random",
    render_views: bool = False,
    image_encoder: Any | None = None,
    image_size: int = 256,
) -> dict[str, Any]:
    """读取 JSON 病例或 LMDB 指定记录并预处理。"""
    data = load_case_data(data, sample_index=sample_index)
    case = DentalCase(data)
    return case.get_condition(
        num_points=num_points,
        crop_mode=crop_mode,
        rng=random.Random(seed + augment_index * 7919),
        render_views=render_views,
        image_encoder=image_encoder,
        image_size=image_size,
    )



def collate_conditions(
    records: list[dict[str, Any]],
    *,
    num_points: int = 4096,
    seed: int = 1024,
) -> dict[str, Any]:
    """将可变长度牙弓上下文整理成 MCDGConditionEncoder 的 batch 输入。

    PTv3 当前封装不支持逐点 padding mask，因此对于非空但点数不足的牙弓，
    只重复已有的真实顶点，绝不补造不存在的牙齿；全空样本用 jaw_present 跳过。
    """
    if not records or num_points < 1:
        raise ValueError("records 不能为空，num_points 必须大于 0")
    missing = [i for i, record in enumerate(records) if not record["ready_for_training"]]
    if missing:
        raise ValueError(f"以下病例缺少真实三视图特征或颈缘线：{missing}")
    rng = np.random.default_rng(seed)
    points_list, labels_list, present_list = [], [], []
    for record in records:
        points = np.asarray(record["jaw_points"], dtype=np.float32)
        labels = np.asarray(record["jaw_labels"], dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 6 or labels.shape != (len(points),):
            raise ValueError("每份 jaw_points 必须为 [N,6]，jaw_labels 必须为 [N]")
        present = len(points) > 0 and bool(record["jaw_present"])
        if present:
            # 先随机选择真实顶点；不足时有放回采样，保持可变牙列数量兼容 PTv3。
            ids = rng.choice(len(points), size=num_points, replace=len(points) < num_points)
            points_list.append(points[ids])
            labels_list.append(labels[ids])
        else:
            points_list.append(np.zeros((num_points, 6), dtype=np.float32))
            labels_list.append(np.zeros((num_points,), dtype=np.int64))
        present_list.append(present)

    return {
        "view_features": torch.as_tensor(np.stack([
            record["view_features"] for record in records
        ]), dtype=torch.float32),
        "view_mask": torch.as_tensor(np.stack([record.get("view_mask", np.ones(3, bool)) for record in records]), dtype=torch.bool),
        "view_scale_mm": torch.as_tensor(np.stack([record.get("view_scale_mm", np.zeros(3, np.float32)) for record in records]), dtype=torch.float32),
        "jaw": torch.tensor([record["jaw"] for record in records], dtype=torch.long),
        "target_position": torch.tensor([
            record["target_position"] for record in records
        ], dtype=torch.long),
        "margin_line": torch.as_tensor(np.stack([
            record["margin_line"] for record in records
        ]), dtype=torch.float32),
        "size": torch.as_tensor(np.stack([
            record["size"] if record["size"] is not None else
            np.zeros(3, dtype=np.float32) for record in records
        ]), dtype=torch.float32),
        "size_known": torch.tensor([
            record["size"] is not None for record in records
        ], dtype=torch.bool),
        "jaw_points": torch.as_tensor(np.stack(points_list), dtype=torch.float32),
        "jaw_labels": torch.as_tensor(np.stack(labels_list), dtype=torch.long),
        "jaw_present": torch.tensor(present_list, dtype=torch.bool),
    }



def main(
    input_path: str | Path,
    *,
    sample_index: int | None = None,
    num_points: int = 4096,
    augment_count: int = 4,
    seed: int = 1024,
) -> None:
    """处理 JSON 病例或 LMDB 指定记录；不自动写入中间文件。"""
    if augment_count < 1:
        raise ValueError("augment_count 必须 >= 1")
    data = load_case_data(input_path, sample_index=sample_index)
    case = DentalCase(data)
    for augment_index in range(augment_count):
        record = case.get_condition(
            num_points=num_points,
            rng=random.Random(seed + augment_index * 7919),
            crop_mode="full" if augment_index == 0 else "random",
        )
        print(
            f"index={augment_index} target={record['source_fdi']} "
            f"crop={record['crop_mode']} points={len(record['jaw_points'])} "
            f"margin={record['margin_line'] is not None} size={record['size'] is not None}"
        )


class FDGPreprocessor:
    """管理一次性病例预处理、配对增强和数据库写入。

    Args:
        converter: 已配置的 CrownFDG 原生编解码器。
        case_class: 可注入病例解析器，便于独立测试。
    """

    def __init__(self, converter: CrownFDG, *, case_class=None):
        self.converter = converter
        self.case_class = DentalCase if case_class is None else case_class

    def build_database(
        self, input_db: str | Path, output_db: str | Path, *,
        max_cases: int | None = None, map_size_mb: int = 32768,
        image_encoder: object | None = None,
        num_jaw_points: int = 4096, image_size: int = 256,
        augment_variants: int = 4, augment_scale: float = 0.10,
        collision_check: bool = True,
        max_collision_attempts: int = 12,
        seed: int = 1024, strict: bool = True,
    ) -> dict:
        """将原始牙颌病例和成对尺寸增强写入 FDG v3 LMDB。

        在预处理阶段完成三视图 DINOv3、开放冠裁剪、FDG 网格编码。
        同一原始病例的尺寸副本使用相同 source_group，防止验证泄漏。

        Args:
            input_db: 原始完整病例 LMDB（不支持训练标签库）。
            output_db: 全新 FDG LMDB 目标路径，不允许覆盖原数据。
            max_cases: 最多转换的原始病例数；None 表示全部。
            map_size_mb: Sindre LMDB 映射文件容量，单位 MB。
            image_encoder: 可选已加载的 SindrePy DINOv3Encoder。
            num_jaw_points: 预处理的同颌点云采样数。
            image_size: 预处理三视图的像素分辨率。
            augment_variants: 每份原始冠生成的配对尺寸样本数。
            augment_scale: MD/BL/H 三轴独立的比例变化幅度。
            collision_check: 是否使用 SindrePy MeshLib 检查新增邻牙/对颌相交。
                默认开启；关闭仅允许研究用数据，不能视为临床合格。
            max_collision_attempts: 每个增强版本最多随机尝试的数量。
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
        if not isinstance(collision_check, bool) or max_collision_attempts < 1:
            raise ValueError("collision_check 必须是 bool，尝试次数必须 >=1")
        from sindre.lmdb import Reader, Writer

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "total": 0, "written": 0, "errors": [],
            "rejected_collision": 0, "skipped_augmentation": 0,
            "warnings": [],
        }
        with Reader(str(input_path)) as reader:
            source_schema = reader.get_meta("schema_version")
            if source_schema is not None and str(source_schema).startswith("toothforge3d_"):
                raise ValueError("只接受原始牙颌病例 LMDB，拒绝任何训练标签数据库")
            writer = Writer(str(output_path), map_size_limit=map_size_mb)
            try:
                writer.put_meta("schema_version", "toothforge3d_fdg_v3")
                writer.put_meta("build_complete", False)
                writer.put_meta("fdg_config", asdict(self.converter.config))
                writer.put_meta("coordinate_frame", "scan_arch_margin_v3")
                writer.put_meta("augmentation", {
                    "variants_per_case": augment_variants,
                    "size_scale": augment_scale,
                    "margin_fixed": True,
                    "collision_check": collision_check,
                    "max_collision_attempts": max_collision_attempts,
                    "collision_backend": "SindrePy AdjacentCollisionDetector MeshLib",
                    "penetration_depth_verified": False,
                })
                source_groups = []
                source_order_groups = []
                count = min(len(reader), max_cases) if max_cases else len(reader)
                for index in range(count):
                    report["total"] += 1
                    case_records = []
                    try:
                        raw = reader[index]
                        order = raw.get("order_name") if isinstance(raw, dict) else None
                        order_key = hashlib.sha256(str(normalize_record(order)).encode("utf-8")).hexdigest() if order is not None and str(order).strip() else ""
                        group = -1 - int(order_key[:15], 16) if order_key else index
                        case = self.case_class(raw)
                        sample = case.get_condition(
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
                        # 正式条件的方向只能来自可观察扫描或明确外部输入。
                        _validate_training_frame(
                            str(sample.get("frame_source", ""))
                        )
                        open_vertices, open_faces = self.converter.get_open_crown(
                            sample["target_vertices"],
                            sample["target_faces"],
                            sample["margin_line"],
                        )
                        self.converter.get_mesh_quality(open_vertices, open_faces)
                        base_size = np.array([np.ptp(open_vertices[:, 0]), np.ptp(open_vertices[:, 1]),
                                              open_vertices[:, 2].max()], dtype=np.float32)
                        if (base_size <= 0).any():
                            raise ValueError("开放目标冠尺寸无效")
                        rng = np.random.default_rng(seed + index * 7919)
                        collision_env = {}
                        baseline_collision = {}
                        if augment_variants > 1 and collision_check:
                            collision_env = self.converter.get_environment_surfaces(
                                case, open_vertices,
                            )
                            baseline_collision = self.converter.get_collision_status(
                                open_vertices, open_faces, collision_env,
                            )
                            if any(
                                item["collision"] or item.get("within_clearance", False)
                                for item in baseline_collision.values()
                            ):
                                report["warnings"].append({
                                    "index": index,
                                    "reason": "原始设计冠已接触/相交或间隙不足 0.05mm；"
                                              "开放 Mesh 无法确定是否加重穿透，跳过尺寸增强",
                                })
                        for variant in range(augment_variants):
                            variant_vertices = open_vertices
                            variant_size = base_size
                            if variant:
                                if collision_check and any(
                                    item["collision"] or item.get("within_clearance", False)
                                    for item in baseline_collision.values()
                                ):
                                    report["skipped_augmentation"] += 1
                                    continue
                                valid = False
                                for _ in range(max_collision_attempts):
                                    factors = tuple(rng.uniform(
                                        1 - augment_scale, 1 + augment_scale, size=3
                                    ).tolist())
                                    candidate_vertices, candidate_size = self.converter.augment_crown_size(
                                        open_vertices, open_faces, sample["margin_line"],
                                        base_size, factors,
                                    )
                                    if collision_check:
                                        try:
                                            self.converter.get_mesh_quality(candidate_vertices, open_faces)
                                            status = self.converter.get_collision_status(
                                                candidate_vertices, open_faces,
                                                collision_env,
                                            )
                                        except ValueError:
                                            # 尺寸变形引起的自交、非流形或退化曲面，
                                            # 不能当作可用样本；仅拒绝当前候选并重试。
                                            report["rejected_collision"] += 1
                                            continue
                                        if any(
                                            item["collision"] or item.get("within_clearance", False)
                                            for item in status.values()
                                        ):
                                            report["rejected_collision"] += 1
                                            continue
                                    variant_vertices, variant_size = (
                                        candidate_vertices, candidate_size
                                    )
                                    valid = True
                                    break
                                if not valid:
                                    report["skipped_augmentation"] += 1
                                    continue
                            converted = self.converter.get_fdg(
                                variant_vertices, open_faces, sample["margin_line"],
                                cut_closed=False,
                            )
                            if DEBUG:
                                self.converter.show(
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
                            record["view_mask"] = sample.get("view_mask", np.ones(3, dtype=bool))
                            record["view_scale_mm"] = sample.get("view_scale_mm", np.zeros(3, dtype=np.float32))
                            record["source_group"] = group
                            record["variant_index"] = variant
                            for key in (
                                "origin_world", "local_frame",
                                "world_to_local_transform", "frame_source", "frame_confidence", "labels_known",
                            ):
                                if key in sample and sample[key] is not None:
                                    record[key] = sample[key]
                            if "source_fdi" in sample:
                                record["source_fdi"] = sample["source_fdi"]
                            record.update({
                                f"fdg_{key}": value for key, value in converted.items()
                            })
                            case_records.append(record)

                    except Exception as exc:
                        report["errors"].append({"index": index, "error": str(exc)})
                        log.exception("FDG 样本 %d 转换失败", index)
                        if strict or not isinstance(exc, ValueError):
                            raise
                        continue
                    # 同一病例所有有效变体一起写入；数据库写入异常必须中止。
                    for record in case_records:
                        writer.put_sample(record)
                        report["written"] += 1
                        source_groups.append(group)
                        source_order_groups.append(order_key)
                if not report["written"]:
                    raise ValueError("没有成功构建的 FDG 样本")
                writer.put_meta("written", report["written"])
                writer.put_meta("source_groups", source_groups)
                writer.put_meta("source_order_groups", source_order_groups)
                writer.put_meta("build_complete", True)
            finally:
                writer.close()
                output_path.with_suffix(".report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return report


def build_training_database(
    input_db: str | Path,
    output_db: str | Path,
    *,
    resolution: int = 128,
    max_cases: int | None = None,
    image_encoder: Any | None = None,
    augment_variants: int = 1,
    collision_check: bool = True,
) -> dict[str, Any]:
    """将原始 LMDB 转换为 train.train 可直接读取的 FDG v3 数据库。"""
    preprocessor = FDGPreprocessor(CrownFDG(FDGConfig(resolution=resolution)))
    build_options = {
        "max_cases": max_cases,
        "augment_variants": augment_variants,
        "collision_check": collision_check,
    }
    if image_encoder is not None:
        report = preprocessor.build_database(
            input_db, output_db, image_encoder=image_encoder, **build_options
        )
    else:
        with DINOv3Encoder() as encoder:
            report = preprocessor.build_database(
                input_db, output_db, image_encoder=encoder, **build_options
            )
    if DEBUG and report["written"]:
        debug_render_saved_database(output_db, sample_index=DEBUG_SAMPLE_INDEX)
        sys.exit()
    return report


def debug_render_saved_database(
    db_path: str | Path,
    *,
    sample_index: int = 0,
) -> None:
    """从已写入的 FDG 数据库读取并叠加渲染冠、邻牙点云和颈缘线。"""
    with Reader(str(db_path)) as reader:
        if len(reader) == 0:
            complete = reader.get_meta("build_complete")
            raise ValueError(
                f"训练数据库没有样本，build_complete={complete!r}: {db_path}"
            )
        if sample_index < 0 or sample_index >= len(reader):
            raise IndexError(
                f"sample_index={sample_index} 超出数据库范围 [0, {len(reader)})"
            )
        row = reader[sample_index]

    target_vertices = np.asarray(row["target_vertices"], dtype=np.float64)
    target_faces = np.asarray(row["target_faces"], dtype=np.int64)
    margin_line = np.asarray(row["margin_line"], dtype=np.float64)
    jaw_points = np.asarray(row["jaw_points"], dtype=np.float64)
    jaw_labels = np.asarray(row["jaw_labels"], dtype=np.int64)
    if (jaw_points.ndim != 2 or jaw_points.shape[1] != 6
            or jaw_labels.shape != (len(jaw_points),)):
        raise ValueError("已保存的 jaw_points / jaw_labels 形状无效")
    if margin_line.ndim != 2 or margin_line.shape[1] != 3 or len(margin_line) < 3:
        raise ValueError("已保存的 margin_line 形状无效")

    actors = []
    if len(jaw_points):
        jaw_cloud = SindrePCD({
            "points": jaw_points[:, :3],
            "point_normals": jaw_points[:, 3:],
            "point_labels": jaw_labels,
        })
        actors.extend(jaw_cloud.show(return_obj=True, create_axes=False))
    margin_cloud = SindrePCD({
        "points": margin_line,
        "point_colors": np.tile(
            np.array([[255, 40, 40]], dtype=np.uint8), (len(margin_line), 1)
        ),
    })
    actors.extend(
        margin_cloud.show(return_obj=True, create_axes=False, show_flags=False)
    )
    target_mesh = SindreMesh([target_vertices, target_faces])
    target_mesh.show(show_append=actors, create_axes=True)
    print(
        f"DEBUG saved sample={sample_index}, "
        f"FDI={row.get('source_fdi')}, frame={row.get('frame_source')}, "
        f"mesh={target_vertices.shape[0]} vertices/{target_faces.shape[0]} faces, "
        f"jaw_points={len(jaw_points)}, margin_points={len(margin_line)}"
    )


if __name__ == "__main__":
    source_db = Path("datasets/全牙_5.db")
    training_db = Path("datasets/toothforge_fdg_v2.db")
    if DEBUG and training_db.exists():
        debug_render_saved_database(
            training_db, sample_index=DEBUG_SAMPLE_INDEX
        )
    else:
        print(build_training_database(source_db, training_db))
