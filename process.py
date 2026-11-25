"""全冠设计数据预处理：同颌随机裁剪、逐点牙位、可选颈缘线。

输入为 current_mesh、opposite_mesh、design_mesh；FDI 确定目标颌。
设计冠仅辅助方向和监督，不参与渲染。颊舌侧渲染上下颌，咬合面只拍目标颌；
对颌不输入 PTv3 点云条件。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

# 与 SindrePy 的 FDI→1~16 规则保持一致。注意下颌 31→1，41→9。
UPPER_FDI = tuple(range(11, 19)) + tuple(range(21, 29))
LOWER_FDI = tuple(range(31, 39)) + tuple(range(41, 49))


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
        if not isinstance(data, dict) or "current_mesh" not in data:
            raise ValueError("病例必须含有 current_mesh")
        self.inference = inference
        if inference:
            if "design_mesh" in data:
                raise ValueError("推理病例不得携带 design_mesh")
            for key in ("target_fdi", "margin_line", "local_frame"):
                if data.get(key) is None:
                    raise ValueError(f"推理病例缺少 {key}")
            raw_fdi = data["target_fdi"]
            design_mesh = {}
        else:
            design_mesh = data.get("design_mesh")
            if not isinstance(design_mesh, dict):
                raise ValueError("训练病例必须提供规范的 design_mesh")
            raw_fdi = design_mesh.get("fdi")
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
        from sindre.utils3d.core.dental.utils import convert_fdi2idx

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

        labels = np.asarray(self.current_mesh["vertex_labels"], dtype=np.int64).reshape(-1)
        if len(labels) != len(self.vertices):
            raise ValueError("vertex_labels 必须与 current_mesh.vertices 一一对应")
        label_table = self.current_mesh.get("fdi")
        if label_table is not None:
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
        render_face_mask = np.all(scene_keep[self.faces], axis=1)
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
        self.context_vertex_ids = np.flatnonzero(allowed)
        valid_labels = labels[self.context_vertex_ids].copy()
        self.context_labels = np.asarray(convert_fdi2idx(valid_labels), dtype=np.int64)
        if not np.isin(self.context_labels, range(17)).all():
            raise ValueError("同颌牙/背景标签转换到 0~16 失败")

        self.margin_line = self.get_margin_line()
        self._views_builder = None
        if self.margin_line is not None or data.get("local_frame") is not None:
            from views import CrownViews

            self._views_builder = CrownViews(
                self.scene_vertices if inference else self.design_vertices,
                self.scene_faces if inference else self.design_faces,
                self.margin_line,
                jaw_points=self.vertices[self.context_vertex_ids],
                jaw_labels=self.context_labels,
                target_position=self.target_position,
                local_frame=data.get("local_frame"),
            )
            self.frame = self._views_builder.frame
            self.origin_world = self._views_builder.origin.astype(np.float32)
            self.frame_verified = self._views_builder.anatomical_verified
            self.frame_source = self._views_builder.frame_source
        else:
            # 封闭 Mesh 无真实颈缘线时不能可靠判别冠向，禁止伪造标准三视图。
            self.frame = np.eye(3, dtype=np.float32)
            self.origin_world = (
                self.margin_line.mean(axis=0) if self.margin_line is not None
                else self.design_vertices.mean(axis=0)
            )
            self.frame_verified = False
            self.frame_source = "unresolved"
        self.size = self.get_size()
        view_features = data.get("view_features")
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
            faces = self.design_faces
            edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            unique, counts = np.unique(
                np.sort(edges, axis=1), axis=0, return_counts=True
            )
            boundary_edges = unique[counts == 1]
            if len(boundary_edges) < 3:
                return None
            links: dict[int, list[int]] = {}
            for a, b in boundary_edges:
                links.setdefault(int(a), []).append(int(b))
                links.setdefault(int(b), []).append(int(a))
            if any(len(neighbors) != 2 for neighbors in links.values()):
                return None
            first = next(iter(links))
            order = [first]
            previous, current = -1, first
            for _ in range(len(links)):
                next_candidates = [n for n in links[current] if n != previous]
                if previous != -1 and len(next_candidates) != 1:
                    return None
                nxt = next_candidates[0]
                if nxt == first:
                    break
                if nxt in order:
                    return None
                order.append(nxt)
                previous, current = current, nxt
            if len(order) != len(links):
                return None  # 多个边界环，不能猜哪个是颈缘线
            points = self.design_vertices[order].astype(np.float64)

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
            return None if self.inference or self._views_builder is None else self._views_builder.get_size()
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
            for tooth in np.unique(position[position > 0]):
                if rng.random() < tooth_dropout:
                    keep = position != tooth
                    candidate, position = candidate[keep], position[keep]

        if len(candidate):
            # 法向来自原网格，随机裁剪不破坏完整网格的表面法向。
            from sindre.utils3d.sindremesh import SindreMesh
            from sindre.utils3d.core.sample import sample_pcd_farthest

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

        # 同颌点云、颈缘线、监督目标使用同一个局部坐标变换。
        # 若提供了视图特征，其相机也必须对应这个 local_frame。
        origin = self.origin_world.astype(np.float32)
        if len(jaw_points):
            jaw_points[:, :3] = (jaw_points[:, :3] - origin) @ self.frame
            jaw_points[:, 3:] = jaw_points[:, 3:] @ self.frame
        local_margin = (
            None if self.margin_line is None else
            ((self.margin_line - origin) @ self.frame).astype(np.float32)
        )
        local_target = (
            None if self.inference
            else ((self.design_vertices - origin) @ self.frame).astype(np.float32)
        )
        view_images = None
        view_features = self.view_features
        camera_info = None
        if render_views or image_encoder is not None:
            if self._views_builder is None:
                raise ValueError("缺少有效颈缘线或 local_frame，无法拍摄当前牙颌")
            if len(self.scene_faces) == 0:
                raise ValueError("current_mesh 中缺少可拍摄的同颌三角面，不能使用设计冠替代")
            if self.opposite_vertices is None:
                raise ValueError(
                    "拍摄颊侧/舌侧上下颌视图需要 opposite_mesh；"
                    "不能用单颌图像冒充双颌视图"
                )

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
                    image_encoder, images=view_images,
                )
            if not render_views:
                view_images = None

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
            "frame_verified": self.frame_verified,
            "frame_source": self.frame_source,
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
    num_points: int = 4096,
    augment_index: int = 0,
    seed: int = 1024,
    crop_mode: str = "random",
    render_views: bool = False,
    image_encoder: Any | None = None,
    image_size: int = 256,
) -> dict[str, Any]:
    """读取单份病例并预处理；不使用 argparse，也不混用提示分割数据结构。"""
    if isinstance(data, (str, Path)):
        with Path(data).open("r", encoding="utf-8") as file:
            data = json.load(file)
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
    import torch

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
    num_points: int = 4096,
    augment_count: int = 4,
    seed: int = 1024,
) -> None:
    """显式参数入口。仅检查数据，不自动写入大量不可复现的中间文件。"""
    if augment_count < 1:
        raise ValueError("augment_count 必须 >= 1")
    with Path(input_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
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


if __name__ == "__main__":
    raise SystemExit("请调用 process.main(完整原始病例JSON路径)")
