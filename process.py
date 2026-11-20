"""全冠设计数据预处理：同颌随机裁剪、逐点牙位、可选颈缘线。

原始数据为 current_mesh / design_mesh / is_upper，而非提示分牙 LMDB。
只生成真实可观察的牙弓上下文；不会从目标设计冠复制几何到输入牙弓。
三视图必须由同一坐标系下的参考牙渲染或从缓存读取，本模块不伪造图像。
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
    """一份真实全冠病例，负责标签映射、边界提取及按需随机裁剪。

    current_mesh 顶点标签可以是原始 FDI，也可以是其 fdi 字典的索引。
    只有符合本颌 FDI 且不是 target_fdi 的牙列点才作为生成上下文。
    预备体、未识别标签和跨颌牙齿不直接混入同颌牙位条件。
    """

    def __init__(self, data: dict[str, Any]):
        if not isinstance(data, dict) or "current_mesh" not in data or "design_mesh" not in data:
            raise ValueError("病例必须含有 current_mesh 和 design_mesh")
        if not isinstance(data.get("is_upper"), (bool, np.bool_)):
            raise ValueError("病例必须明确 is_upper 布尔值")
        self.data = data
        self.jaw = 0 if data["is_upper"] else 1
        self.valid_fdi = UPPER_FDI if self.jaw == 0 else LOWER_FDI
        self.target_fdi = int(data["design_mesh"]["fdi"])
        if self.target_fdi not in self.valid_fdi:
            raise ValueError(f"目标牙位 {self.target_fdi} 与当前颌不匹配")

        # SindrePy 已有 FDI 转换函数，这里只处理被验证过的真实 FDI。
        from sindre.utils3d.dental_algorithms.dental_tools import convert_fdi2idx

        self.target_position = int(convert_fdi2idx(np.array([self.target_fdi]))[0])
        self.current_mesh = data["current_mesh"]
        self.design_mesh = data["design_mesh"]
        self.vertices = np.asarray(self.current_mesh["vertices"], dtype=np.float32)
        self.faces = np.asarray(self.current_mesh["faces"], dtype=np.int64)
        self.design_vertices = np.asarray(self.design_mesh["vertices"], dtype=np.float32)
        self.design_faces = np.asarray(self.design_mesh["faces"], dtype=np.int64)
        for name, vertices, faces in (
            ("current_mesh", self.vertices, self.faces),
            ("design_mesh", self.design_vertices, self.design_faces),
        ):
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

        # 先过滤掉目标牙及未知特殊标签，避免将训练目标泄漏给网络。
        allowed = np.isin(labels, self.valid_fdi) & (labels != self.target_fdi)
        self.context_vertex_ids = np.flatnonzero(allowed)
        valid_labels = labels[self.context_vertex_ids].copy()
        self.context_labels = np.asarray(convert_fdi2idx(valid_labels), dtype=np.int64)
        if not np.isin(self.context_labels, range(1, 17)).all():
            raise ValueError("同颌牙 FDI 转换到 1~16 失败")

        self.margin_line = self.get_margin_line()
        self.size = self.get_size()

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
        """只接受明确标注的 MD/BL/H 毫米尺寸，不把全局 AABB 假称为牙体尺寸。"""
        value = self.data.get("size_mm", self.design_mesh.get("size_mm"))
        if value is None:
            return None
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
            for tooth in np.unique(position):
                if rng.random() < tooth_dropout:
                    keep = position != tooth
                    candidate, position = candidate[keep], position[keep]

        if len(candidate):
            # 法向来自原网格，随机裁剪不破坏完整网格的表面法向。
            from sindre.utils3d.sindremesh import SindreMesh
            from sindre.utils3d.sample import sample_pcd_farthest

            sm = SindreMesh([self.vertices, self.faces])
            normals = np.asarray(sm.vertex_normals, dtype=np.float32)
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

        return {
            "jaw": self.jaw,
            "target_position": self.target_position,
            "jaw_points": jaw_points,
            "jaw_labels": position.astype(np.int64),
            "jaw_present": len(jaw_points) > 0,
            "margin_line": self.margin_line,
            "size": self.size,
            "target_vertices": self.design_vertices,
            "target_faces": self.design_faces,
            "source_fdi": self.target_fdi,
            "crop_mode": crop_mode,
            "ready_for_training": self.margin_line is not None and self.size is not None,
        }


def process(
    data: dict[str, Any] | str | Path,
    *,
    num_points: int = 4096,
    augment_index: int = 0,
    seed: int = 1024,
    crop_mode: str = "random",
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
    )


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
    for augment_index in range(augment_count):
        record = process(
            input_path, num_points=num_points, augment_index=augment_index,
            seed=seed, crop_mode="full" if augment_index == 0 else "random",
        )
        print(
            f"index={augment_index} target={record['source_fdi']} "
            f"crop={record['crop_mode']} points={len(record['jaw_points'])} "
            f"margin={record['margin_line'] is not None} size={record['size'] is not None}"
        )


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "datasets" / "0.json")
