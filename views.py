"""真实设计牙冠的局部坐标系与标准三视图构建。

本模块只渲染 *输入的设计冠*。目标设计冠产生的图像用于
image-conditioned reconstruction；若要训练无目标参考的新冠设计，
应改用独立参考牙图，避免把监督答案当成推理时可获得的条件。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class CrownViews:
    """从单冠 Mesh + 颈缘线建立共享坐标与三视图。

    local_frame 的列依次是 MD、BL、crown 方向的世界坐标单位向量。
    当同颌几何不足以确定牙弓外侧时，可以渲染两个相对侧面，
    但 anatomical_verified=False，此时不应把两面当作已经确认的颊/舌面。
    """

    NAMES = ("buccal", "lingual", "occlusal")

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        margin_line: np.ndarray | None = None,
        *,
        jaw_points: np.ndarray | None = None,
        jaw_labels: np.ndarray | None = None,
        target_position: int | None = None,
        local_frame: np.ndarray | None = None,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3:
            raise ValueError("vertices 必须是 [N,3]，至少三个顶点")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or len(self.faces) < 1:
            raise ValueError("faces 必须是 [F,3]")
        if not np.isfinite(self.vertices).all() or (self.faces < 0).any() or (self.faces >= len(self.vertices)).any():
            raise ValueError("Mesh 含无效坐标或三角面索引")

        self.margin = None if margin_line is None else np.asarray(margin_line, dtype=np.float64)
        if self.margin is not None and (
            self.margin.ndim != 2 or self.margin.shape[1] != 3
            or len(self.margin) < 3 or not np.isfinite(self.margin).all()
        ):
            raise ValueError("margin_line 必须是有限的 [L,3]")

        self.origin = (
            self.margin.mean(axis=0) if self.margin is not None
            else self.vertices.mean(axis=0)
        )
        if local_frame is not None:
            frame = np.asarray(local_frame, dtype=np.float64)
            if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                    or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(frame), 1.0, atol=1e-3)):
                raise ValueError("local_frame 必须是有效的 3×3 右手正交矩阵")
            self.frame = frame.astype(np.float32)
            self.anatomical_verified = True
            self.axis_verified = True
            self.frame_source = "provided"
            return

        if self.margin is None:
            raise ValueError("自动建立三视图坐标系需要唯一的冠底边界；否则须提供 local_frame")
        # SindrePy 返回根向（冠顶 -> 边界），三视图的 Z 需要反向指向牙尖。
        from sindre.utils3d.dental_algorithms.dental_tools import estimate_tooth_insertion_direction
        from sindre.utils3d.sindremesh import SindreMesh

        mesh = SindreMesh([self.vertices, self.faces])
        z = -np.asarray(
            estimate_tooth_insertion_direction(mesh, boundary_points=self.margin),
            dtype=np.float64,
        )
        z /= np.linalg.norm(z)

        # 优先利用同侧已标注牙的近远中相对位置确定切向。
        x = None
        if jaw_points is not None and jaw_labels is not None and target_position is not None:
            points = np.asarray(jaw_points, dtype=np.float64)
            labels = np.asarray(jaw_labels, dtype=np.int64)
            if points.ndim != 2 or points.shape[1] < 3 or labels.shape != (len(points),):
                raise ValueError("jaw_points / jaw_labels 的数量和维度不匹配")
            target_half = (target_position - 1) // 8
            ordinal = (target_position - 1) % 8 + 1
            candidate = []
            for neighbor in sorted(set(labels.tolist()) - {0, int(target_position)}):
                if not 1 <= neighbor <= 16 or (neighbor - 1) // 8 != target_half:
                    continue
                n_ordinal = (neighbor - 1) % 8 + 1
                center = points[labels == neighbor, :3].mean(axis=0)
                # 近中是较小序数，远中是较大序数。
                oriented = (center - self.origin) * (1 if n_ordinal < ordinal else -1)
                candidate.append((abs(n_ordinal - ordinal), oriented))
            for _, direction in sorted(candidate, key=lambda it: it[0]):
                tangent = direction - np.dot(direction, z) * z
                if np.linalg.norm(tangent) > 1e-5:
                    x = tangent / np.linalg.norm(tangent)
                    break

        if x is None:
            # 单冠也能确定两条互相正交的侧向轴，但无法仅凭 PCA 确认颊/舌。
            centered = self.vertices - self.vertices.mean(axis=0)
            tangent = centered - np.outer(centered @ z, z)
            values, vectors = np.linalg.eigh(tangent.T @ tangent)
            x = vectors[:, np.argmax(values)]
            x = x - np.dot(x, z) * z
            if np.linalg.norm(x) < 1e-8:
                raise ValueError("牙冠在垂直于长轴的平面内退化")
            x /= np.linalg.norm(x)
            if x[np.argmax(abs(x))] < 0:
                x = -x

        y = np.cross(z, x)
        y /= np.linalg.norm(y)

        verified = False
        if jaw_points is not None and jaw_labels is not None:
            labels = np.asarray(jaw_labels, dtype=np.int64)
            points = np.asarray(jaw_points, dtype=np.float64)
            # 有两侧足够多的牙弓上下文，才尝试用牙弓中心定位颊侧。
            present = sorted(set(labels.tolist()) - {0})
            both_sides = any(1 <= p <= 8 for p in present) and any(9 <= p <= 16 for p in present)
            if len(present) >= 4 and both_sides:
                centers = np.array([points[labels == p, :3].mean(axis=0) for p in present])
                radial = self.origin - centers.mean(axis=0)
                radial -= np.dot(radial, z) * z
                align = float(np.dot(radial, y))
                if abs(align) > 0.2 * np.linalg.norm(radial):
                    if align < 0:
                        # 同时反转侧向两轴以保持右手坐标系。
                        x, y = -x, -y
                    verified = True

        self.frame = np.column_stack((x, y, z)).astype(np.float32)
        self.axis_verified = True
        self.anatomical_verified = verified
        self.frame_source = "design_crown+same_jaw" if verified else "design_crown_pca"

    def get_local_vertices(self) -> np.ndarray:
        """返回同一世界原点与坐标基下的目标牙冠（mm）。"""
        return ((self.vertices - self.origin) @ self.frame).astype(np.float32)

    def get_size(self) -> np.ndarray:
        """在冠体坐标系中计算 MD、BL、从边界中心到冠顶的尺寸（mm）。"""
        vertices = self.get_local_vertices()
        width = np.ptp(vertices[:, 0])
        thickness = np.ptp(vertices[:, 1])
        height = vertices[:, 2].max()
        size = np.array([width, thickness, height], dtype=np.float32)
        if not np.isfinite(size).all() or np.any(size <= 1e-6):
            raise ValueError("牙冠尺寸退化，无法构造条件大小")
        return size

    def get_images(self, image_size: int = 256) -> np.ndarray:
        """固定正交相机渲染，返回 [3,H,W,3] 的 uint8 RGB 图像。

        三个相机使用相同焦点、正交缩放、材质和背景，便于 DINOv3
        学习一致的几何视角。调用时需要安装 vedo 与屏幕外 VTK 渲染环境。
        """
        if image_size < 64:
            raise ValueError("image_size 不能小于 64")
        import vedo
        from PIL import Image

        vertices = self.get_local_vertices()
        center = (vertices.min(0) + vertices.max(0)) / 2.0
        span = float(np.max(vertices.max(0) - vertices.min(0)))
        distance = max(span * 4.0, 10.0)
        scale = span * 0.75
        # 方向在局部坐标：颊侧 +Y、舌侧 -Y、咬合面 +Z。
        directions = (
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        )
        images = []
        with TemporaryDirectory(prefix="toothforge_views_") as directory:
            for i, (direction, up) in enumerate(directions):
                output = Path(directory) / f"{i:02d}_{self.NAMES[i]}.png"
                actor = vedo.Mesh([vertices, self.faces]).compute_normals(
                    points=True, cells=False
                ).c("#e5e9ef").lighting("glossy")
                plotter = vedo.Plotter(
                    offscreen=True, size=(image_size, image_size), bg="#ffffff"
                )
                try:
                    plotter.show(actor, resetcam=True, axes=0, interactive=False)
                    camera = plotter.camera
                    camera.SetFocalPoint(*center)
                    camera.SetPosition(*(center + direction * distance))
                    camera.SetViewUp(*up)
                    camera.ParallelProjectionOn()
                    camera.SetParallelScale(scale)
                    plotter.renderer.ResetCameraClippingRange()
                    plotter.render()
                    plotter.screenshot(str(output))
                    with Image.open(output) as pic:
                        images.append(np.asarray(pic.convert("RGB")).copy())
                finally:
                    plotter.close()
        return np.stack(images, axis=0)

    def encode(self, encoder: object, image_size: int = 256) -> np.ndarray:
        """复用 SindrePy DINOv3Encoder.encode(return_tokens=True)。"""
        images = self.get_images(image_size)
        features = [
            np.asarray(encoder.encode(image, return_tokens=True), dtype=np.float32)
            for image in images
        ]
        if len({tuple(feat.shape) for feat in features}) != 1:
            raise ValueError("三个视角的 DINOv3 token 形状不一致")
        return np.stack(features, axis=0)
