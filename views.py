"""使用设计冠推算牙齿方向，随后仅渲染当前牙颌的标准三视图。

设计冠仅参与坐标系估计和尺寸监督，不允许出现在输入图像中。
相机目标固定为颈缘线中心，距离/视场依据裁剪基座或颈缘线范围确定。
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

    def get_cut_base(self, jaw_vertices: np.ndarray, jaw_faces: np.ndarray) -> np.ndarray | None:
        """用 SindrePy 的 MeshLib 实现沿颈缘线裁剪，选根向基座。

        不按网格面积判断两侧。失败时返回 None；相机自动退回颈缘线
        尺度，不能使用 design_mesh 的外轮廓来计算拍摄距离。
        """
        if self.margin is None:
            return None
        vertices = np.asarray(jaw_vertices, dtype=np.float64)
        faces = np.asarray(jaw_faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            return None
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            return None

        import vedo
        from sindre.utils3d.algorithm import cut_mesh_by_meshlib

        try:
            # 先选取最靠近颈缘线的连通区域，避免对其他牙或远处牙龈切割。
            regions = vedo.Mesh([vertices, faces]).split()
            if not regions:
                return None
            region = min(
                regions,
                key=lambda item: float(np.linalg.norm(
                    np.asarray(item.vertices, dtype=np.float64) - self.origin,
                    axis=1,
                ).min()),
            )
            parts = cut_mesh_by_meshlib(
                np.asarray(region.vertices, dtype=np.float64),
                np.asarray(region.cells, dtype=np.int64),
                self.margin, get_bigger_part=False, smooth_boundary=False,
            )
            margin_span = max(float(np.ptp(
                (self.margin - self.origin) @ self.frame, axis=0,
            ).max()), 1e-6)
            candidates = []
            for part_v, part_f in ((parts[0], parts[1]), (parts[2], parts[3])):
                points = np.asarray(part_v, dtype=np.float64)
                cells = np.asarray(part_f, dtype=np.int64)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or len(cells) == 0:
                    continue
                local = (points - self.origin) @ self.frame
                span = float(np.ptp(local, axis=0).max())
                rootward = float(np.median(local[:, 2]))
                if rootward >= -1e-4 or span <= 0 or span > margin_span * 4.0:
                    continue
                # 根侧候选优先；以与颈缘线中心的距离和局部尺度选取，
                # 不根据“较大/较小面积”决定基座。
                score = (
                    max(float(np.linalg.norm(local.mean(axis=0))), 1e-6)
                    + 0.1 * span
                )
                candidates.append((score, points))
            return min(candidates, key=lambda item: item[0])[1] if candidates else None
        except (ImportError, AssertionError, RuntimeError, ValueError, AttributeError, TypeError, IndexError):
            # MeshLib 版本、坏拓扑、边缘线不贴合等均不能阻断整批渲染。
            return None

    def get_camera_setup(
        self,
        *,
        base_vertices: np.ndarray | None = None,
    ) -> dict[str, dict[str, np.ndarray | float]]:
        """三张图统一注视颈缘线中心；按基座/颈缘线计算距离与正交视场。

        即使没有基座也可以拍摄；永远不会读取设计冠外轮廓来计算镜头距离。
        """
        if self.margin is None:
            raise ValueError("生成牙颌三视图必须提供颈缘线，作为相机焦点")
        margin_local = (self.margin - self.origin) @ self.frame
        margin_span = float(np.ptp(margin_local, axis=0).max())
        if margin_span <= 1e-6:
            raise ValueError("颈缘线空间范围过小，无法确定相机")
        span = margin_span
        if base_vertices is not None:
            base = np.asarray(base_vertices, dtype=np.float64)
            if base.ndim != 2 or base.shape[1] != 3 or not np.isfinite(base).all():
                raise ValueError("base_vertices 必须是有限的 [N,3] 坐标")
            if len(base):
                base_local = (base - self.origin) @ self.frame
                base_span = float(np.ptp(base_local, axis=0).max())
                # 错误提取的整颌不应造成相机无限远离。
                if 0 < base_span <= 4.0 * margin_span:
                    span = max(span, base_span)

        distance = max(3.2 * span, 10.0)
        parallel_scale = max(0.9 * span, 2.0)
        # 在局部坐标定义观察方向，再转换回世界坐标。
        directions = (
            (np.array([0., 1., 0.]), np.array([0., 0., 1.])),
            (np.array([0., -1., 0.]), np.array([0., 0., 1.])),
            (np.array([0., 0., 1.]), np.array([0., 1., 0.])),
        )
        result = {}
        for name, (direction, up) in zip(self.NAMES, directions):
            result[name] = {
                "position": self.origin + self.frame @ direction * distance,
                "focal_point": self.origin.copy(),
                "view_up": self.frame @ up,
                "distance": distance,
                "parallel_scale": parallel_scale,
            }
        return result

    def get_images(
        self,
        image_size: int = 256,
        *,
        jaw_vertices: np.ndarray,
        jaw_faces: np.ndarray,
        opposite_vertices: np.ndarray,
        opposite_faces: np.ndarray,
        base_vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """颊、舌侧同时渲染上下颌；咬合视图只渲染目标牙所在颌。

        jaw_* 是 current_mesh，不包含设计成品冠。opposite_* 仅在两
        个侧向视图中参与图像渲染，不送入 PTv3 点云条件。
        """
        if image_size < 64:
            raise ValueError("image_size 不能小于 64")
        meshes = []
        for name, vertices_input, faces_input in (
            ("target_jaw", jaw_vertices, jaw_faces),
            ("opposite_jaw", opposite_vertices, opposite_faces),
        ):
            vertices = np.asarray(vertices_input, dtype=np.float64)
            faces = np.asarray(faces_input, dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
                raise ValueError(f"{name}.vertices 必须是 [N,3]")
            if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
                raise ValueError(f"{name}.faces 必须是 [F,3]")
            if not np.isfinite(vertices).all() or (faces < 0).any() or (faces >= len(vertices)).any():
                raise ValueError(f"{name} 的网格坐标或面片索引非法")
            meshes.append((vertices, faces))

        import vedo
        from PIL import Image

        cameras = self.get_camera_setup(base_vertices=base_vertices)
        images = []
        with TemporaryDirectory(prefix="toothforge_jaw_views_") as directory:
            for i, name in enumerate(self.NAMES):
                path = Path(directory) / f"{i:02d}_{name}.png"
                # 只用目标牙颌生成咬合图；侧视图才增加对颌。
                count = 1 if name == "occlusal" else 2
                actors = [
                    vedo.Mesh([vertices, faces]).compute_normals(
                        points=True, cells=False,
                    ).c("#e5e9ef" if index == 0 else "#c9cdd2").lighting("glossy")
                    for index, (vertices, faces) in enumerate(meshes[:count])
                ]
                plotter = vedo.Plotter(
                    offscreen=True, size=(image_size, image_size), bg="#ffffff"
                )
                try:
                    plotter.show(*actors, resetcam=True, axes=0, interactive=False)
                    camera = plotter.camera
                    spec = cameras[name]
                    camera.SetFocalPoint(*spec["focal_point"])
                    camera.SetPosition(*spec["position"])
                    camera.SetViewUp(*spec["view_up"])
                    camera.ParallelProjectionOn()
                    camera.SetParallelScale(spec["parallel_scale"])
                    plotter.renderer.ResetCameraClippingRange()
                    plotter.render()
                    plotter.screenshot(str(path))
                    with Image.open(path) as pic:
                        images.append(np.asarray(pic.convert("RGB")).copy())
                finally:
                    plotter.close()
        return np.stack(images, axis=0)

    def encode(
        self,
        encoder: object,
        image_size: int = 256,
        *,
        jaw_vertices: np.ndarray,
        jaw_faces: np.ndarray,
        opposite_vertices: np.ndarray,
        opposite_faces: np.ndarray,
        base_vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """使用 SindrePy DINOv3Encoder 提取牙颌三视图 patch tokens。"""
        images = self.get_images(
            image_size,
            jaw_vertices=jaw_vertices,
            jaw_faces=jaw_faces,
            opposite_vertices=opposite_vertices,
            opposite_faces=opposite_faces,
            base_vertices=base_vertices,
        )
        features = [
            np.asarray(encoder.encode(image, return_tokens=True), dtype=np.float32)
            for image in images
        ]
        if len({tuple(feat.shape) for feat in features}) != 1:
            raise ValueError("三个视角的 DINOv3 token 形状不一致")
        return np.stack(features, axis=0)
