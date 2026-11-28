"""使用扫描几何和冠向先验确定局部轴，渲染真实牙颌三视图。

设计冠不参与坐标系估计或输入图像。
相机目标固定为颈缘线中心，距离/视场依据裁剪基座或颈缘线范围确定。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def get_world_to_local_transform(
    origin: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Build SindrePy's affine matrix for ``(point - origin) @ frame``."""
    origin = np.asarray(origin, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    if (
        origin.shape != (3,)
        or frame.shape != (3, 3)
        or not np.isfinite(origin).all()
        or not np.isfinite(frame).all()
        or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(frame), 1.0, atol=1e-3)
    ):
        raise ValueError("origin/frame 必须构成有效的牙科局部坐标系")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = frame.T
    transform[:3, 3] = -(frame.T @ origin)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Use SindrePCD to apply an affine transform to points."""
    from sindre.utils3d.sindrepcd import SindrePCD

    return SindrePCD(points).apply_transform(transform).points


class CrownViews:
    """从同颌扫描 + 颈缘线建立共享坐标与三视图。

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
        jaw_fdi_labels: np.ndarray | None = None,
        target_fdi: int | None = None,
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
            self.frame_confidence = None
            self.world_to_local_transform = get_world_to_local_transform(
                self.origin, self.frame
            )
            return

        if self.margin is None:
            raise ValueError("自动建立局部坐标系需要真实冠底边界")
        from sindre.utils3d.core.dental.arch.alignment import DentalArchFrame

        if jaw_points is None:
            raise ValueError("自动坐标估计必须提供真实同颌扫描，不能使用设计冠")
        try:
            result = DentalArchFrame(jaw_points, self.margin).get_frame()
        except ValueError:
            # 只允许真实扫描中的牙位作为回退锚点，不读取设计冠形态。
            if jaw_fdi_labels is None or target_fdi is None:
                raise
            result = self.get_fdi_frame(jaw_points, jaw_fdi_labels, target_fdi, self.margin)
        self.frame = result["frame"].astype(np.float32)
        self.origin = result["origin"]
        self.frame_source = result["frame_source"]
        self.frame_confidence = result["confidence"]
        self.anatomical_verified = False  # 自动估计尚需真实病例验证
        self.axis_verified = True
        self.world_to_local_transform = result["transform"]

    @staticmethod
    def get_fdi_frame(points, labels, target_fdi, margin):
        """无标签牙弓拟合失败时，复用 SindrePy FDI 锚点配准。

        Args:
            points: 同颌真实扫描顶点。
            labels: 与顶点对应的真实 FDI 标签。
            target_fdi: 目标修复牙位。
            margin: 真实颈缘线，禁止读取设计冠体。

        Returns:
            保持冠向 +Z 的右手局部坐标及诊断；低置信度拒绝。
        """
        from sindre.utils3d.core.dental.arch.alignment import DentalArchAlignment

        points, labels = np.asarray(points), np.asarray(labels).reshape(-1)
        target_fdi = int(target_fdi)
        upper = target_fdi // 10 in (1, 2)
        quadrants = (1, 2) if upper else (3, 4)
        if len(points) != len(labels):
            raise ValueError("真实牙位锚点与扫描点数不一致")
        teeth = {}
        for fdi in (q*10+i for q in quadrants for i in range(1,9)):
            if fdi != target_fdi:
                vertices = points[labels == fdi]
                if len(vertices) >= 20:
                    teeth[fdi] = (vertices, None)
        # 真实颈缘中心是可观察目标位置锚点，不借用设计冠形状。
        if len(teeth) < 2:
            raise ValueError("真实邻牙锚点不足，无法可靠恢复颊舌方向")
        teeth[target_fdi] = (np.asarray(margin), None)
        try:
            aligned = DentalArchAlignment(
                None, teeth, superior_hint=(0., 0., -1. if upper else 1.),
            ).align()
            if aligned["requires_review"] or aligned["confidence"] not in ("high", "medium"):
                raise ValueError("FDI 全弓配准置信度不足")
        except ValueError:
            # 全弓配准失败才尝试局部锚点；已成功的旧路径完全不变。
            aligned = CrownViews.get_local_fdi_alignment(teeth, target_fdi, upper)
        # 标准牙弓的切线确定局部 MD，凸向确定 BL；最终仍保持冠向 +Z。
        quadrant, index = divmod(target_fdi, 10)
        _, tx, ty = DentalArchAlignment.fdi_xy(target_fdi)
        if index == 1:
            other = {1:2, 2:1, 3:4, 4:3}[quadrant]
            left, right = other*10+1, target_fdi+1
        elif index == 8:
            left, right = target_fdi-1, target_fdi
        else:
            left, right = target_fdi-1, target_fdi+1
        a = np.array(DentalArchAlignment.fdi_xy(left)[1:])
        b = np.array(DentalArchAlignment.fdi_xy(right)[1:])
        tangent = (b-a) / np.linalg.norm(b-a)
        outward = np.array([tx,ty]) - tangent * (tangent @ np.array([tx,ty]))
        outward /= np.linalg.norm(outward)
        y = aligned["rotation"].T @ np.r_[outward, 0.]
        z = np.array([0.,0.,1.])
        y = DentalArchAlignment.unit(y - z*(y @ z), "buccal")
        x = np.cross(y, z)
        frame = np.column_stack((x, y, z))
        origin = np.asarray(margin, dtype=float).mean(axis=0)
        return {"frame": frame, "origin": origin,
                "transform": get_world_to_local_transform(origin, frame),
                "frame_source": "scan_fdi_anchors",
                "confidence": aligned.get("local_confidence", 0.85 if aligned["confidence"] == "high" else 0.60)}

    @staticmethod
    def get_local_fdi_alignment(teeth, target_fdi, upper):
        """复用 SindrePy 稳健平面配准，以目标颈缘为锚点验证局部邻牙方向。

        至少两颗真实邻牙；排除近竖直、距离不合理的锚点。全弓失败才调用。
        不从设计冠或单一颈缘线猜测颊舌正负方向。
        """
        from sindre.utils3d.core.dental.arch.alignment import DentalArchAlignment

        origin = np.asarray(teeth[target_fdi][0], dtype=float).mean(axis=0)
        target = np.asarray(DentalArchAlignment.fdi_xy(target_fdi)[1:])
        reflection = np.diag([1., -1. if upper else 1.])
        anchors = []
        for fdi, (vertices, _) in teeth.items():
            if fdi == target_fdi:
                continue
            delta = np.asarray(vertices, dtype=float).mean(axis=0) - origin
            expected = (np.asarray(DentalArchAlignment.fdi_xy(fdi)[1:]) - target) @ reflection
            measured = float(np.linalg.norm(delta[:2]))
            nominal = float(np.linalg.norm(expected))
            if not (4. <= measured <= 32. and 4. <= nominal <= 30.
                    and .55 <= measured / nominal <= 1.8
                    and abs(delta[2]) <= max(8., measured)):
                continue
            anchors.append((nominal, int(fdi), delta[:2], expected))
        anchors.sort(key=lambda item: (item[0], item[1]))
        if len(anchors) < 2:
            raise ValueError("局部真实邻牙锚点不足，请提供 local_frame")
        # 使用最近2~4颗进行候选拟合，要求相互一致，不用单一最优残差掩盖歧义。
        candidates = []
        for count in range(2, min(4, len(anchors)) + 1):
            observed = np.vstack((np.zeros(2), *[a[2] for a in anchors[:count]]))
            expected = np.vstack((np.zeros(2), *[a[3] for a in anchors[:count]]))
            q, scale, rms = DentalArchAlignment.template_fit(observed, expected)
            if not (.65 <= scale <= 1.5 and rms <= 3.0):
                continue
            errors = np.linalg.norm(observed @ q * scale - expected, axis=1)
            if errors.max() > 3.5:
                continue
            candidates.append((count, q, float(errors.max())))
        if not candidates:
            raise ValueError("局部牙位锚点无法一致配准，请提供 local_frame")
        count, q, error = candidates[-1]
        if any(np.trace(other.T @ q) < 2 * np.cos(np.deg2rad(20.))
               for _, other, _ in candidates):
            raise ValueError("局部牙弓方向存在歧义，请提供 local_frame")
        # 反射只用于标准上颌坐标约定；实际返回的是三维右手旋转。
        rotation = np.eye(3)
        rotation[:2, :2] = reflection @ q.T
        rotation[2, 2] = -1. if upper else 1.
        return {"rotation": rotation, "confidence": "medium",
                "local_confidence": float(.45 + .10 * min(count / 4., 1.) * (1. - error / 3.5))}

    def to_local_points(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.world_to_local_transform)

    def to_world_points(self, points: np.ndarray) -> np.ndarray:
        from sindre.utils3d.sindrepcd import SindrePCD

        return SindrePCD(points).apply_inv_transform(
            self.world_to_local_transform
        ).points

    def to_world_directions(self, directions: np.ndarray) -> np.ndarray:
        from sindre.utils3d.sindrepcd import SindrePCD

        directions = np.asarray(directions, dtype=np.float64)
        cloud = SindrePCD({
            "points": np.zeros_like(directions),
            "point_normals": directions,
        })
        return cloud.apply_inv_transform(
            self.world_to_local_transform
        ).point_normals

    def get_local_vertices(self) -> np.ndarray:
        """返回同一世界原点与坐标基下的目标牙冠（mm）。"""
        from sindre.utils3d.sindremesh import SindreMesh

        mesh = SindreMesh([self.vertices, self.faces])
        mesh.apply_transform(self.world_to_local_transform)
        return np.asarray(mesh.vertices, dtype=np.float32)

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
        from sindre.utils3d.core.algorithm import cut_mesh_by_meshlib

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
                self.to_local_points(self.margin), axis=0,
            ).max()), 1e-6)
            candidates = []
            for part_v, part_f in ((parts[0], parts[1]), (parts[2], parts[3])):
                points = np.asarray(part_v, dtype=np.float64)
                cells = np.asarray(part_f, dtype=np.int64)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or len(cells) == 0:
                    continue
                local = self.to_local_points(points)
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
        opposite_vertices: np.ndarray | None = None,
    ) -> dict[str, dict[str, np.ndarray | float]]:
        """三张图统一注视颈缘线中心；按基座/颈缘线计算距离与正交视场。

        即使没有基座也可以拍摄；永远不会读取设计冠外轮廓来计算镜头距离。
        """
        if self.margin is None:
            raise ValueError("生成牙颌三视图必须提供颈缘线，作为相机焦点")
        margin_local = self.to_local_points(self.margin)
        margin_span = float(np.ptp(margin_local, axis=0).max())
        if margin_span <= 1e-6:
            raise ValueError("颈缘线空间范围过小，无法确定相机")
        span = margin_span
        if base_vertices is not None:
            base = np.asarray(base_vertices, dtype=np.float64)
            if base.ndim != 2 or base.shape[1] != 3 or not np.isfinite(base).all():
                raise ValueError("base_vertices 必须是有限的 [N,3] 坐标")
            if len(base):
                base_local = self.to_local_points(base)
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
                "position": self.to_world_points(
                    (direction * distance)[None]
                )[0],
                "focal_point": self.origin.copy(),
                "view_up": self.to_world_directions(up[None])[0],
                "distance": distance,
                "parallel_scale": parallel_scale,
            }
        if opposite_vertices is not None:
            opposite = np.asarray(opposite_vertices, dtype=np.float64)
            if opposite.ndim != 2 or opposite.shape[1] != 3 or not np.isfinite(opposite).all():
                raise ValueError("opposite_vertices 必须为有限的 [N,3]")
            # 颊/舌侧图的垂直视野需要包含附近对颌，但不能因为远处全牙弓
            # 或错误的咬合扫描而无限放大。相机焦点和距离不受对颌影响。
            if len(opposite):
                local = self.to_local_points(opposite)
                near = (
                    (np.abs(local[:, 0]) <= 2.5 * span)
                    & (np.abs(local[:, 1]) <= 2.5 * span)
                    & (np.abs(local[:, 2]) <= 6.0 * span)
                )
                if near.any():
                    selected = local[near]
                    side_scale = max(
                        parallel_scale,
                        1.15 * float(np.abs(selected[:, [0, 2]]).max()),
                    )
                    for side in ("buccal", "lingual"):
                        result[side]["parallel_scale"] = side_scale
        return result

    def get_images(
        self,
        image_size: int = 256,
        *,
        jaw_vertices: np.ndarray,
        jaw_faces: np.ndarray,
        opposite_vertices: np.ndarray | None,
        opposite_faces: np.ndarray | None,
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
            if name == "opposite_jaw" and vertices_input is None and faces_input is None:
                continue
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

        cameras = self.get_camera_setup(
            base_vertices=base_vertices, opposite_vertices=opposite_vertices,
        )
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
        jaw_vertices: np.ndarray | None = None,
        jaw_faces: np.ndarray | None = None,
        opposite_vertices: np.ndarray | None = None,
        opposite_faces: np.ndarray | None = None,
        base_vertices: np.ndarray | None = None,
        images: np.ndarray | None = None,
        view_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """复用 SindrePy DINOv3 编码器；可直接编码已渲染的三张图，避免重复截图。"""
        if images is None:
            if any(value is None for value in (
                jaw_vertices, jaw_faces,
            )):
                raise ValueError("未提供 images 时必须传入同颌 Mesh")
            images = self.get_images(
                image_size,
                jaw_vertices=jaw_vertices, jaw_faces=jaw_faces,
                opposite_vertices=opposite_vertices, opposite_faces=opposite_faces,
                base_vertices=base_vertices,
            )
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[0] != 3 or images.shape[-1] != 3:
            raise ValueError("三视图 RGB images 必须是 [3,H,W,3]")
        valid = np.ones(3, dtype=bool) if view_mask is None else np.asarray(view_mask, dtype=bool)
        if valid.shape != (3,) or not valid.any():
            raise ValueError("至少需要一个有效三视图")
        features = {i: np.asarray(encoder.encode(images[i], return_tokens=True), dtype=np.float32)
                    for i in np.flatnonzero(valid)}
        if len({tuple(feat.shape) for feat in features.values()}) != 1:
            raise ValueError("有效视角的 DINOv3 token 形状不一致")
        template = next(iter(features.values()))
        return np.stack([features.get(i, np.zeros_like(template)) for i in range(3)])
