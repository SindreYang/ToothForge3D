"""基于病例路径的开放牙冠 FDG 推理。

不需要 argparse/CLI：直接调用 infer(path, checkpoint_path, 必要条件)。
输入 JSON 或原始病例 LMDB；不允许让真实设计冠网格进入条件编码器。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import LightFDGConfig, create_light_fdg_model
from process import DentalCase, collate_conditions
from .utils import CrownFDG, FDGConfig
from sindre.ai.utils import load_checkpoint


def get_case(
    path: str | Path,
    *,
    index: int = 0,
    target_fdi: int | None = None,
    margin_line: np.ndarray | None = None,
    size_mm: tuple[float, float, float] | np.ndarray | None = None,
    local_frame: np.ndarray | None = None,
) -> dict[str, Any]:
    """加载原始牙颌病例，清除任何目标设计冠。

    Args:
        path: 原始病例 JSON 文件路径或 Sindre LMDB 路径。
        index: LMDB 内病例索引；JSON 文件时应为 0。
        target_fdi: 目标修复牙的真实 FDI 编号。
        margin_line: 输入扫描坐标下的真实颈缘线 [M,3]，毫米。
        size_mm: 可选目标 (MD, BL, H)，单位毫米；不提供则由网络估计。
        local_frame: 局部 MD/BL/冠向右手正交坐标系 [3,3]。

    Returns:
        可以直接送给 DentalCase(..., inference=True) 的数据字典。

    Raises:
        ValueError: 缺少必要条件、格式不合法或路径中试图注入设计冠。
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if index < 0:
        raise ValueError("index 不能为负")
    if path.is_file() and path.suffix.lower() == ".json":
        if index:
            raise ValueError("JSON 输入只有一份病例，请使用 index=0")
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    else:
        from sindre.lmdb import Reader

        with Reader(str(path)) as reader:
            if index >= len(reader):
                raise IndexError("LMDB index 超出病例数量")
            schema = reader.get_meta("schema_version")
            if schema is not None and str(schema).startswith("toothforge3d_"):
                raise ValueError(
                    "推理要求原始病例 LMDB；训练标签库可能含 GT 条件，"
                    "请改用原始扫描或独立构造无 GT 的病例 JSON"
                )
            raw = reader[index]
    if not isinstance(raw, dict) or "current_mesh" not in raw:
        raise ValueError("输入必须包含真实牙颌扫描 current_mesh")
    data = dict(raw)
    # 牙位仅来自用户条件/扫描病例，禁止从设计成品冠推导。
    target_overridden = target_fdi is not None
    if target_fdi is None:
        target_fdi = data.get("target_fdi")
    had_target_design = "design_mesh" in data
    data.pop("design_mesh", None)
    data["target_fdi"] = target_fdi
    if had_target_design:
        # 原始样本可能曾由成品冠计算视图坐标和尺寸标签。
        # 用户未显式传 size_mm 时，不可信任同文件中残留的设计冠尺寸。
        data.pop("view_features", None)
        data.pop("view_mask", None)
        data.pop("view_scale_mm", None)
        data.pop("local_frame", None)
        if size_mm is None:
            data.pop("size_mm", None)
    if margin_line is not None:
        data["margin_line"] = np.asarray(margin_line, dtype=np.float32)
    if size_mm is not None:
        data["size_mm"] = np.asarray(size_mm, dtype=np.float32)
    if local_frame is not None:
        data["local_frame"] = np.asarray(local_frame, dtype=np.float32)
    # 改变牙位、颈缘线或方向都会改变取景/相机，旧视图缓存必须失效。
    if target_overridden or margin_line is not None or local_frame is not None or data.get("local_frame") is None:
        data.pop("view_features", None)
        data.pop("view_mask", None)
        data.pop("view_scale_mm", None)
    for key in ("target_fdi", "margin_line"):
        if data.get(key) is None:
            raise ValueError(
                f"推理缺少 {key}；必须从病例或 infer(...) 显式传入，"
                "禁止从真实设计冠 Mesh 推算"
            )
    return data


def get_condition(
    data: dict[str, Any] | DentalCase,
    *,
    jaw_points: int = 2048,
    image_encoder: Any | None = None,
    image_size: int = 256,
) -> dict[str, torch.Tensor]:
    """从无设计冠病例构建模型的条件批次。

    Args:
        data: get_case 返回的原始病例（无设计冠）。
        jaw_points: 同颌 PTv3 输入顶点数量。
        image_encoder: 预处理三视图时使用的 SindrePy DINOv3 编码器。
        image_size: 生成三视图的像素尺寸。

    Returns:
        批量大小为 1 的 MCDG 条件张量字典。
    """
    case = data if isinstance(data, DentalCase) else DentalCase(data, inference=True)
    record = case.get_condition(
        crop_mode="full", tooth_dropout=0.0,
        num_points=jaw_points,
        image_encoder=image_encoder, image_size=image_size,
    )
    if not record["ready_for_training"]:
        raise ValueError(
            "缺少三视图 DINOv3 特征；若病例没有 view_features，"
            "请传入 SindrePy DINOv3Encoder 或使用自动加载"
        )
    return collate_conditions([record], num_points=jaw_points)


def get_world_vertices(
    local_vertices: np.ndarray, world_to_local_transform: np.ndarray,
) -> np.ndarray:
    """使用预处理时的原始刚体矩阵恢复世界毫米坐标，不重复计算颈缘中心。"""
    from sindre.utils3d.sindrepcd import SindrePCD

    matrix = np.asarray(world_to_local_transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("需要有效的世界到局部 4x4 变换矩阵")
    cloud = SindrePCD(np.asarray(local_vertices, dtype=np.float32))
    cloud.apply_inv_transform(matrix)
    return np.asarray(cloud.points, dtype=np.float32)


def show_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    case_data: dict[str, Any] | None = None,
    show_context: bool = True,
    show_opposite: bool = True,
    show_margin: bool = True,
) -> None:
    """使用 Vedo 交互显示生成的牙冠与输入牙颌环境。

    生成冠使用世界毫米坐标；输入环境只读取 current/opposite_mesh，
    且自动去除 current_mesh 中与目标设计冠对应的成品牙位面片。

    Args:
        vertices: 生成冠世界坐标顶点 [V,3]。
        faces: 生成冠三角面片 [F,3]。
        case_data: 可选的无设计冠原始病例。
        show_context: 是否显示同颌扫描与预备体。
        show_opposite: 是否显示对颌扫描。
        show_margin: 是否显示红色闭合颈缘线。

    Raises:
        ValueError: Mesh 为空或无效。
    """
    import vedo

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if (v.ndim != 2 or v.shape[1] != 3 or not len(v)
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or not np.isfinite(v).all() or (f < 0).any()
            or (f >= len(v)).any()):
        raise ValueError("Vedo 渲染需要有效的生成牙冠 Mesh")
    actors = []
    if case_data is not None and (show_context or show_opposite):
        # 与模型预处理一致：不混入当前扫描中误含的设计成品冠。
        case = DentalCase(case_data, inference=True)
        if show_context and len(case.scene_faces):
            actors.append(
                vedo.Mesh([case.scene_vertices, case.scene_faces])
                .c("#c5cbd3").alpha(0.40)
            )
        if show_opposite and case.opposite_vertices is not None:
            actors.append(
                vedo.Mesh([case.opposite_vertices, case.opposite_faces])
                .c("#cbd5e1").alpha(0.28)
            )
    actors.append(vedo.Mesh([v, f]).c("#60a5fa").alpha(0.98))
    if show_margin and case_data is not None and case_data.get("margin_line") is not None:
        margin = np.asarray(case_data["margin_line"], dtype=np.float32)
        if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 3:
            raise ValueError("渲染颈缘线必须是 [M,3]")
        actors.append(
            vedo.Line(np.concatenate((margin, margin[:1]), axis=0))
            .c("#dc2626").lw(4)
        )
    plotter = vedo.Plotter(offscreen=False, size=(1200, 900), bg="#f7f9fc")
    try:
        plotter.show(*actors, axes=1, interactive=True, resetcam=True)
    finally:
        plotter.close()


class CrownInference:
    """按 checkpoint 创建可复用推理器；网络首次使用时加载，后续病例复用。"""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto"):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        target_device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )
        if target_device.type != "cuda":
            raise RuntimeError("当前原生 O-Voxel FDG 网格提取需要 CUDA 设备")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or "state_dict" not in saved:
            raise ValueError("checkpoint 不包含训练模型参数")
        if "model_config" not in saved:
            raise ValueError("checkpoint 缺少 LightFDGConfig；请用新版 train.py 训练")
        if saved.get("fdg_schema") != "toothforge3d_fdg_v3":
            raise ValueError(
                "仅支持 scan_arch_margin_v3 的 FDG 自动尺寸权重；旧模型必须重新训练"
            )
        model_config = LightFDGConfig(**saved["model_config"])
        train_config = saved.get("train_config", {})
        self.default_jaw_points = int(train_config.get("jaw_points", 2048))
        self.device = target_device
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.model = None
        self.converter = CrownFDG(FDGConfig(resolution=model_config.resolution,
                                            padding_ratio=model_config.padding_ratio))

    def infer(
        self, path: str | Path,
        *,
        index: int = 0,
        target_fdi: int | None = None,
        margin_line: np.ndarray | None = None,
        size_mm: tuple[float, float, float] | np.ndarray | None = None,
        local_frame: np.ndarray | None = None,
        image_encoder: Any | None = None,
        jaw_points: int | None = None,
        image_size: int = 256,
        output_path: str | Path | None = None,
        refine_margin: bool = True,
        show: bool = False,
        show_context: bool = True,
        show_opposite: bool = True,
        validate_geometry: bool = True,
        max_margin_error_mm: float = 0.5,
    ) -> dict[str, Any]:
        """从真实扫描生成开放冠并回写原世界坐标。"""
        path = Path(path).expanduser().resolve()
        if not np.isfinite(max_margin_error_mm) or max_margin_error_mm <= 0:
            raise ValueError("max_margin_error_mm 必须为正有限毫米值")
        model_config, target_device = self.model_config, self.device
        used_jaw_points = jaw_points or self.default_jaw_points
        case_data = get_case(
            path, index=index, target_fdi=target_fdi,
            margin_line=margin_line, size_mm=size_mm, local_frame=local_frame,
        )
        case = DentalCase(case_data, inference=True)
        # 原始病例没有缓存 DINOv3，则仅在推理预处理时临时创建编码器，
        # 不将其加入 FDG 模型训练。
        if case.view_features is None and image_encoder is None:
            from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder

            with DINOv3Encoder() as encoder:
                condition = get_condition(
                    case, jaw_points=used_jaw_points,
                    image_encoder=encoder, image_size=image_size,
                )
        else:
            condition = get_condition(
                case, jaw_points=used_jaw_points,
                image_encoder=image_encoder, image_size=image_size,
            )
        if self.model is None:
            model = create_light_fdg_model(
                image_dim=condition["view_features"].shape[-1],
                margin_points=condition["margin_line"].shape[1],
                config=self.model_config,
            ).to(self.device)
            # 复用 SindrePy 权重加载和参数检查；不允许静默载入不完整参数。
            epoch, _, _ = load_checkpoint(
                path=str(self.checkpoint_path), net=model, strict=True,
                map_location=str(self.device),
            )
            if epoch < 1:
                raise RuntimeError("模型权重未包含训练纪元，检查 checkpoint 来源")
            self.model = model.eval()
        model = self.model
        condition = {
            key: value.to(target_device, non_blocking=True)
            if torch.is_tensor(value) else value
            for key, value in condition.items()
        }
        model.eval()
        with torch.inference_mode():
            prediction = model.predict(condition)[0]
        if not prediction["valid"]:
            raise RuntimeError("未预测到足够的有效 FDG 体素，无法生成牙冠 Mesh")
        converter = self.converter
        center, scale = converter.get_condition_frame(
            condition["margin_line"][0].detach().cpu().numpy(),
        )
        with torch.inference_mode():
            vertices, faces = converter.get_mesh(
                {
                    "coords": prediction["coords"],
                    "center_mm": center, "scale_mm": scale,
                    "resolution": model_config.resolution,
                },
                dual_offsets=prediction["dual_offsets"],
                intersected_logits=prediction["intersected_logits"],
            )
            if not len(faces):
                raise RuntimeError("O-Voxel 返回空 Mesh，无法代表有效牙冠")
            if refine_margin:
                vertices = converter.refine_margin(
                    vertices, faces, condition["margin_line"][0],
                )
        # 模型使用局部 MD/BL/冠向坐标；CAD 输出恢复到原始扫描坐标。
        vertices_np = get_world_vertices(
            vertices.detach().cpu().numpy(),
            case.world_to_local_transform,
        )
        faces_np = faces.detach().cpu().numpy().astype(np.int32)
        # 不以用户预估或先验尺寸冒充真实输出尺寸：直接测量生成冠。
        local_vertices = vertices.detach().cpu().numpy().astype(np.float32)
        quality: dict[str, Any] = {"verified": False}
        if validate_geometry:
            from .losses import CrownFDGLoss

            quality = converter.get_mesh_quality(local_vertices, faces_np)
            margin_result = CrownFDGLoss().get_margin_loss(
                torch.from_numpy(local_vertices),
                torch.from_numpy(faces_np.astype(np.int64)),
                condition["margin_line"][0].detach().cpu(),
            )
            quality["margin_mean_mm"] = float(margin_result["mean_distance_mm"])
            quality["margin_max_mm"] = float(margin_result["max_distance_mm"])
            if quality["margin_max_mm"] > max_margin_error_mm:
                raise ValueError(
                    f"生成冠颈缘线最大误差 {quality['margin_max_mm']:.3f}mm "
                    f"超过 {max_margin_error_mm:.3f}mm，拒绝不合格 Mesh"
                )
            environment = converter.get_environment_surfaces(case, local_vertices, require_opposite=False)
            contacts = converter.get_collision_status(local_vertices, faces_np, environment)
            quality["contacts"] = contacts
            if any(item["collision"] for item in contacts.values()):
                raise ValueError("生成冠与邻牙/对颌发生表面相交或接触；拒绝不合格 Mesh")
            quality["opposite_present"] = case.opposite_vertices is not None
            quality["labels_known"] = case.labels_known
            quality["verified"] = bool(quality["opposite_present"] and quality["labels_known"])
            if not quality["verified"]:
                quality["preview_reason"] = "缺少对颌或邻牙标签，尚未完成可区分的邻接/咬合验收"
        generated_size = np.array([
            np.ptp(local_vertices[:, 0]),
            np.ptp(local_vertices[:, 1]),
            local_vertices[:, 2].max(),
        ], dtype=np.float32)
        saved_path = None
        if output_path is not None:
            output_path = Path(output_path).expanduser().resolve()
            if output_path.suffix.lower() not in (".stl", ".ply", ".obj"):
                raise ValueError("output_path 必须是 STL、PLY 或 OBJ")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            from sindre.utils3d.sindremesh import SindreMesh
            SindreMesh([vertices_np, faces_np]).save(str(output_path))
            saved_path = str(output_path)
        if show:
            show_mesh(
                vertices_np, faces_np, case_data=case_data,
                show_context=show_context, show_opposite=show_opposite,
            )
        return {
            "vertices": vertices_np, "faces": faces_np,
            "coordinate_system": "input_world_mm",
            "target_fdi": int(case_data["target_fdi"]),
            "size_mm": generated_size,
            "generated_size_mm": generated_size,
            "requested_size_mm": (
                None if case_data.get("size_mm") is None
                else np.asarray(case_data["size_mm"], dtype=np.float32)
            ),
            "estimated_size_mm": prediction["estimated_size_mm"].detach().cpu().numpy(),
            "conditioned_size_mm": prediction["conditioned_size_mm"].detach().cpu().numpy(),
            "size_source": "user" if case_data.get("size_mm") is not None else "model",
            "geometry_quality": quality,
            "preview": not quality["verified"],
            "output_path": saved_path,
        }

def infer(
    path: str | Path,
    checkpoint_path: str | Path,
    *,
    index: int = 0,
    target_fdi: int | None = None,
    margin_line: np.ndarray | None = None,
    size_mm: tuple[float, float, float] | np.ndarray | None = None,
    local_frame: np.ndarray | None = None,
    image_encoder: Any | None = None,
    jaw_points: int | None = None,
    image_size: int = 256,
    device: str = "auto",
    output_path: str | Path | None = None,
    refine_margin: bool = True,
    show: bool = False,
    show_context: bool = True,
    show_opposite: bool = True,
    validate_geometry: bool = True,
    max_margin_error_mm: float = 0.5,
) -> dict[str, Any]:
    """单病例便捷入口；批量调用请复用 CrownInference。"""
    return CrownInference(checkpoint_path, device=device).infer(
        path, index=index, target_fdi=target_fdi, margin_line=margin_line,
        size_mm=size_mm, local_frame=local_frame, image_encoder=image_encoder,
        jaw_points=jaw_points, image_size=image_size, output_path=output_path,
        refine_margin=refine_margin, show=show, show_context=show_context,
        show_opposite=show_opposite, validate_geometry=validate_geometry,
        max_margin_error_mm=max_margin_error_mm,
    )


def main(
    path: str | Path,
    checkpoint_path: str | Path,
    *,
    target_fdi: int | None = None,
    size_mm: tuple[float, float, float] | None = None,
    margin_line: np.ndarray | None = None,
    local_frame: np.ndarray | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
    show_context: bool = True,
    show_opposite: bool = True,
    validate_geometry: bool = True,
    max_margin_error_mm: float = 0.5,
) -> dict[str, Any]:
    """用显式参数推理，无 CLI、无 argparse。

    Args:
        path: 原始病例路径。
        checkpoint_path: 训练所得权重路径。
        target_fdi: 目标真实牙位 FDI。
        size_mm: 可选的近远中宽、颊舌厚、高度，毫米；不传自动估计。
        margin_line: 世界坐标下的真实颈缘线。
        local_frame: 世界坐标到局部的正交轴。
        output_path: 可选结果 Mesh 保存路径。
        show: 是否打开 Vedo 交互渲染窗口。
        show_context: 是否显示同颌扫描。
        show_opposite: 是否显示对颌扫描。
        validate_geometry: 是否进行严格几何质量检查。
        max_margin_error_mm: 颈缘线最大误差，毫米。

    Returns:
        infer(...) 的完整结果。
    """
    return infer(
        path, checkpoint_path,
        target_fdi=target_fdi, size_mm=size_mm,
        margin_line=margin_line, local_frame=local_frame,
        output_path=output_path, show=show,
        show_context=show_context, show_opposite=show_opposite,
        validate_geometry=validate_geometry,
        max_margin_error_mm=max_margin_error_mm,
    )
