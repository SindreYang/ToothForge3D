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

from model import LightFDGConfig, create_light_fdg_model
from process import DentalCase, collate_conditions
from utils import CrownFDG, FDGConfig
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
        size_mm: 目标冠 (MD, BL, H)，单位毫米；不可取自目标冠 GT。
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
            if schema in ("toothforge3d_fdg_v1", "toothforge3d_occ_v1"):
                raise ValueError(
                    "推理要求原始病例 LMDB；训练标签库可能含 GT 条件，"
                    "请改用原始扫描或独立构造无 GT 的病例 JSON"
                )
            raw = reader[index]
    if not isinstance(raw, dict) or "current_mesh" not in raw:
        raise ValueError("输入必须包含真实牙颌扫描 current_mesh")
    data = dict(raw)
    # 只可读取牙位标签（如果没有显式指定）；目标设计 Mesh 一律删除。
    if target_fdi is None:
        target_fdi = data.get("target_fdi")
    if target_fdi is None:
        reference = data.get("design", data.get("design_mesh", data.get("desig")))
        if isinstance(reference, dict):
            target_fdi = reference.get("fdi")
    had_target_design = any(
        key in data for key in ("design", "design_mesh", "desig")
    )
    for key in ("design", "design_mesh", "desig", "merge_info", "order_info"):
        data.pop(key, None)
    data["target_fdi"] = target_fdi
    if had_target_design:
        # 原始样本可能曾利用真实设计冠估计视图坐标系；默认丢弃此缓存，
        # 由输入颈缘/指定局部方向重新渲染，避免目标形态隐式泄漏。
        data.pop("view_features", None)
    if margin_line is not None:
        data["margin_line"] = np.asarray(margin_line, dtype=np.float32)
    if size_mm is not None:
        data["size_mm"] = np.asarray(size_mm, dtype=np.float32)
    if local_frame is not None:
        data["local_frame"] = np.asarray(local_frame, dtype=np.float32)
    # 缓存特征只有当来源是当前牙颌和坐标系时才可使用；若用户重写
    # local_frame，应删除旧缓存并要求重新渲染。
    if local_frame is not None:
        data.pop("view_features", None)
    for key in ("target_fdi", "margin_line", "size_mm", "local_frame"):
        if data.get(key) is None:
            raise ValueError(
                f"推理缺少 {key}；必须从病例或 infer(...) 显式传入，"
                "禁止从真实设计冠 Mesh 推算"
            )
    return data


def get_condition(
    data: dict[str, Any],
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
    record = DentalCase(data, inference=True).get_condition(
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
    local_vertices: np.ndarray,
    world_margin: np.ndarray,
    local_frame: np.ndarray,
) -> np.ndarray:
    """将 FDG 局部输出正确还原到原始扫描的世界毫米坐标。

    Args:
        local_vertices: 模型生成的局部 Mesh 顶点 [V,3]。
        world_margin: 原始扫描中的颈缘线 [M,3]。
        local_frame: 列向量依次为 MD、BL、牙冠朝向的 [3,3] 正交矩阵。

    Returns:
        与输入病例扫描使用同一坐标系的 Mesh 顶点 [V,3]。

    Raises:
        ValueError: 矩阵、颈缘线或局部顶点形状不正确。
    """
    points = np.asarray(local_vertices, dtype=np.float32)
    margin = np.asarray(world_margin, dtype=np.float32)
    frame = np.asarray(local_frame, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 3
            or margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 3
            or frame.shape != (3,3) or not np.isfinite(points).all()
            or not np.isfinite(margin).all() or not np.isfinite(frame).all()):
        raise ValueError("世界坐标转换需要有限的局部 Mesh、颈缘线和 [3,3] 坐标系")
    return (points @ frame.T + margin.mean(axis=0)[None]).astype(np.float32)


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
) -> dict[str, Any]:
    """根据真实牙颌路径及长宽高生成一颗开放冠。

    Args:
        path: 病例 JSON 或原始 Sindre LMDB。
        checkpoint_path: 由 train.py 保存的 best.pt/last.pt。
        index: LMDB 内的病例索引。
        target_fdi: 要生成的牙位；病例没有时必填。
        margin_line: 实际颈缘线 [M,3]，世界/扫描毫米坐标。
        size_mm: 目标冠 [MD,BL,H]，单位毫米，不能读取 GT 牙冠。
        local_frame: 局部坐标基 [3,3]，MD/BL/冠向。
        image_encoder: 可选预先创建的 SindrePy DINOv3Encoder。
        jaw_points: 同颌点数量；默认用训练配置。
        image_size: 当需要渲染输入三视图时的尺寸。
        device: auto、cuda、cpu；FDG Mesh 解码需要 CUDA。
        output_path: 可选的 .stl / .ply / .obj 保存路径。
        refine_margin: 是否轻微移动开放边界靠近真实颈缘线。
        show: 是否调用 Vedo 打开交互窗口；显示时会等待窗口关闭。
        show_context: 显示原始同颌扫描与预备体。
        show_opposite: 显示原始对颌扫描。

    Returns:
        包含世界扫描坐标 vertices、faces、target_fdi、size_mm
        和 output_path 的字典。

    Raises:
        RuntimeError: 权重不匹配、没有可生成的 FDG 结构或 CUDA 解码失败。
        ValueError: 必要输入不完整或目标设计冠信息泄漏。
    """
    path = Path(path).expanduser().resolve()
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
    model_config = LightFDGConfig(**saved["model_config"])
    train_config = saved.get("train_config", {})
    used_jaw_points = jaw_points or int(train_config.get("jaw_points", 2048))
    case_data = get_case(
        path, index=index, target_fdi=target_fdi,
        margin_line=margin_line, size_mm=size_mm, local_frame=local_frame,
    )
    # 原始病例没有缓存 DINOv3，则仅在推理预处理时临时创建编码器，
    # 不将其加入 FDG 模型训练。
    if case_data.get("view_features") is None and image_encoder is None:
        from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder

        with DINOv3Encoder() as encoder:
            condition = get_condition(
                case_data, jaw_points=used_jaw_points,
                image_encoder=encoder, image_size=image_size,
            )
    else:
        condition = get_condition(
            case_data, jaw_points=used_jaw_points,
            image_encoder=image_encoder, image_size=image_size,
        )
    model = create_light_fdg_model(
        image_dim=condition["view_features"].shape[-1],
        margin_points=condition["margin_line"].shape[1],
        config=model_config,
    ).to(target_device)
    # 复用 SindrePy 权重加载和参数检查；不允许静默载入不完整参数。
    epoch, _, _ = load_checkpoint(
        path=str(checkpoint_path), net=model, strict=True,
        map_location=str(target_device),
    )
    if epoch < 1:
        raise RuntimeError("模型权重未包含训练纪元，检查 checkpoint 来源")
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
    converter = CrownFDG(FDGConfig(
        resolution=model_config.resolution,
        padding_ratio=model_config.padding_ratio,
    ))
    center, scale = converter.get_condition_frame(
        condition["margin_line"][0].detach().cpu().numpy(),
        condition["size"][0].detach().cpu().numpy(),
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
        np.asarray(case_data["margin_line"]),
        np.asarray(case_data["local_frame"]),
    )
    faces_np = faces.detach().cpu().numpy().astype(np.int32)
    saved_path = None
    if output_path is not None:
        import vedo

        output_path = Path(output_path).expanduser().resolve()
        if output_path.suffix.lower() not in (".stl", ".ply", ".obj"):
            raise ValueError("output_path 必须是 STL、PLY 或 OBJ")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vedo.Mesh([vertices_np, faces_np]).write(str(output_path))
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
        "size_mm": np.asarray(case_data["size_mm"], dtype=np.float32),
        "output_path": saved_path,
    }


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
) -> dict[str, Any]:
    """用显式参数推理，无 CLI、无 argparse。

    Args:
        path: 原始病例路径。
        checkpoint_path: 训练所得权重路径。
        target_fdi: 目标真实牙位 FDI。
        size_mm: 近远中宽、颊舌厚、高度，毫米。
        margin_line: 世界坐标下的真实颈缘线。
        local_frame: 世界坐标到局部的正交轴。
        output_path: 可选结果 Mesh 保存路径。
        show: 是否打开 Vedo 交互渲染窗口。
        show_context: 是否显示同颌扫描。
        show_opposite: 是否显示对颌扫描。

    Returns:
        infer(...) 的完整结果。
    """
    return infer(
        path, checkpoint_path,
        target_fdi=target_fdi, size_mm=size_mm,
        margin_line=margin_line, local_frame=local_frame,
        output_path=output_path, show=show,
        show_context=show_context, show_opposite=show_opposite,
    )
