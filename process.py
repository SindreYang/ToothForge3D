"""牙齿提示分割数据预处理：保牙随机裁剪、FPS 采样和 LMDB 生成。"""

import argparse
from functools import lru_cache
import math
import os
from pathlib import Path
import random

import numpy as np

from sindre.lmdb import LMDBParallel_Batch, Reader
from sindre.utils3d.dental_algorithms.dental_tools import convert_fdi2idx
from sindre.utils3d.sindremesh import SindreMesh
from sindre.utils3d.sample import sample_pcd_farthest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_PATH = PROJECT_ROOT / "datasets" / "三维分割_比赛分牙_1800.db"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "三维分割_比赛分牙_1800_处理.db"
DEBUG = False


def random_cut(
    sm: SindreMesh,
    rng: random.Random | None = None,
    *,
    min_vertices: int = 1,
    max_attempts: int = 32,
    radius_margin: float = 1.02,
    radius_fraction_range: tuple[float, float] = (0.0, 1.0),
) -> SindreMesh:
    """以随机牙齿顶点为球心裁剪，至少完整覆盖所选牙齿及其邻接面片。

    仅保留球内三角面片；同步重映射顶点及标签，并保留原始索引。
    如果无法找到满足最小顶点数的非平凡裁剪，则返回原始网格的副本。
    不会通过重复顶点满足下游 FPS 的点数要求。
    """
    vertices = np.asarray(sm.vertices)
    faces = np.asarray(sm.faces, dtype=np.int64)
    labels = np.asarray(sm.vertex_labels).reshape(-1)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices 必须是非空的 [N, 3] 数组")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces 必须是非空的 [M, 3] 数组")
    if len(labels) != len(vertices):
        raise ValueError("vertex_labels 长度必须与 vertices 一致")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces 存在无效顶点索引")
    if not np.isfinite(vertices).all():
        raise ValueError("网格顶点包含非有限坐标")
    if min_vertices < 1 or max_attempts < 1 or radius_margin < 1:
        raise ValueError("min_vertices、max_attempts 必须为正数，radius_margin >= 1")
    low, high = radius_fraction_range
    if not (0 <= low <= high <= 1):
        raise ValueError("radius_fraction_range 必须满足 0 <= low <= high <= 1")
    if len(vertices) < min_vertices:
        raise ValueError(
            f"原始网格仅有 {len(vertices)} 个点，低于要求的 {min_vertices} 个点"
        )

    tooth_ids = np.unique(labels[labels > 0]).tolist()
    if not tooth_ids:
        raise ValueError("原始网格不存在有效牙齿标签")

    rng = rng if rng is not None else random
    selected_face_ids = np.arange(len(faces), dtype=np.int64)
    selected_vertex_ids = np.arange(len(vertices), dtype=np.int64)
    cut_info = {"fallback": True, "reason": "未找到满足约束的裁剪"}

    for _ in range(max_attempts):
        # 按牙位均匀抽样，避免顶点数多的牙齿拥有更高被选概率。
        tooth_label = int(rng.choice(tooth_ids))
        tooth_mask = labels == tooth_label
        tooth_vertex_ids = np.flatnonzero(tooth_mask)
        center_id = int(rng.choice(tooth_vertex_ids.tolist()))
        center = vertices[center_id]

        # 将所有接触该牙的面片顶点也纳入覆盖范围。
        # 单纯覆盖牙齿标签点仍可能在“整面保留”规则下切掉牙颈边界。
        touching_faces = np.any(tooth_mask[faces], axis=1)
        if not np.any(touching_faces):
            continue
        footprint_ids = np.unique(faces[touching_faces].reshape(-1))
        distances = np.linalg.norm(vertices - center, axis=1)
        min_radius = float(distances[footprint_ids].max()) * radius_margin
        max_radius = float(distances.max())
        if min_radius >= max_radius:
            continue

        # 大/中/局部三档均按当前牙弓跨度选择半径，不固定毫米尺寸。
        radius_fraction = rng.uniform(low, high)
        radius = min_radius + (max_radius - min_radius) * radius_fraction
        inside = distances <= radius
        keep_faces = np.all(inside[faces], axis=1)
        kept_face_ids = np.flatnonzero(keep_faces)

        # 避免空裁剪或根本没有裁剪。
        if not 0 < len(kept_face_ids) < len(faces):
            continue
        kept_vertex_ids = np.unique(faces[kept_face_ids].reshape(-1))
        if len(kept_vertex_ids) < min_vertices:
            continue
        # 包括牙齿邻接三角面：不能在面片过滤和顶点重映射后丢牙。
        if not np.all(np.isin(tooth_vertex_ids, kept_vertex_ids)):
            continue

        selected_face_ids = kept_face_ids
        selected_vertex_ids = kept_vertex_ids
        cut_info = {
            "fallback": False,
            "target_fdi": tooth_label,
            "center_vertex_id": center_id,
            "radius": float(radius),
            "radius_fraction": float(radius_fraction),
            "min_radius": min_radius,
        }
        break

    selected_faces = faces[selected_face_ids]
    vertex_id_map = np.full(len(vertices), -1, dtype=np.int64)
    vertex_id_map[selected_vertex_ids] = np.arange(len(selected_vertex_ids))
    remapped_faces = vertex_id_map[selected_faces]

    cropped = SindreMesh([vertices[selected_vertex_ids].copy(), remapped_faces.copy()])
    cropped.set_vertex_labels(labels[selected_vertex_ids].copy())
    cropped.source_vertex_ids = selected_vertex_ids
    cropped.source_face_ids = selected_face_ids
    cropped.cut_info = cut_info
    return cropped


def choose_fdi(vertex_labels, jaw, rng=None):
    """从当前网格已有的 FDI 顶点标签中随机选择一个牙位。

    数据库中的标签本身就是 FDI（0 为牙龈/背景），所以这里直接使用
    顶点标签，不重新推断，也不把标签转换成面片标签。
    """
    rng = rng or random
    labels = np.asarray(vertex_labels).reshape(-1).astype(np.int64, copy=False)
    valid_quadrants = {1, 2, 5, 6} if jaw == "upper" else {3, 4, 7, 8}
    candidates = []
    for value in np.unique(labels):
        if value <= 0:
            continue
        quadrant, tooth = divmod(int(value), 10)
        max_tooth = 5 if quadrant >= 5 else 8
        if quadrant in valid_quadrants and 1 <= tooth <= max_tooth:
            candidates.append(int(value))
    if not candidates:
        return -1
    return int(rng.choice(candidates))


def make_fdi_prompt(sm: SindreMesh, jaw, rng=None):
    """生成 FDI 提示及其对应的顶点目标掩码。"""
    fdi = choose_fdi(sm.vertex_labels, jaw, rng)
    labels = np.asarray(sm.vertex_labels).reshape(-1).astype(np.int64, copy=False)
    target_mask = labels == fdi if fdi > 0 else np.zeros(len(labels), dtype=bool)
    return fdi, target_mask


def make_point_prompt(sm: SindreMesh, rng=None):
    """随机选择一颗牙的一个顶点，V1 只返回一个正点提示。"""
    rng = rng or random
    labels = np.asarray(sm.vertex_labels).reshape(-1).astype(np.int64, copy=False)
    tooth_vertex_ids = np.flatnonzero(labels > 0)
    if len(tooth_vertex_ids) == 0:
        return -1, -1, np.zeros(3, dtype=np.float32), np.zeros(len(labels), dtype=bool)

    prompt_vertex_id = int(rng.choice(tooth_vertex_ids))
    target_fdi = int(labels[prompt_vertex_id])
    target_mask = labels == target_fdi
    prompt_point = np.asarray(sm.vertices[prompt_vertex_id], dtype=np.float32)
    return prompt_vertex_id, target_fdi, prompt_point, target_mask
    






# 预设的是相对半径范围，而非牙位数的硬保证；真实覆盖程度写入每条样本元数据。
CROP_RADIUS_FRACTIONS = {
    "random": (0.0, 1.0),
    "large": (0.65, 0.95),
    "medium": (0.30, 0.70),
    "local": (0.0, 0.35),
}


def choose_crop_mode(
    augment_index: int, augment_count: int, include_full_arch: bool,
) -> str:
    """保证第 0 份是完整牙弓，其余依次大/中/局部裁剪；旧版单份默认随机裁剪。"""
    if augment_count < 1 or not 0 <= augment_index < augment_count:
        raise ValueError("增强次数或增强序号无效")
    if include_full_arch and augment_index == 0:
        return "full"
    if augment_count == 1:
        return "random"
    modes = ("large", "medium", "local")
    ordinal = augment_index - 1 if include_full_arch else augment_index
    return modes[ordinal % len(modes)]


def process(
    reader: Reader,
    index: int = 0,
    *,
    num_points: int = 20000,
    rng: random.Random | None = None,
    augment_index: int = 0,
    augment_count: int = 1,
    include_full_arch: bool = False,
) -> dict:
    """每份原始病例按增强序号生成一个 20k 点样本，保存可追溯裁剪元数据。"""
    if num_points < 1:
        raise ValueError("num_points 必须大于 0")
    crop_mode = choose_crop_mode(augment_index, augment_count, include_full_arch)
    data = reader[index]
    name = str(data["name"])
    jaw = str(data["jaw"])
    if jaw not in ("upper", "lower"):
        raise ValueError(f"无效 jaw={jaw!r}, name={name}")

    sm = SindreMesh([data["vertices"], data["faces"]])
    sm.set_vertex_labels(data["vertex_labels"])
    source_vertex_count = len(sm.vertices)
    source_tooth_count = int(np.unique(
        np.asarray(sm.vertex_labels).reshape(-1)[
            np.asarray(sm.vertex_labels).reshape(-1) > 0
        ]
    ).size)
    if crop_mode != "full":
        sm = random_cut(
            sm, rng=rng, min_vertices=num_points,
            radius_fraction_range=CROP_RADIUS_FRACTIONS[crop_mode],
        )

    points = np.asarray(sm.vertices)
    if len(points) < num_points:
        raise ValueError(f"病例 {name} 点数不足：{len(points)} < {num_points}")
    pts_ids = np.asarray(sample_pcd_farthest(points, num_points), dtype=np.int64)
    if pts_ids.shape != (num_points,) or np.any((pts_ids < 0) | (pts_ids >= len(points))):
        raise ValueError(f"病例 {name} FPS 返回无效索引")
    vertices = points[pts_ids]
    normals = np.asarray(sm.vertex_normals)[pts_ids]
    fdi_labels = np.asarray(sm.vertex_labels).reshape(-1)[pts_ids]
    if not np.any(fdi_labels > 0):
        raise ValueError(f"病例 {name} FPS 后没有有效牙齿点")

    labels = np.asarray(convert_fdi2idx(fdi_labels)).reshape(-1)
    if not np.any(labels > 0):
        raise ValueError(f"病例 {name} FDI 转换后没有有效牙齿标签")
    features = np.column_stack((vertices, normals, labels)).astype(np.float32)
    if features.shape != (num_points, 7) or not np.isfinite(features).all():
        raise ValueError(f"病例 {name} 特征形状或数值异常")

    if DEBUG:
        import vedo
        from sindre.utils3d import labels2colors_freeze
        pcd = vedo.Points(vertices.copy(), c="red")
        pcd.render_points_as_spheres(False)
        pcd.pointcolors = labels2colors_freeze(labels)
        vedo.show(pcd).close()

    cut_info = getattr(sm, "cut_info", {})
    original_labels = np.asarray(sm.vertex_labels).reshape(-1)
    return {
        "name": name, "jaw_code": 0 if jaw == "upper" else 1,
        "features": features,
        "source_index": int(index),
        "augment_index": int(augment_index),
        "augment_count": int(augment_count),
        "crop_mode": crop_mode,
        "crop_fallback": bool(cut_info.get("fallback", False)),
        "crop_radius": float(cut_info.get("radius", 0.0)),
        "radius_fraction": float(cut_info.get("radius_fraction", 0.0)),
        "source_tooth_count": source_tooth_count,
        "retained_tooth_count": int(np.unique(original_labels[original_labels > 0]).size),
        "source_vertex_count": int(source_vertex_count),
        "retained_vertex_count": int(len(points)),
        "vertex_retention_ratio": float(len(points) / source_vertex_count),
    }


@lru_cache(maxsize=2)
def _get_worker_reader(db_path: str) -> Reader:
    """每个 spawn 子进程分别缓存自己的 LMDB Reader，避免逐条重复打开。"""
    return Reader(db_path, multiprocessing=True)


def process_index(task: tuple[str, int, int, int, int, int, bool]) -> dict:
    """多次增强使用独立确定性随机种子，不依赖多进程执行顺序。"""
    db_path, index, augment_index, augment_count, num_points, seed, include_full_arch = task
    rng = random.Random(seed + index * 1_000_003 + augment_index * 7_919)
    return process(
        _get_worker_reader(db_path),
        index=index,
        num_points=num_points,
        rng=rng,
        augment_index=augment_index,
        augment_count=augment_count,
        include_full_arch=include_full_arch,
    )


def generate_dataset(
    input_path: str | Path = DATASETS_PATH,
    output_path: str | Path = OUTPUT_PATH,
    *,
    num_points: int = 20000,
    num_workers: int = 4,
    batch_size: int = 16,
    map_size_mb: int = 1024,
    seed: int = 1024,
    limit: int | None = None,
    augment_count: int = 1,
    include_full_arch: bool | None = None,
) -> dict:
    """由 sindre.lmdb 负责多进程处理及单数据库批量写入。"""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists() or not (input_path.is_file() or input_path.is_dir()):
        raise FileNotFoundError(f"输入 LMDB 不存在：{input_path}")
    if input_path == output_path or output_path.exists():
        raise FileExistsError(f"输出路径已存在或与输入相同，拒绝覆盖：{output_path}")
    if num_points < 1 or num_workers < 1 or batch_size < 1 or map_size_mb < 1:
        raise ValueError("点数、进程数、批大小和 LMDB 容量均必须大于 0")
    if augment_count < 1:
        raise ValueError("augment_count 必须大于 0")
    # 单份默认保留旧行为（随机裁剪）；多次增强默认第 0 份为完整牙弓。
    if include_full_arch is None:
        include_full_arch = augment_count > 1
    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于 0")

    reader = Reader(str(input_path), multiprocessing=True)
    try:
        total = len(reader)
    finally:
        reader.close()

    count = total if limit is None else min(limit, total)
    if count == 0:
        raise ValueError("输入 LMDB 没有样本")
    tasks = [
        (str(input_path), index, aug, augment_count, num_points, seed, include_full_arch)
        for index in range(count) for aug in range(augment_count)
    ]
    # 每条至少包含 20k×7×float32，4 次增强通常需要超过 4GB 的 map。
    min_map_mb = math.ceil(len(tasks) * num_points * 7 * 4 * 1.5 / (1024 * 1024)) + 256
    effective_map_mb = max(map_size_mb, min_map_mb)
    if effective_map_mb > map_size_mb:
        print(f"LMDB map-size 自动增加：{map_size_mb} -> {effective_map_mb} MiB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parallel = LMDBParallel_Batch(
        output_dir=str(output_path),
        map_size_limit=effective_map_mb,
        num_processes=min(num_workers, len(tasks)),
        batch_size=batch_size,
        multiprocessing=True,
    )
    result = parallel.process_files(tasks, process_index)
    if not result["success"]:
        details = "; ".join(map(str, result.get("errors", [])[:5]))
        raise RuntimeError(
            f"LMDB 批量生成失败，可能存在不完整输出 {output_path}：{details}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="保牙随机裁剪 + 并行 LMDB 数据预处理")
    parser.add_argument("--input", type=Path, default=DATASETS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--map-size-mb", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--augment-count", type=int, default=1, help="每份原始病例生成的总样本数；4 代表 1 完整 + 3 不同尺度裁剪")
    arch = parser.add_mutually_exclusive_group()
    arch.add_argument("--include-full-arch", dest="include_full_arch", action="store_true", help="每份病例的第 0 个增强样本为完整牙弓")
    arch.add_argument("--no-full-arch", dest="include_full_arch", action="store_false", help="全部生成随机裁剪样本")
    parser.set_defaults(include_full_arch=None)
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个样本")
    args = parser.parse_args()
    result = generate_dataset(
        args.input,
        args.output,
        num_points=args.num_points,
        num_workers=args.workers,
        batch_size=args.batch_size,
        map_size_mb=args.map_size_mb,
        seed=args.seed,
        limit=args.limit,
        augment_count=args.augment_count,
        include_full_arch=args.include_full_arch,
    )
    print(f"批量生成完成：{result['processed']} 个增强样本 -> {args.output}")


if __name__ == "__main__":
    main()
