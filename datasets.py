"""FDG v2 训练数据集：分层采样稀疏体素及牙颌条件。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


class FDGDataset(Dataset):
    """从 Sindre LMDB 读取开放冠 FDG 标签和配对条件。

    同病例长宽高增强版本使用 source_groups 进行独立验证集划分。
    在线仅增强邻牙裁剪、邻牙缺失/轻噪声及离线视图特征 Dropout。
    不随机改动 size/margin/FDG 标签；尺寸变化已在预处理阶段配对生成。

    Args:
        db_path: 已完成构建的 FDG LMDB 路径。
        resolution: FDG 细体素分辨率。
        coarse_resolution: 粗结构分辨率。
        query_count: 每例细体素正负混合查询点数量。
        jaw_points: 每例输入 PTv3 的同颌点数。
        augment: 是否使用在线邻牙增强。
        tooth_dropout: 随机删除一整颗邻牙的概率。
        jaw_noise_mm: 同颌点云位置高斯噪声标准差，单位 mm。
        view_dropout: 随机屏蔽一个预计算 DINO 视角的概率。
        size_dropout: 训练时隐藏原始设计冠长宽高的概率；增强变体不隐藏。
        seed: 随机种子。
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
        tooth_dropout: float = 0.20,
        jaw_noise_mm: float = 0.02,
        view_dropout: float = 0.10,
        size_dropout: float = 0.65,
        seed: int = 1024,
    ):
        from sindre.lmdb import Reader

        if (resolution < coarse_resolution or coarse_resolution < 4
                or resolution % coarse_resolution or query_count < 4 or jaw_points < 1):
            raise ValueError("FDG 分辨率/查询数/牙颌点数参数无效")
        self.reader = Reader(str(db_path))
        if (self.reader.get_meta("schema_version") != "toothforge3d_fdg_v2"
                or self.reader.get_meta("build_complete") is not True
                or self.reader.get_meta("coordinate_frame") != "margin_only_v2"):
            self.reader.close()
            raise ValueError(
                "FDG LMDB 必须是 v2 margin_only_v2 坐标规范；旧库必须重建；"
                "请用最新版 utils.CrownFDG.build_database 重建"
            )
        self.resolution = resolution
        self.coarse_resolution = coarse_resolution
        self.query_count = query_count
        self.jaw_points = jaw_points
        self.augment = augment
        if (not 0 <= tooth_dropout <= 1 or jaw_noise_mm < 0
                or not 0 <= view_dropout < 1 or not 0 <= size_dropout <= 1):
            raise ValueError("牙齿缺失增强、坐标噪声和视图 Dropout 参数无效")
        self.tooth_dropout = tooth_dropout
        self.jaw_noise_mm = jaw_noise_mm
        self.view_dropout = view_dropout
        self.size_dropout = size_dropout
        groups = self.reader.get_meta("source_groups")
        if groups is None or len(groups) != len(self.reader):
            self.reader.close()
            raise ValueError("FDG 数据库缺少原始病例分组信息，请重新构建 FDG DB")
        self.source_groups = tuple(int(item) for item in groups)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # 持久 worker 的 epoch 不会重置 seed；复用前进的随机数状态。
        self._worker_rng = None

    def get_source_groups(self) -> tuple[int, ...]:
        """获取原始病例分组，用于防止尺寸增强副本泄漏到验证集。

        Returns:
            与 LMDB 样本索引一一对应的原始病例组编号。
        """
        return self.source_groups

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.reader[index]
        worker = get_worker_info()
        if not self.augment:
            rng = np.random.default_rng(self.seed + index * 7919)
        elif worker is None:
            rng = self.rng
        else:
            if self._worker_rng is None:
                self._worker_rng = np.random.default_rng(worker.seed)
            rng = self._worker_rng
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
        if jaw.ndim != 2 or jaw.shape[1] != 6 or labels.shape != (len(jaw),):
            raise ValueError("同颌点云与逐点牙位不匹配")
        if self.augment and len(jaw):
            # 仅删减可观察的邻牙，不动颈缘线、FDI 和 MD/BL/H 监督。
            mode = rng.choice(
                ("full", "large", "medium", "local", "none"),
                p=(0.30, 0.25, 0.25, 0.15, 0.05),
            )
            if mode == "none":
                jaw, labels = jaw[:0], labels[:0]
            elif mode != "full":
                center = np.asarray(row["margin_line"], dtype=np.float32).mean(0)
                distances = np.linalg.norm(jaw[:, :3] - center, axis=1)
                quantiles = {
                    "large": (0.60, 0.95), "medium": (0.35, 0.75),
                    "local": (0.10, 0.45),
                }
                low, high = quantiles[mode]
                radius = np.quantile(distances, rng.uniform(low, high))
                mask = distances <= radius
                jaw, labels = jaw[mask], labels[mask]
            if len(jaw) and self.tooth_dropout:
                for tooth in np.unique(labels[labels > 0]):
                    if rng.random() < self.tooth_dropout:
                        keep = labels != tooth
                        jaw, labels = jaw[keep], labels[keep]
        present = len(jaw) > 0
        if present:
            ids = rng.choice(len(jaw), size=self.jaw_points, replace=len(jaw) < self.jaw_points)
            jaw, labels = jaw[ids].copy(), labels[ids].copy()
            if self.augment and self.jaw_noise_mm:
                jaw[:, :3] += rng.normal(
                    0, self.jaw_noise_mm, size=(len(jaw), 3),
                ).astype(np.float32)
        else:
            jaw = np.zeros((self.jaw_points, 6), np.float32)
            labels = np.zeros((self.jaw_points,), np.int64)
        if "view_features" not in row or row["view_features"] is None:
            raise ValueError("FDG 数据库缺少预计算 DINOv3 view_features")
        view = np.asarray(row["view_features"], dtype=np.float32).copy()
        if view.ndim != 3 or view.shape[0] != 3 or not np.isfinite(view).all():
            raise ValueError("DINOv3 特征必须是有效的 [3,T,C]")
        # 一次最多屏蔽一个视角；不重新渲染，也不随机破坏目标尺寸。
        if self.augment and rng.random() < self.view_dropout:
            view[int(rng.integers(0, 3))] = 0
        size_mm = np.asarray(row["size"], dtype=np.float32)
        if size_mm.shape != (3,) or not np.isfinite(size_mm).all() or (size_mm <= 0).any():
            raise ValueError("MD/BL/H 条件必须为3个正的有限毫米值")
        # 同病例所有尺寸增强版本共享输入三视图与邻牙。
        # 对增强版本隐藏 size，会制造同样输入->不同冠的冲突监督。
        # 因此仅原始冠参与自动尺寸先验；增强冠始终明确提供尺寸。
        original_design = int(row.get("variant_index", 0)) == 0
        size_known = (
            bool(rng.random() >= self.size_dropout)
            if original_design and self.augment
            else (not original_design)
        )
        return {
            "jaw": torch.tensor(row["jaw"], dtype=torch.long),
            "target_position": torch.tensor(row["target_position"], dtype=torch.long),
            "margin_line": torch.as_tensor(np.asarray(row["margin_line"]), dtype=torch.float32),
            "size": torch.as_tensor(size_mm, dtype=torch.float32),
            # size 保留 GT 专供监督；当 size_known=False 时模型不得读取它作输入。
            "size_known": torch.tensor(size_known),
            "size_prior_mask": torch.tensor(original_design),
            "jaw_points": torch.from_numpy(jaw),
            "jaw_labels": torch.as_tensor(labels, dtype=torch.long),
            "jaw_present": torch.tensor(present),
            "view_features": torch.as_tensor(view, dtype=torch.float32),
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
