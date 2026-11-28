"""FDG v2 训练数据集：分层采样稀疏体素及牙颌条件。"""

from __future__ import annotations

from pathlib import Path
import os
from torch import Tensor

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
        padding_ratio: float = 0.08,
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
        self.db_path = str(db_path)
        self._reader = Reader(self.db_path)
        self._reader_pid = os.getpid()
        if (self.reader.get_meta("schema_version") != "toothforge3d_fdg_v3"
                or self.reader.get_meta("build_complete") is not True
                or self.reader.get_meta("coordinate_frame") != "scan_arch_margin_v3"):
            self.reader.close()
            raise ValueError(
                "FDG LMDB 必须是 v3 scan_arch_margin_v3 坐标规范；旧库必须重建；"
                "请用最新版 process.FDGPreprocessor.build_database 重建"
            )
        fdg_config = self.reader.get_meta("fdg_config")
        if not isinstance(fdg_config, dict) or int(fdg_config.get("resolution", -1)) != resolution or not np.isclose(float(fdg_config.get("padding_ratio", -1)), padding_ratio):
            self.reader.close()
            raise ValueError("FDG 数据库分辨率或 padding_ratio 与模型配置不一致")
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
        self._length = len(self.reader)
        self.close()

    @property
    def reader(self):
        if self._reader is None or self._reader_pid != os.getpid():
            from sindre.lmdb import Reader
            self.close()
            self._reader = Reader(self.db_path)
            self._reader_pid = os.getpid()
        return self._reader

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_reader'] = None
        state['_reader_pid'] = None
        state['_worker_rng'] = None
        return state

    def get_source_groups(self) -> tuple[int, ...]:
        """获取原始病例分组，用于防止尺寸增强副本泄漏到验证集。

        Returns:
            与 LMDB 样本索引一一对应的原始病例组编号。
        """
        return self.source_groups

    def __len__(self):
        return self._length

    def __getitem__(self, index: int | tuple[int, bool]) -> dict[str, Tensor]:
        augment = self.augment
        if isinstance(index, tuple):
            index, augment = index
            if not isinstance(augment, (bool, np.bool_)):
                raise TypeError("样本索引中的 augment 必须是布尔值")
        index = int(index)
        row = self.reader[index]
        worker = get_worker_info()
        if not augment:
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
        if augment and len(jaw):
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
                for tooth in np.unique(labels[(labels > 0) & (labels <= 16)]):
                    if rng.random() < self.tooth_dropout:
                        keep = labels != tooth
                        jaw, labels = jaw[keep], labels[keep]
        present = len(jaw) > 0
        if present:
            ids = rng.choice(len(jaw), size=self.jaw_points, replace=len(jaw) < self.jaw_points)
            jaw, labels = jaw[ids].copy(), labels[ids].copy()
            if augment and rng.random() < 0.25:
                labels[:] = 17  # 模拟真实无标签输入，不伪造背景
            if augment and self.jaw_noise_mm:
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
        if augment and rng.random() < self.view_dropout:
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
            if original_design and augment
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
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._reader_pid = None
