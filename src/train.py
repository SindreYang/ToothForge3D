"""专用开放冠 FDG 训练流程。

预处理通过 Sindre LMDB 缓存 DINOv3/FDG；训练阶段只进行 GPU 前向与监督。
包含按原始病例分组验证、BF16、tqdm、SindrePy Checkpoint 和可选 Profiler。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .datasets import FDGDataset
from .model import LightFDGConfig, LightFDGModel, create_light_fdg_model
from .losses import CrownFDGLoss, FDGLossConfig
from .utils import CrownFDG, FDGConfig
from sindre.ai.utils import set_global_seeds, save_checkpoint, load_checkpoint

log = logging.getLogger("ToothForge3D.train")


@dataclass(frozen=True)
class FDGTrainingConfig:
    """训练配置；尺寸条件增强只在预处理阶段改变目标几何。

    Attributes:
        epochs: 总训练轮数。
        batch_size: GPU 小批大小。
        query_count: 每病例细体素查询数量。
        margin_every: 每隔多少个训练批计算一次 CUDA 颈缘线 Loss；0 为关闭。
        size_dropout: 随机隐藏用户 MD/BL/H 输入的概率，0～1。
        profile_steps: Profiler 激活记录步数，0 为关闭。
    """

    epochs: int = 100
    batch_size: int = 2
    query_count: int = 2048
    jaw_points: int = 2048
    num_workers: int = 2
    val_fraction: float = 0.10
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    use_bf16: bool = True
    margin_every: int = 16
    size_dropout: float = 0.65
    seed: int = 1024
    checkpoint_dir: str = "checkpoints/toothforge_fdg"
    device: str = "auto"
    resume: bool = False
    profile_steps: int = 0
    generation_val_cases: int = 4  # 独立 CUDA 自主生成验证数，0 关闭
    generation_margin_mm: float = 0.5

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.query_count, self.jaw_points,
               self.grad_accum_steps) <= 0:
            raise ValueError("Epoch/Batch/query/jaw/accum 必须为正数")
        if (self.num_workers < 0 or self.margin_every < 0 or self.profile_steps < 0
                or not 0 < self.val_fraction < 1
                or self.lr <= 0 or self.grad_clip <= 0 or self.weight_decay < 0
                or self.generation_val_cases < 0 or self.generation_margin_mm <= 0
                or not 0 <= self.size_dropout <= 1):
            raise ValueError("训练数值参数无效")


def get_margin_meshes(
    model: LightFDGModel,
    cache: dict,
    batch: dict,
    converter: CrownFDG,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """在真实 FDG 拓扑上计算用于辅助训练的开放冠 Mesh。

    Args:
        model: 当前 LightFDGModel。
        cache: 含批量空间特征和尺寸的模型中间结果。
        batch: 包含真实 FDG 坐标与相交标签的监督批次。
        converter: 原生 O-Voxel FDG 转换工具。

    Returns:
        每个病例对应的预测顶点和三角面列表。

    Raises:
        RuntimeError: CUDA/O-Voxel 不可用。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("margin_every>0 需要可用的 O-Voxel CUDA，CPU 请设置 margin_every=0")
    result = []
    for index, gt_coords_cpu in enumerate(batch["fdg_coords"]):
        coords = gt_coords_cpu.to(cache["volume"].device, non_blocking=True)
        view = {
            "volume": cache["volume"][index:index+1],
            "global": cache["global"][index:index+1],
            "size": cache["size"][index:index+1],
            "scale_mm": cache["scale_mm"][index:index+1],
        }
        predicted_offsets = []
        for start in range(0, len(coords), 8192):
            out = model.predict_at_coords(view, coords[None, start:start+8192])
            predicted_offsets.append(out["dual_offsets"][0].float())
        offsets = torch.cat(predicted_offsets)
        record = {
            "coords": gt_coords_cpu, "intersected": batch["fdg_intersected"][index],
            "center_mm": batch["fdg_center_mm"][index],
            "scale_mm": batch["fdg_scale_mm"][index].item(),
            "resolution": model.config.resolution,
        }
        verts, faces = converter.get_mesh(record, dual_offsets=offsets)
        result.append((verts, faces))
    return result


class FDGTrainer:
    """持有训练配置、网络、优化器、数据加载和自主生成验证的训练器。"""

    def __init__(self, *, config=None, model_config=None, loss_config=None, model=None):
        config = config or FDGTrainingConfig()
        model_config = model_config or LightFDGConfig()
        set_global_seeds(config.seed)
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto" else config.device,
        )
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        if config.margin_every and device.type != "cuda":
            raise ValueError("没有 CUDA 时请设置 FDGTrainingConfig(margin_every=0)")

        self.config = config
        self.model_config = model_config
        self.loss_config = loss_config
        self.model = model
        self.device = device
        self.converter = CrownFDG(FDGConfig(resolution=model_config.resolution,
                                            padding_ratio=model_config.padding_ratio))

    def make_dataset(self, path, augment):
        config, model_config = self.config, self.model_config
        return FDGDataset(
            path, resolution=model_config.resolution,
            coarse_resolution=model_config.coarse_resolution,
            padding_ratio=model_config.padding_ratio,
            query_count=config.query_count, jaw_points=config.jaw_points,
            augment=augment, size_dropout=config.size_dropout, seed=config.seed,
        )

    def make_loader(self, dataset, training):
        config = self.config
        return DataLoader(
            dataset, batch_size=config.batch_size, shuffle=training,
            num_workers=config.num_workers, pin_memory=self.device.type == "cuda",
            collate_fn=FDGDataset.collate,
            persistent_workers=config.num_workers > 0,
        )

    @torch.inference_mode()
    def validate_generation(self, loader):
        """不读取 GT 拓扑进行预测，仅在生成后比较真实 FDG 并检查开放冠。"""
        self.model.eval()
        total = valid = decoded = 0
        iou_sum = margin_sum = 0.
        limit = self.config.generation_val_cases
        resolution = self.model_config.resolution
        keys = ("view_features", "jaw", "target_position", "margin_line",
                "jaw_points", "jaw_labels", "jaw_present", "view_mask", "view_scale_mm", "size", "size_known")
        for batch in loader:
            for i in range(len(batch["jaw"])):
                if "size_prior_mask" in batch and not bool(batch["size_prior_mask"][i]):
                    continue
                total += 1
                condition = {k: batch[k][i:i+1].to(self.device) for k in keys if k in batch}
                condition["size_known"] = torch.zeros(1, dtype=torch.bool, device=self.device)
                if "size" in condition:
                    condition["size"] = torch.zeros_like(condition["size"])
                try:
                    pred = self.model.predict(condition)[0]
                    if pred["valid"]:
                        pc = pred["coords"].long().cpu()
                        gt = batch["fdg_coords"][i].long().cpu()
                        pred_ids = pc[:, 0] * resolution**2 + pc[:, 1] * resolution + pc[:, 2]
                        gt_ids = gt[:, 0] * resolution**2 + gt[:, 1] * resolution + gt[:, 2]
                        intersection = int(torch.isin(pred_ids, gt_ids).sum())
                        iou_sum += intersection / max(len(pred_ids) + len(gt_ids) - intersection, 1)
                        # FDG 尺度仅取自输入颈缘线，不从监督记录读取。
                        center, scale = self.converter.get_condition_frame(condition["margin_line"][0].cpu().numpy())
                        vertices, faces = self.converter.get_mesh(
                            {"coords": pred["coords"], "center_mm": center, "scale_mm": scale,
                             "resolution": resolution},
                            dual_offsets=pred["dual_offsets"],
                            intersected_logits=pred["intersected_logits"],
                        )
                        self.converter.get_mesh_quality(vertices.cpu().numpy(), faces.cpu().numpy())
                        margin = self.criterion.get_margin_loss(vertices, faces, condition["margin_line"][0])
                        error = float(margin["max_distance_mm"])
                        decoded += 1
                        margin_sum += error
                        valid += int(error <= self.config.generation_margin_mm)
                except (ValueError, RuntimeError) as exc:
                    log.warning("自主生成验证第 %d 例失败: %s", total, exc)
                if total >= limit:
                    break
            if total >= limit:
                break
        return {"val_generation_count": total, "val_generation_valid_rate": valid / max(total, 1),
                "val_generation_fdg_iou": iou_sum / max(total, 1),
                "val_generation_margin_max_mm": margin_sum / decoded if decoded else None}

    def run(self, train_db: str | Path, *, val_db: str | Path | None = None) -> dict:
        """运行训练并返回监督验证及自主生成指标。"""
        config, model_config, loss_config = self.config, self.model_config, self.loss_config
        device, model = self.device, self.model
        if val_db is not None and Path(train_db).resolve() == Path(val_db).resolve():
            raise ValueError("显式 val_db 必须与 train_db 不同")
        train_set = self.make_dataset(train_db, True)
        val_set = train_set
        try:
            if val_db is not None:
                val_set = self.make_dataset(val_db, False)
                overlap = getattr(train_set, "order_groups", frozenset()) & getattr(val_set, "order_groups", frozenset())
                if overlap:
                    raise ValueError(f"显式训练/验证库存在 {len(overlap)} 个重复订单分组")
            if not len(train_set) or not len(val_set):
                raise ValueError("FDG 数据库不能为空")
            if val_db is None:
                # 同一原始病例的尺寸增强副本必须处于同一集合，杜绝验证数据泄漏。
                source_groups = train_set.get_source_groups()
                unique_groups = sorted(set(source_groups))
                if len(unique_groups) < 2:
                    raise ValueError("至少两组不同原始病例才可自动划分训练/验证")
                generator = torch.Generator().manual_seed(config.seed)
                order = torch.randperm(len(unique_groups), generator=generator).tolist()
                val_count = max(
                    1, min(len(unique_groups) - 1, round(len(unique_groups) * config.val_fraction))
                )
                val_groups = {unique_groups[i] for i in order[:val_count]}
                train_indices = [
                    (i, True) for i, key in enumerate(source_groups) if key not in val_groups
                ]
                val_indices = [
                    (i, False) for i, key in enumerate(source_groups) if key in val_groups
                ]
                train_data = Subset(train_set, train_indices)
                val_data = Subset(val_set, val_indices)
            else:
                train_data, val_data = train_set, val_set
            first = train_data[0]
            if model is None:
                model = create_light_fdg_model(
                    image_dim=first["view_features"].shape[-1],
                    margin_points=first["margin_line"].shape[0],
                    config=model_config,
                )
            model = model.to(device)
            self.model = model
            criterion = CrownFDGLoss(loss_config).to(device)
            self.criterion = criterion
            converter = self.converter

            train_loader = self.make_loader(train_data, True)
            val_loader = self.make_loader(val_data, False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
            self.optimizer = optimizer
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs,
            )
            self.scheduler = scheduler
            output = Path(config.checkpoint_dir)
            output.mkdir(parents=True, exist_ok=True)
            start_epoch = 0
            best = float("inf")
            if config.resume:
                saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
                if saved.get("fdg_schema") != "toothforge3d_fdg_v3" or saved.get("model_config") != asdict(model_config):
                    raise ValueError("恢复训练权重的 FDG 版本或模型配置不匹配")
                start_epoch, _, extra = load_checkpoint(
                    path=str(output / "last.pt"), net=model, optimizer=optimizer,
                    strict=True, map_location=str(device),
                )
                best = float(extra.get("best_val", best))
                if "scheduler" in extra:
                    scheduler.load_state_dict(extra["scheduler"])
            bf16 = (
                config.use_bf16 and device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            )
            profile = None
            if config.profile_steps:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profile = torch.profiler.profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(
                        wait=0, warmup=1, active=config.profile_steps, repeat=1,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(output / "profiler")
                    ),
                    record_shapes=True, profile_memory=True,
                )
                profile.__enter__()
            metrics = {}
            try:
                for epoch in range(start_epoch, config.epochs):
                    for training, data_loader in (
                        (True, train_loader), (False, val_loader)
                    ):
                        model.train(training)
                        sums = {"loss": 0., "coarse": 0., "fine": 0.,
                                "offset": 0., "intersection": 0.,
                                "coarse_recall": 0., "fine_precision": 0.}
                        size_prior_sum = 0.0
                        size_prior_mae = 0.0
                        size_prior_count = 0
                        count = 0
                        mesh_count = 0
                        mesh_sums = {"size": 0., "size_mae_mm": 0., "margin_mm": 0.}
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                        stage = "train" if training else "val"
                        bar = tqdm(
                            data_loader,
                            desc=f"FDG epoch {epoch + 1}/{config.epochs} {stage}",
                            dynamic_ncols=True,
                        )
                        for step, cpu_batch in enumerate(bar):
                            # 完整 FDG 标签只在间隔 Mesh Loss 时上传到 GPU。
                            cpu_only = {
                                "fdg_coords", "fdg_offsets", "fdg_intersected",
                                "fdg_center_mm", "fdg_scale_mm",
                            }
                            batch = {
                                key: value if key in cpu_only else
                                [tensor.to(device, non_blocking=True) for tensor in value]
                                if isinstance(value, list) else
                                value.to(device, non_blocking=True)
                                for key, value in cpu_batch.items()
                            }
                            with torch.set_grad_enabled(training):
                                with torch.autocast(
                                    device_type=device.type, dtype=torch.bfloat16,
                                    enabled=bf16,
                                ):
                                    prediction = model(batch, batch["fine_coords"])
                                # 几何和分类损失强制 fp32，避免 sigmoid 和边界距离半精度不稳定。
                                margin_meshes = None
                                if config.margin_every and step % config.margin_every == 0:
                                    margin_meshes = get_margin_meshes(
                                        model, prediction, batch, converter,
                                    )
                                terms = criterion(
                                    prediction, batch,
                                    margin_meshes=margin_meshes,
                                )
                                if not bool(torch.isfinite(terms["loss"])):
                                    raise FloatingPointError(
                                        f"epoch={epoch + 1} {stage} step={step} Loss 无效"
                                    )
                                if training:
                                    group_start = (
                                        step // config.grad_accum_steps
                                    ) * config.grad_accum_steps
                                    group_size = min(
                                        config.grad_accum_steps,
                                        len(data_loader) - group_start,
                                    )
                                    (terms["loss"] / group_size).backward()
                                    if (
                                        (step + 1) % config.grad_accum_steps == 0
                                        or step + 1 == len(data_loader)
                                    ):
                                        grad_norm = torch.nn.utils.clip_grad_norm_(
                                            model.parameters(), config.grad_clip,
                                            error_if_nonfinite=True,
                                        )
                                        optimizer.step()
                                        optimizer.zero_grad(set_to_none=True)
                            n = batch["jaw"].shape[0]
                            count += n
                            # 只同步一次 GPU 标量，避免 7 次 .item() 阻塞 kernel 队列。
                            values = torch.stack([
                                terms[key].detach().float() for key in sums
                            ]).cpu().tolist()
                            for key, value in zip(sums, values):
                                sums[key] += value * n
                            if "size_prior" in terms:
                                # 无需每步渲染或解码 Mesh，就能监控自动尺寸估计质量。
                                prior_vals = torch.stack([
                                    terms["size_prior"].detach().float(),
                                    terms["size_prior_mae_mm"].detach().float(),
                                ]).cpu().tolist()
                                # 只统计原始冠，人工增强版本不参与先验尺寸 MAE。
                                valid_prior = int(cpu_batch["size_prior_mask"].sum())
                                size_prior_count += valid_prior
                                size_prior_sum += prior_vals[0] * valid_prior
                                size_prior_mae += prior_vals[1] * valid_prior
                            if "size" in terms:
                                # O-Voxel Mesh Loss 每 margin_every 步才计算，单独统计，
                                # 不把未评估的步骤错误地当作尺寸损失为0。
                                mesh_values = torch.stack([
                                    terms[key].detach().float() for key in mesh_sums
                                ]).cpu().tolist()
                                mesh_count += n
                                for key, value in zip(mesh_sums, mesh_values):
                                    mesh_sums[key] += value * n
                            bar.set_postfix(
                                loss=f"{sums['loss'] / count:.4f}",
                                coarse=f"{sums['coarse'] / count:.3f}",
                                fine=f"{sums['fine'] / count:.3f}",
                                offset=f"{sums['offset'] / count:.3f}",
                                recall=f"{sums['coarse_recall'] / count:.2f}",
                                size=f"{mesh_sums['size_mae_mm'] / mesh_count:.3f}mm"
                                     if mesh_count else "-",
                                auto_size=f"{size_prior_mae / size_prior_count:.3f}mm"
                                          if size_prior_count else "-",
                                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                            )
                            if profile is not None and training:
                                profile.step()
                        if not count:
                            raise ValueError("训练或验证 DataLoader 为空")
                        metrics.update({
                            f"{stage}_{key}": value / count
                            for key, value in sums.items()
                        })
                        if size_prior_count:
                            metrics[f"{stage}_size_prior"] = size_prior_sum / size_prior_count
                            metrics[f"{stage}_size_prior_mae_mm"] = (
                                size_prior_mae / size_prior_count
                            )
                        if mesh_count:
                            metrics.update({
                                f"{stage}_{key}": value / mesh_count
                                for key, value in mesh_sums.items()
                            })
                    if device.type == "cuda" and config.generation_val_cases:
                        metrics.update(self.validate_generation(val_loader))
                    scheduler.step()
                    val_loss = metrics["val_loss"]
                    if val_loss < best:
                        best = val_loss
                        save_checkpoint(
                            save_path=str(output / "best.pt"),
                            network=model, optimizer=optimizer,
                            loss=val_loss, curr_iter=epoch + 1,
                            extra_info={
                                "best_val": best,
                                "fdg_schema": "toothforge3d_fdg_v3",
                                "train_config": asdict(config),
                                "model_config": asdict(model_config),
                                "loss_config": asdict(criterion.config),
                            }, save_best_only=False,
                        )
                    save_checkpoint(
                        save_path=str(output / "last.pt"),
                        network=model, optimizer=optimizer,
                        loss=val_loss, curr_iter=epoch + 1,
                        extra_info={
                            "best_val": best, "scheduler": scheduler.state_dict(),
                            "fdg_schema": "toothforge3d_fdg_v3",
                            "train_config": asdict(config),
                            "model_config": asdict(model_config),
                            "loss_config": asdict(criterion.config),
                        }, save_best_only=False,
                    )
                    log.info(
                        "FDG %d: train %.5f, val %.5f, recall %.3f, best %.5f",
                        epoch + 1, metrics["train_loss"], val_loss,
                        metrics["val_coarse_recall"], best,
                    )
            finally:
                if profile is not None:
                    profile.__exit__(None, None, None)
            return {
                **metrics, "best_val":best,
                "best_checkpoint":str(output / "best.pt"),
                "last_checkpoint":str(output / "last.pt"),
            }
        finally:
            train_set.close()
            if val_set is not train_set:
                val_set.close()



def train(
    train_db: str | Path, *, val_db: str | Path | None = None,
    config: FDGTrainingConfig | None = None,
    model_config: LightFDGConfig | None = None,
    loss_config: FDGLossConfig | None = None,
    model: LightFDGModel | None = None,
) -> dict:
    """通过显式参数创建训练器并执行一次训练。"""
    return FDGTrainer(config=config, model_config=model_config, loss_config=loss_config, model=model).run(
        train_db, val_db=val_db,
    )


def main(
    train_db: str | Path,
    val_db: str | Path | None = None,
    *, epochs: int = 100, batch_size: int = 2,
    margin_every: int = 16, num_workers: int = 2,
):
    """通过 Python 显式参数启动 FDG 训练，无 CLI/argparse。

    Args:
        train_db: FDG 训练库路径。
        val_db: 可选独立验证库路径。
        epochs: 训练轮数。
        batch_size: 单次小批大小。
        margin_every: O-Voxel 开放边界 Loss 的计算间隔。
        num_workers: 数据加载子进程数。

    Returns:
        训练与验证指标、最佳权重和最近权重路径。
    """
    return train(
        train_db, val_db=val_db,
        config=FDGTrainingConfig(
            epochs=epochs, batch_size=batch_size,
            num_workers=num_workers, margin_every=margin_every,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(
        "请在 Python 中调用 train.main('data/toothforge_fdg_v2.db')"
    )
