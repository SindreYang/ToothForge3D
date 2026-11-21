"""ToothForge3D 一阶段 Occupancy/SDF 训练、验证与 tqdm 进度。

训练真实的 MCDG 多模态条件编码器 + 旧私有口腔库启发的隐式查询头；
TRELLIS.2 Flow 不在这里伪造 x_0 latent，需官方 Structure/Shape VAE
预编码后另行启动 Flow 训练。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from datasets import MyDataset
from losses import CrownLossConfig, CrownReconstructionLoss
from mcdg import (
    CrownImplicitDecoder, MCDGConditionEncoder, MCDGReconstructionModel,
)
from sindre.ai.utils import (
    load_checkpoint, save_checkpoint, set_global_seeds,
)

log = logging.getLogger("ToothForge3D.train")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    val_fraction: float = 0.10
    num_workers: int = 0
    hidden_dim: int = 256
    num_heads: int = 8
    view_tokens: int = 16
    jaw_tokens: int = 16
    jaw_points: int = 4096
    train_queries: int = 4096
    val_queries: int = 8192
    query_chunk_size: int = 512
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    seed: int = 1024
    use_amp: bool = True
    device: str = "auto"
    checkpoint_dir: str = "checkpoints/toothforge_implicit"
    resume: bool = False

    def __post_init__(self):
        positive_ints = (
            self.epochs, self.batch_size, self.hidden_dim, self.num_heads,
            self.view_tokens, self.jaw_tokens, self.jaw_points,
            self.train_queries, self.val_queries,
            self.query_chunk_size, self.grad_accum_steps,
        )
        if min(positive_ints) < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("正整数训练参数非法，或 hidden_dim 不能整除 num_heads")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("优化器与梯度截断参数非法")
        if not 0 < self.val_fraction < 1 or self.num_workers < 0:
            raise ValueError("val_fraction 应在 (0,1)，num_workers 不能小于 0")


def create_model(
    *,
    image_dim: int,
    margin_points: int,
    config: TrainingConfig,
    point_encoder: torch.nn.Module | None = None,
) -> MCDGReconstructionModel:
    """仅在未提供 PTv3 时由 SindrePy 实例化，不在项目内重写 Transformer。"""
    if point_encoder is None:
        from sindre.ai.vision3d.pointcloud_utils.point_transformerV3 import (
            PointTransformerV3,
        )
        point_encoder = PointTransformerV3(
            in_channels=6, cls_mode=False, enable_flash=False,
        )
    encoder = MCDGConditionEncoder(
        image_dim=image_dim,
        hidden_dim=config.hidden_dim,
        point_dim=64,  # SindrePy PTv3 默认 dec_channels[0] = 64。
        num_heads=config.num_heads,
        margin_points=margin_points,
        view_tokens=config.view_tokens,
        jaw_tokens=config.jaw_tokens,
        point_encoder=point_encoder,
    )
    decoder = CrownImplicitDecoder(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        query_chunk_size=config.query_chunk_size,
    )
    return MCDGReconstructionModel(encoder, decoder)


def train(
    train_db: str | Path,
    *,
    val_db: str | Path | None = None,
    config: TrainingConfig | None = None,
    loss_config: CrownLossConfig | None = None,
    model: MCDGReconstructionModel | None = None,
) -> dict[str, float | str]:
    """完整阶段一训练：单机 CUDA/CPU，tqdm、验证、保存 best/last。

    若未传 val_db，则以**样本**级稳定划分同一个 LMDB。正式跨病例、
    跨患者评估应先生成严格分组的 train/val 两个 DB，防止来源泄漏。
    """
    config = config or TrainingConfig()
    set_global_seeds(config.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto" else config.device
    )
    train_set = MyDataset(
        train_db, augment=True, jaw_points=config.jaw_points,
        query_count=config.train_queries, include_images=False,
        include_mesh=False, seed=config.seed,
    )
    val_set = MyDataset(
        val_db if val_db is not None else train_db,
        augment=False, jaw_points=config.jaw_points,
        query_count=config.val_queries, include_images=False,
        include_mesh=False, seed=config.seed,
    )
    try:
        if not len(train_set) or not len(val_set):
            raise ValueError("训练/验证数据库不能为空")
        if val_db is None:
            if len(train_set) < 2:
                raise ValueError("至少两个病例才能自动划分 train / validation")
            generator = torch.Generator().manual_seed(config.seed)
            indices = torch.randperm(len(train_set), generator=generator).tolist()
            val_size = min(max(1, round(config.val_fraction * len(indices))), len(indices) - 1)
            train_data = Subset(train_set, indices[val_size:])
            val_data = Subset(val_set, indices[:val_size])
        else:
            if Path(val_db).resolve() == Path(train_db).resolve():
                raise ValueError("显式 val_db 不应与 train_db 相同；请省略 val_db 自动划分")
            train_data, val_data = train_set, val_set

        sample = train_data[0]
        if "view_features" not in sample:
            raise ValueError(
                "LMDB 没有 view_features。先使用 SindrePy DINOv3Encoder "
                "调用 datasets.build_database(..., image_encoder=dino) 预计算，"
                "训练阶段不将 RGB 图像伪装成 DINO patch tokens"
            )
        if sample["view_features"].ndim != 3 or sample["view_features"].shape[0] != 3:
            raise ValueError("view_features 必须为 [3,T,C]")
        if model is None:
            model = create_model(
                image_dim=sample["view_features"].shape[-1],
                margin_points=sample["margin_line"].shape[0],
                config=config,
            )
        model = model.to(device)
        loss_fn = CrownReconstructionLoss(loss_config).to(device)

        pin_memory = device.type == "cuda"
        def get_loader(data, shuffle):
            return DataLoader(
                data, batch_size=config.batch_size, shuffle=shuffle,
                num_workers=config.num_workers,
                pin_memory=pin_memory,
                collate_fn=MyDataset.collate,
                persistent_workers=config.num_workers > 0,
            )

        train_loader = get_loader(train_data, True)
        val_loader = get_loader(val_data, False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
        )
        outdir = Path(config.checkpoint_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        best_path, last_path = outdir / "best.pt", outdir / "last.pt"
        start_epoch = 0
        best_loss = float("inf")
        if config.resume:
            start_epoch, _, extra = load_checkpoint(
                path=str(last_path), net=model, optimizer=optimizer,
                strict=True, map_location=str(device),
            )
            best_loss = float(extra.get("best_val", float("inf")))
            if "scheduler" in extra:
                scheduler.load_state_dict(extra["scheduler"])

        amp = (
            config.use_amp and device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        metrics = {}
        for epoch in range(start_epoch, config.epochs):
            for training, loader in ((True, train_loader), (False, val_loader)):
                model.train(training)
                running = {"loss": 0.0, "occupancy": 0.0, "sdf": 0.0, "iou": 0.0}
                observed = 0
                if training:
                    optimizer.zero_grad(set_to_none=True)
                stage = "train" if training else "val"
                bar = tqdm(
                    loader,
                    desc=f"Epoch {epoch + 1:03d}/{config.epochs:03d} {stage}",
                    dynamic_ncols=True, leave=True,
                )
                for step, cpu_batch in enumerate(bar):
                    batch = {
                        key: value.to(device, non_blocking=pin_memory)
                        if torch.is_tensor(value) else value
                        for key, value in cpu_batch.items()
                    }
                    with torch.set_grad_enabled(training):
                        with torch.autocast(
                            device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp,
                        ):
                            prediction = model(batch)
                            terms = loss_fn(prediction, batch)
                            loss = terms["loss"]
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(
                                f"epoch={epoch + 1}, {stage}, step={step} Loss=NaN/Inf"
                            )
                        if training:
                            # 最后一组可能不足 grad_accum_steps，避免梯度被多除。
                            group_start = (step // config.grad_accum_steps) * config.grad_accum_steps
                            actual_steps = min(
                                config.grad_accum_steps, len(loader) - group_start
                            )
                            (loss / actual_steps).backward()
                            if ((step + 1) % config.grad_accum_steps == 0
                                    or step + 1 == len(loader)):
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), config.grad_clip,
                                )
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                    size = batch["jaw"].shape[0]
                    observed += size
                    for key in running:
                        running[key] += terms[key].detach().float().item() * size
                    bar.set_postfix(
                        loss=f"{running['loss'] / observed:.4f}",
                        occ=f"{running['occupancy'] / observed:.4f}",
                        sdf=f"{running['sdf'] / observed:.4f}",
                        IoU=f"{running['iou'] / observed:.3f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                if observed == 0:
                    raise ValueError(f"{stage} DataLoader 不应为空")
                values = {f"{stage}_{key}": value / observed for key, value in running.items()}
                metrics.update(values)
            scheduler.step()
            val_loss = metrics["val_loss"]
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    save_path=str(best_path), network=model, optimizer=optimizer,
                    loss=val_loss, curr_iter=epoch + 1,
                    extra_info={"config": asdict(config), "best_val": best_loss},
                    save_best_only=False,
                )
            save_checkpoint(
                save_path=str(last_path), network=model, optimizer=optimizer,
                loss=val_loss, curr_iter=epoch + 1,
                extra_info={
                    "config": asdict(config), "best_val": best_loss,
                    "scheduler": scheduler.state_dict(),
                },
                save_best_only=False,
            )
            log.info(
                "epoch %s train=%.5f val=%.5f val_IoU=%.4f best=%.5f",
                epoch + 1, metrics["train_loss"], val_loss,
                metrics["val_iou"], best_loss,
            )
        return {
            **metrics, "best_val": best_loss,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        }
    finally:
        train_set.close()
        val_set.close()


def main(
    train_db: str | Path,
    val_db: str | Path | None = None,
    *,
    epochs: int = 100,
    batch_size: int = 2,
    num_workers: int = 0,
    resume: bool = False,
):
    """Windows / WSL2 显式参数入口，不采用 argparse。"""
    config = TrainingConfig(
        epochs=epochs, batch_size=batch_size,
        num_workers=num_workers, resume=resume,
    )
    return train(train_db, val_db=val_db, config=config)


if __name__ == "__main__":
    raise SystemExit(
        "请直接调用 train.main('data/toothforge_occ_v1.db', epochs=100)"
    )
