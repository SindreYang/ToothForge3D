"""小型条件 FDG 网络训练：不渲染视图，不依赖真实冠坐标做推理。

预处理：utils.CrownFDG.build_database() 缓存开放冠 FDG + DINOv3。
训练：训练 coarse 结构、fine 稀疏坐标、FDG 几何；间隔性计算可微 Margin。
Windows/WSL2：直接从 main(...) 传参数，不使用 argparse。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from datasets import FDGDataset
from fdg_model import LightFDGConfig, LightFDGModel, create_light_fdg_model
from losses import CrownFDGLoss, FDGLossConfig
from utils import CrownFDG, FDGConfig
from sindre.ai.utils import set_global_seeds, save_checkpoint, load_checkpoint

log = logging.getLogger("ToothForge3D.train_fdg")


@dataclass(frozen=True)
class FDGTrainingConfig:
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
    seed: int = 1024
    checkpoint_dir: str = "checkpoints/toothforge_fdg"
    device: str = "auto"
    resume: bool = False
    profile_steps: int = 0

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.query_count, self.jaw_points,
               self.grad_accum_steps) <= 0:
            raise ValueError("Epoch/Batch/query/jaw/accum 必须为正数")
        if (self.num_workers < 0 or self.margin_every < 0 or self.profile_steps < 0
                or not 0 < self.val_fraction < 1
                or self.lr <= 0 or self.grad_clip <= 0 or self.weight_decay < 0):
            raise ValueError("训练数值参数无效")


def get_margin_meshes(
    model: LightFDGModel,
    cache: dict,
    batch: dict,
    converter: CrownFDG,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """使用 GT 拓扑只训练边界顶点偏移，不将 GT 坐标输入结构生成分支。

    重建从模型共享的条件 cache 输出完整 FDG GT 位置预测，
    再调用官方 O-Voxel CUDA 可微顶点转换（仅间隔性执行）。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("margin_every>0 需要可用的 O-Voxel CUDA，CPU 请设置 margin_every=0")
    result = []
    for index, gt_coords_cpu in enumerate(batch["fdg_coords"]):
        coords = gt_coords_cpu.to(cache["volume"].device, non_blocking=True)
        view = {
            "volume": cache["volume"][index:index+1],
            "global": cache["global"][index:index+1],
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


def train_fdg(
    train_db: str | Path, *, val_db: str | Path | None = None,
    config: FDGTrainingConfig | None = None,
    model_config: LightFDGConfig | None = None,
    loss_config: FDGLossConfig | None = None,
    model: LightFDGModel | None = None,
) -> dict:
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

    def make_dataset(path, augment):
        return FDGDataset(
            path, resolution=model_config.resolution,
            coarse_resolution=model_config.coarse_resolution,
            query_count=config.query_count, jaw_points=config.jaw_points,
            augment=augment, seed=config.seed,
        )

    train_set = make_dataset(train_db, True)
    val_set = make_dataset(val_db if val_db is not None else train_db, False)
    try:
        if not len(train_set) or not len(val_set):
            raise ValueError("FDG 数据库不能为空")
        if val_db is None:
            if len(train_set) < 2:
                raise ValueError("需要至少两份数据才能划分 train/validation")
            generator = torch.Generator().manual_seed(config.seed)
            ids = torch.randperm(len(train_set), generator=generator).tolist()
            val_count = max(1, min(len(ids) - 1, round(len(ids) * config.val_fraction)))
            train_data = Subset(train_set, ids[val_count:])
            val_data = Subset(val_set, ids[:val_count])
        else:
            if Path(train_db).resolve() == Path(val_db).resolve():
                raise ValueError("显式 val_db 必须与 train_db 不同")
            train_data, val_data = train_set, val_set
        first = train_data[0]
        if model is None:
            model = create_light_fdg_model(
                image_dim=first["view_features"].shape[-1],
                margin_points=first["margin_line"].shape[0],
                config=model_config,
            )
        model = model.to(device)
        criterion = CrownFDGLoss(loss_config).to(device)
        converter = CrownFDG(FDGConfig(resolution=model_config.resolution))

        def loader(dataset, training):
            return DataLoader(
                dataset, batch_size=config.batch_size, shuffle=training,
                num_workers=config.num_workers, pin_memory=device.type == "cuda",
                collate_fn=FDGDataset.collate,
                persistent_workers=config.num_workers > 0,
            )

        train_loader = loader(train_data, True)
        val_loader = loader(val_data, False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
        )
        output = Path(config.checkpoint_dir)
        output.mkdir(parents=True, exist_ok=True)
        start_epoch = 0
        best = float("inf")
        if config.resume:
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
                    count = 0
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    stage = "train" if training else "val"
                    bar = tqdm(
                        data_loader,
                        desc=f"FDG epoch {epoch + 1}/{config.epochs} {stage}",
                        dynamic_ncols=True,
                    )
                    for step, cpu_batch in enumerate(bar):
                        batch = {
                            key: [tensor.to(device, non_blocking=True)
                                  for tensor in value]
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
                        for key in sums:
                            sums[key] += float(terms[key].detach()) * n
                        bar.set_postfix(
                            loss=f"{sums['loss'] / count:.4f}",
                            coarse=f"{sums['coarse'] / count:.3f}",
                            fine=f"{sums['fine'] / count:.3f}",
                            offset=f"{sums['offset'] / count:.3f}",
                            recall=f"{sums['coarse_recall'] / count:.2f}",
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
        val_set.close()


@torch.inference_mode()
def predict_mesh(
    model: LightFDGModel, condition: dict[str, torch.Tensor],
    *, refine_margin: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    """推理只需真实病例条件：没有任何 fdg_coords/target_vertices 输入。

    无合法稀疏体素/未形成面片时返回 None，不能生成伪 Mesh。
    """
    model.eval()
    converter = CrownFDG(FDGConfig(resolution=model.config.resolution))
    predicted = model.predict(condition)
    results = []
    for i, row in enumerate(predicted):
        if not row["valid"]:
            results.append(None)
            continue
        center, scale = converter.get_condition_frame(
            condition["margin_line"][i].detach().cpu().numpy(),
            condition["size"][i].detach().cpu().numpy(),
        )
        fdg = {
            "coords":row["coords"],
            "resolution":model.config.resolution,
            "scale_mm":scale, "center_mm":center,
        }
        try:
            vertices, faces = converter.get_mesh(
                fdg, dual_offsets=row["dual_offsets"],
                intersected_logits=row["intersected_logits"],
            )
            if not len(faces):
                results.append(None)
                continue
            if refine_margin:
                vertices = converter.refine_margin(
                    vertices, faces, condition["margin_line"][i],
                )
            results.append((vertices, faces))
        except (ValueError, RuntimeError) as exc:
            log.warning("预测牙冠 %s FDG 解码失败: %s", i, exc)
            results.append(None)
    return results


def main(
    train_db: str | Path,
    val_db: str | Path | None = None,
    *, epochs: int = 100, batch_size: int = 2,
    margin_every: int = 16, num_workers: int = 2,
):
    return train_fdg(
        train_db, val_db=val_db,
        config=FDGTrainingConfig(
            epochs=epochs, batch_size=batch_size,
            num_workers=num_workers, margin_every=margin_every,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(
        "请在 Python 中调用 train_fdg.main('data/toothforge_fdg_v1.db')"
    )
