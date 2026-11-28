"""WSL2 本地真实病例回归；不上传网格、病例标识、日志和模型权重。"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sindre.lmdb import Reader
from sindre.utils3d.sindremesh import SindreMesh
from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
from process import DentalCase, FDGPreprocessor
from src.utils import CrownFDG, FDGConfig
from src.losses import CrownFDGLoss
from src.train import FDGTrainingConfig, train
from src.model import LightFDGConfig
from src.datasets import FDGDataset


class RealCrownRegression:
    """独立验证实际病例、几何转换和单轮训练；仅公开聚合诊断。"""

    def __init__(self, database: Path, output: Path, max_cases: int = 5):
        self.database = database
        self.output = output
        self.max_cases = max_cases
        self.converter = CrownFDG(FDGConfig(resolution=128))
        self.loss = CrownFDGLoss()
        self.summary = {"source_cases": 0, "tested": 0, "roundtrip_pass": 0,
                        "case_results": [], "preprocess": {}, "training": {},
                        "limits": {"margin_max_mm": 0.5, "surface_p95_mm": 0.5},
                        "clinical_validation": False}

    def save_summary(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def get_error(self, stage: str, exc: Exception):
        """详细异常仅进入本地日志；公开结果只保留类型、函数、行号。"""
        traceback.print_exc()
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return {"stage": stage, "error_type": type(exc).__name__,
                "function": last.name, "line": last.lineno}

    def check_roundtrip(self):
        with Reader(str(self.database)) as reader:
            self.summary["source_cases"] = len(reader)
            for index in range(min(len(reader), self.max_cases)):
                started = time.perf_counter()
                result = {"case_index": index, "passed": False}
                stage = "case_frame"
                try:
                    raw = dict(reader[index])
                    # 验证无外部坐标的真实路径；不允许旧设计冠方向或视图缓存。
                    for key in ("local_frame", "view_features", "view_mask", "view_scale_mm", "frame_source"):
                        raw.pop(key, None)
                    case = DentalCase(raw)
                    if case.margin_line is None or case.frame_source != "scan_arch_curve":
                        raise ValueError("没有可靠的扫描坐标或颈缘线")
                    sample = case.get_condition(crop_mode="full", tooth_dropout=0, num_points=2048)
                    result["frame_confidence"] = round(float(case.frame_confidence), 4)
                    # 同一真实颈缘和 FDI（作为模拟用户输入）必须产生一致的扫描坐标。
                    stage = "train_infer_frame"
                    inference = dict(raw)
                    inference.pop("design_mesh", None)
                    inference.pop("size_mm", None)
                    inference["target_fdi"] = case.target_fdi
                    inference["margin_line"] = case.margin_line
                    check = DentalCase(inference, inference=True)
                    if not np.allclose(case.world_to_local_transform, check.world_to_local_transform, atol=1e-4):
                        raise AssertionError("训练与推理坐标不一致")
                    stage = "open_crown"
                    vertices, faces = self.converter.get_open_crown(
                        sample["target_vertices"], sample["target_faces"], sample["margin_line"])
                    self.converter.get_mesh_quality(vertices, faces)
                    stage = "fdg_encode"
                    fdg = self.converter.get_fdg(vertices, faces, sample["margin_line"], cut_closed=False)
                    result["fdg_cells"] = len(fdg["coords"])
                    result["offset_min"] = round(float(fdg["dual_offsets"].min()), 5)
                    result["offset_max"] = round(float(fdg["dual_offsets"].max()), 5)
                    stage = "fdg_decode_gradient"
                    offsets = torch.tensor(fdg["dual_offsets"], device="cuda", requires_grad=True)
                    dv, df = self.converter.get_mesh(fdg, dual_offsets=offsets)
                    if not df.numel():
                        raise ValueError("空解码冠")
                    dv.square().mean().backward()
                    result["gradient_finite_nonzero"] = bool(
                        offsets.grad is not None and torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0)
                    if not result["gradient_finite_nonzero"]:
                        raise AssertionError("FDG 解码顶点无有效反向梯度")
                    out_v, out_f = dv.detach().cpu().numpy(), df.detach().cpu().numpy()
                    stage = "surface_quality"
                    original = SindreMesh([vertices, faces])
                    decoded = SindreMesh([out_v, out_f])
                    original.save(str(self.output / f"case_{index:02d}_reference.ply"))
                    decoded.save(str(self.output / f"case_{index:02d}_decoded.ply"))
                    # 真实三角面最近距离；限量采样不冒充精确 Hausdorff。
                    distances = []
                    for target, source in ((decoded, vertices), (original, out_v)):
                        ids = np.linspace(0, len(source) - 1, min(1024, len(source)), dtype=int)
                        _, distance = target.get_closest_point(source[ids], return_distance=True)
                        distances.extend(np.asarray(distance).tolist())
                    result["surface_p95_mm"] = round(float(np.quantile(distances, 0.95)), 5)
                    margin = self.loss.get_margin_loss(
                        dv.detach(), df, torch.as_tensor(sample["margin_line"], device=dv.device))
                    result["margin_max_mm"] = round(float(margin["max_distance_mm"]), 5)
                    result["margin_mean_mm"] = round(float(margin["mean_distance_mm"]), 5)
                    # 局部截图保存供本地阅图，不上传真实牙颌。
                    stage = "render"
                    self.converter.show(out_v, out_f, sample["margin_line"],
                                        screenshot=self.output / f"case_{index:02d}_decoded.png")
                    stage = "topology"
                    self.converter.get_mesh_quality(out_v, out_f)
                    stage = "tolerance"
                    result["passed"] = bool(result["surface_p95_mm"] <= self.summary["limits"]["surface_p95_mm"]
                                            and result["margin_max_mm"] <= self.summary["limits"]["margin_max_mm"])
                    if result["passed"]:
                        self.summary["roundtrip_pass"] += 1
                except Exception as exc:
                    result.update(self.get_error(stage, exc))
                finally:
                    result["seconds"] = round(time.perf_counter() - started, 3)
                    self.summary["tested"] += 1
                    self.summary["case_results"].append(result)
                    self.save_summary()

    def check_preprocess_and_train(self):
        target = self.output / "fdg_v3_smoke.db"
        stage = "preprocess"
        started = time.perf_counter()
        try:
            with DINOv3Encoder() as encoder:
                report = FDGPreprocessor(self.converter).build_database(
                    self.database, target, max_cases=self.max_cases, map_size_mb=4096,
                    image_encoder=encoder, num_jaw_points=2048, augment_variants=1,
                    collision_check=True, strict=False,
                )
            self.summary["preprocess"] = {
                "attempted": int(report["total"]), "written": int(report["written"]),
                "errors": len(report["errors"]), "seconds": round(time.perf_counter()-started, 3),
                "dimension_augmentation_tested": False}
            self.save_summary()
            stage = "dataset"
            dataset = FDGDataset(target, query_count=2048, jaw_points=2048, augment=False)
            try:
                groups = len(set(dataset.get_source_groups()))
                sample = dataset[0]
                self.summary["preprocess"]["groups"] = groups
                self.summary["preprocess"]["dataset_read"] = True
                if not torch.isfinite(sample["view_features"]).all():
                    raise AssertionError("视图条件包含非有限值")
            finally:
                dataset.close()
            if groups < 2:
                self.summary["training"] = {"status": "blocked", "reason": "fewer_than_two_source_groups"}
                return
            stage = "train"
            torch.cuda.reset_peak_memory_stats()
            train_started = time.perf_counter()
            metrics = train(
                target, config=FDGTrainingConfig(
                    epochs=1, batch_size=1, query_count=2048, jaw_points=2048, num_workers=0,
                    margin_every=1, generation_val_cases=1, device="cuda",
                    checkpoint_dir=str(self.output / "checkpoints")),
                model_config=LightFDGConfig(resolution=128),
            )
            self.summary["training"] = {
                "status": "passed", "epochs": 1,
                "seconds": round(time.perf_counter()-train_started, 3),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            for key in ("train_loss", "val_loss", "val_generation_count",
                        "val_generation_valid_rate", "val_generation_fdg_iou", "val_generation_margin_max_mm"):
                value = metrics.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    self.summary["training"][key] = float(value)
            if not all(np.isfinite(metrics.get(k, float("nan"))) for k in ("train_loss", "val_loss")):
                raise FloatingPointError("训练或验证损失无效")
        except Exception as exc:
            key = "training" if stage == "train" else "preprocess"
            self.summary[key].update(self.get_error(stage, exc))
            self.summary[key]["status"] = "failed"
        finally:
            self.save_summary()

    def run(self) -> bool:
        try:
            self.check_roundtrip()
        except Exception as exc:
            self.summary["fatal"] = self.get_error("read_source", exc)
            self.save_summary()
        self.check_preprocess_and_train()
        passed = (self.summary["tested"] > 0
                  and self.summary["roundtrip_pass"] == self.summary["tested"]
                  and self.summary["preprocess"].get("written") == self.summary["tested"]
                  and self.summary["training"].get("status") == "passed")
        self.summary["passed"] = bool(passed)
        self.save_summary()
        return bool(passed)


def main():
    regression = RealCrownRegression(Path(os.environ["TF_DB"]), Path(os.environ["TF_RESULTS"]))
    raise SystemExit(0 if regression.run() else 1)


if __name__ == "__main__":
    main()
