"""WSL2 本地真实病例回归；不上传网格、病例标识及详细日志。"""

from __future__ import annotations

import json
import io
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
from src.datasets import FDGDataset


class RealCrownRegression:
    """独立验证实际病例、几何转换和数据契约；仅公开聚合诊断。"""

    def __init__(self, database: Path, output: Path, max_cases: int = 140):
        self.database = database
        self.output = output
        self.max_cases = max_cases
        self.converter = CrownFDG(FDGConfig(resolution=128))
        self.summary = {"source_cases": 0, "tested": 0, "roundtrip_pass": 0,
                        "case_results": [], "preprocess": {}, "stage": "data",
                        "limits": {"margin_to_surface_max_mm": 0.5, "surface_p95_mm": 0.5, "surface_sample_max_mm": 0.5},
                        "raw_topology_is_delivery_gate": False,
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
                    if case.margin_line is None or case.frame_source not in ("scan_arch_curve", "scan_fdi_anchors"):
                        raise ValueError("没有可靠的扫描坐标或颈缘线")
                    sample = case.get_condition(crop_mode="full", tooth_dropout=0, num_points=2048)
                    result["frame_confidence"] = round(float(case.frame_confidence), 4)
                    result["frame_source"] = case.frame_source
                    # 同一真实颈缘和 FDI（作为模拟用户输入）必须产生一致的扫描坐标。
                    stage = "train_infer_frame"
                    inference = dict(raw)
                    inference.pop("design_mesh", None)
                    inference.pop("size_mm", None)
                    inference["target_fdi"] = case.target_fdi
                    inference["margin_line"] = case.margin_source_points
                    check = DentalCase(inference, inference=True)
                    result["frame_max_abs_diff"] = float(np.abs(case.world_to_local_transform - check.world_to_local_transform).max())
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
                    stage = "fdg_decode"
                    with torch.no_grad():
                        dv, df = self.converter.get_mesh(fdg)
                    if not df.numel():
                        raise ValueError("空解码冠")
                    out_v, out_f = dv.detach().cpu().numpy(), df.detach().cpu().numpy()
                    stage = "surface_quality"
                    result.update(self.converter.get_roundtrip_geometry(
                        vertices, faces, sample["margin_line"], dv, df))
                    stage = "render"
                    SindreMesh([vertices, faces]).save(str(self.output / f"case_{index:03d}_reference.ply"))
                    SindreMesh([out_v, out_f]).save(str(self.output / f"case_{index:03d}_decoded.ply"))
                    self.converter.show(vertices, faces, sample["margin_line"],
                                        screenshot=self.output / f"case_{index:03d}_reference.png")
                    self.converter.show(out_v, out_f, sample["margin_line"],
                                        screenshot=self.output / f"case_{index:03d}_decoded.png")
                    stage = "tolerance"
                    result["passed"] = all(result[key] <= limit for key, limit in self.summary["limits"].items())
                    if result["passed"]:
                        self.summary["roundtrip_pass"] += 1
                except Exception as exc:
                    result.update(self.get_error(stage, exc))
                finally:
                    result["seconds"] = round(time.perf_counter() - started, 3)
                    self.summary["tested"] += 1
                    self.summary["case_results"].append(result)
                    self.save_summary()

    def check_preprocess(self):
        target = self.output / "fdg_v3_full.db"
        stage = "preprocess"
        started = time.perf_counter()
        try:
            with DINOv3Encoder() as encoder:
                report = FDGPreprocessor(self.converter).build_database(
                    self.database, target, max_cases=self.max_cases, map_size_mb=16384,
                    image_encoder=encoder, num_jaw_points=4096, augment_variants=4,
                    collision_check=True, strict=False, scan_augmentation=True,
                )
            self.summary["preprocess"] = {
                "attempted": int(report["total"]), "written": int(report["written"]),
                "errors": len(report["errors"]), "seconds": round(time.perf_counter()-started, 3),
                "skipped_augmentation": int(report["skipped_augmentation"]),
                "rejected_collision": int(report["rejected_collision"]),
                "dimension_augmentation_tested": True,
                "scan_variants": report["scan_variants"],
                "scan_variant_skipped": report["scan_variant_skipped"],
                "size_environment_unavailable": report.get("size_environment_unavailable", 0),
                "size_fdg_rejected": report.get("size_fdg_rejected", 0),
                "error_stages": {stage: sum(e.get("stage") == stage for e in report["errors"])
                                 for stage in sorted({e.get("stage", "unknown") for e in report["errors"]})}}
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
            stage = "record_contract"
            originals = set()
            base = None
            base_key = None
            variants = 0
            roundtrip_records = 0
            checked_scan_keys = set()
            topology_records = 0
            max_p95 = 0.0
            with Reader(str(target)) as reader:
                for i in range(len(reader)):
                    record = reader[i]
                    v = np.asarray(record["target_vertices"])
                    f = np.asarray(record["target_faces"])
                    margin = np.asarray(record["margin_line"])
                    size = np.array([np.ptp(v[:, 0]), np.ptp(v[:, 1]),
                                     v[:, 2].max() - margin[:, 2].mean()])
                    if not np.allclose(size, record["size"], atol=1e-4):
                        raise AssertionError("实际尺寸与监督不一致")
                    if not np.isfinite(record["view_features"]).all():
                        raise AssertionError("非有限 DINOv3 特征")
                    self.converter.get_mesh_quality(v, f)
                    group = int(record["source_group"])
                    # One original per source row, not necessarily one tooth per order.
                    key = (group, int(record["source_fdi"]), margin.tobytes())
                    if int(record["variant_index"]) == 0:
                        originals.add(key)
                        base, base_key = record, key
                    else:
                        variants += 1
                        if base is None or key != base_key:
                            raise AssertionError("增强样本不属于当前原始病例")
                        ids = self.converter.get_boundary_indices(torch.as_tensor(f), len(v)).numpy()
                        if not np.array_equal(v[ids], np.asarray(base["target_vertices"])[ids]):
                            raise AssertionError("增强导致真实颈缘边界移动")
                        if not np.array_equal(record["view_features"], base["view_features"]):
                            raise AssertionError("配对增强图像条件发生变化")
                    for scan_key in np.asarray(record.get("scan_keys", [])).reshape(-1):
                        scan_key = str(scan_key)
                        if scan_key in checked_scan_keys:
                            continue
                        payload = reader.get_meta(scan_key)
                        if not isinstance(payload, bytes):
                            raise AssertionError("扫描增强引用丢失")
                        with np.load(io.BytesIO(payload), allow_pickle=False) as condition:
                            if not np.isfinite(condition["view_features"]).all() or not condition["view_mask"].all():
                                raise AssertionError("离线三视图无效")
                            if len(condition["jaw_points"]) != len(condition["jaw_labels"]):
                                raise AssertionError("点云和标签不一致")
                            if "_preparation_" in scan_key and bool(condition["jaw_present"]):
                                raise AssertionError("仅预备牙包含邻牙")
                        checked_scan_keys.add(scan_key)
                    # 逐条验证实际落盘的 FDG 标签（包含增强样本）。
                    fdg = {k[4:]: value for k, value in record.items() if k.startswith("fdg_")}
                    with torch.no_grad():
                        dv, df = self.converter.get_mesh(fdg)
                    measured = self.converter.get_roundtrip_geometry(v, f, margin, dv, df)
                    geometry_pass = all(measured[k] <= limit for k, limit in self.summary["limits"].items())
                    roundtrip_records += int(geometry_pass)
                    topology_records += int(measured["topology_pass"])
                    max_p95 = max(max_p95, measured["surface_p95_mm"])
                    if not geometry_pass:
                        self.summary["preprocess"].setdefault("failed_record_indices", []).append(i)
            self.summary["preprocess"].update(
                roundtrip_records_pass=roundtrip_records, raw_topology_records_pass=topology_records,
                scan_conditions_checked=len(checked_scan_keys),
                worst_surface_p95_mm=max_p95,
                original_records=len(originals), augmented_records=variants,
                status="passed" if roundtrip_records == report["written"] else "failed", all_records_checked=True,
                output_database=str(target.name))
        except Exception as exc:
            self.summary["preprocess"].update(self.get_error(stage, exc))
            self.summary["preprocess"]["status"] = "failed"
        finally:
            self.save_summary()

    def run(self) -> bool:
        try:
            self.check_roundtrip()
        except Exception as exc:
            self.summary["fatal"] = self.get_error("read_source", exc)
            self.save_summary()
        self.check_preprocess()
        passed = (self.summary["tested"] == self.summary["source_cases"] == self.max_cases
                  and self.summary["roundtrip_pass"] == self.summary["tested"]
                  and self.summary["preprocess"].get("original_records") == self.summary["tested"]
                  and self.summary["preprocess"].get("status") == "passed"
                  and self.summary["preprocess"].get("augmented_records", 0) > 0
                  and self.summary["preprocess"].get("scan_conditions_checked", 0) > 0)
        self.summary["passed"] = bool(passed)
        self.summary["raw_topology_pass"] = sum(bool(c.get("topology_pass")) for c in self.summary["case_results"])
        self.summary["visual_review"] = "pending_local_review"
        self.save_summary()
        # 完整逐病例诊断保留本地，CI 只公开聚合结果。
        public = {k: value for k, value in self.summary.items() if k != "case_results"}
        public["frame_sources"] = {source: sum(c.get("frame_source") == source for c in self.summary["case_results"])
                                   for source in ("scan_arch_curve", "scan_fdi_anchors")}
        public["case_failures"] = [{k: c[k] for k in ("case_index","stage","error_type","function","line") if k in c}
                                   for c in self.summary["case_results"] if not c["passed"]]
        for name in self.summary["limits"]:
            values = [c[name] for c in self.summary["case_results"] if name in c]
            if values:
                public[name] = {"mean": float(np.mean(values)), "max": float(np.max(values))}
        (self.output / "aggregate.json").write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return bool(passed)


def main():
    regression = RealCrownRegression(Path(os.environ["TF_DB"]), Path(os.environ["TF_RESULTS"]))
    raise SystemExit(0 if regression.run() else 1)


if __name__ == "__main__":
    main()
