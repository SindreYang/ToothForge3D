"""主机病例级 FDG 缓存；仅保存正式预处理结果，不缓存失败病例。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
from importlib import metadata
from typing import Any

import numpy as np


class FDGCaseCache:
    """按真实输入、预处理配置、源码和依赖指纹缓存病例，不使用 pickle。

    缓存保存已计算的 FDG 目标、所有有效扫描条件和 DINOv3 特征。
    每轮独立几何验收仍然执行，缓存命中不代表临床验收。
    """

    # 已知历史回归的预处理代码与首次缓存版本一致；仅用于一次性恢复。
    LEGACY_RUN = "46134417550-1"
    LEGACY_COMPATIBLE = {
    "process/process.py": "bddd305d979989f135eab654013a12a518366203",
    "src/views.py": "5a70a92cd49010d83918f6dfb46dc88aa332370f",
    "src/utils.py": "70e6dc1613ee05f958c5fd98343240b805ca21e3",
    "src/losses.py": "41fb070d32607ad8303a9c40de286e1ab5a7ab58"
}
    LEGACY_SINDRE = "5c872863b8ae96464a8924356cac12337db1445c"

    def __init__(self, directory: str | Path, encoder, options: dict):
        from sindre.ai.vision2d.dinov3_utils import DINOv3Encoder
        from sindre.general import get_default_models_path
        import sindre
        import torch

        if type(encoder) is not DINOv3Encoder:
            raise ValueError("缓存要求正式 DINOv3Encoder，避免无法识别的权重")
        self.root = Path(directory).expanduser().resolve()
        self.options = options
        self.code_root = Path(__file__).resolve().parents[1]
        # 真实权重和处理器文件按内容哈希；不加载 GPU 模型也能命中缓存。
        model = Path(encoder.model_name).expanduser()
        if not model.is_dir():
            model = Path(get_default_models_path("DINOv3")) / encoder.model_name.replace("/", "__")
        self.model_files = sorted(p for p in model.rglob("*") if p.is_file() and
                                  p.suffix in (".safetensors", ".bin", ".json") and
                                  not p.name.startswith("."))
        if not any(p.suffix in (".safetensors", ".bin") for p in self.model_files):
            # 第一次运行需要先获得正式权重，不能使用仅名称作为缓存指纹。
            encoder._ensure_loaded()
            model = Path(encoder._model_path)
            self.model_files = sorted(p for p in model.rglob("*") if p.is_file()
                                      and p.suffix in (".safetensors", ".bin", ".json")
                                      and not p.name.startswith("."))
        if not any(p.suffix in (".safetensors", ".bin") for p in self.model_files):
            raise ValueError("无法确认 DINOv3 实际权重文件，拒绝启用缓存")
        self.code_hashes = {}
        for path in (*self.LEGACY_COMPATIBLE, "process/cache.py", ".github/scripts/build_cuda_wheels.sh"):
            payload = (self.code_root / path).read_bytes()
            self.code_hashes[path] = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest()
        self.sindre_sha = os.environ.get("SINDRE_SHA", "")
        marker = Path(sindre.__file__).resolve().parent.parent / ".source-sha"
        if not self.sindre_sha and marker.is_file():
            self.sindre_sha = marker.read_text().strip()
        # 非 CI 运行也检查实际使用的 SindrePy 代码，不只信任版本字符串。
        sindre_root = Path(sindre.__file__).resolve().parent
        dependency_paths = [sindre_root / name for name in (
            "utils3d/core/algorithm.py", "utils3d/core/dental/arch/alignment.py",
            "utils3d/core/dental/utils.py", "utils3d/core/sample.py",
            "utils3d/sindremesh.py", "utils3d/sindrepcd.py",
            "ai/vision2d/dinov3_utils/infer.py",
        )]
        dependencies = {str(p.relative_to(sindre_root)): self.file_digest(p)
                        for p in dependency_paths if p.is_file()}
        versions = {}
        for name in ("numpy", "scipy", "torch", "transformers", "meshlib",
                     "vedo", "vtk", "o-voxel", "cumesh", "flex_gemm"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        config = {
            "format": 1, "options": options, "code": self.code_hashes,
            "sindre_commit": self.sindre_sha, "sindre_files": dependencies,
            "versions": versions, "torch_cuda": torch.version.cuda,
            "encoder": {"name": encoder.model_name, "device": encoder.device_name,
                        "source": encoder.source,
                        "files": {str(p.relative_to(model)): self.file_digest(p) for p in self.model_files}},
        }
        payload = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        self.signature = hashlib.sha256(payload).hexdigest()
        self.directory = self.root / self.signature
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.atomic_write(self.directory / "config.json", lambda stream: stream.write(payload))

    @staticmethod
    def file_digest(path: Path) -> str:
        """流式哈希大文件，避免模型权重读入内存。"""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def get_key(raw: dict, index: int) -> str:
        """按病例全部真实字段的内容哈希；索引参与确定增强随机种子。"""
        digest = hashlib.sha256()

        def update(value: Any):
            if isinstance(value, np.ndarray):
                digest.update(b"array")
                update(value.dtype.str)
                update(tuple(value.shape))
                if value.dtype.hasobject:
                    for item in value.flat:
                        update(item)
                else:
                    digest.update(np.ascontiguousarray(value).tobytes())
            elif isinstance(value, np.generic):
                update(np.asarray(value))
            elif isinstance(value, dict):
                digest.update(b"dict")
                update(len(value))
                for key in sorted(value):
                    update(key)
                    update(value[key])
            elif isinstance(value, (tuple, list)):
                digest.update(b"sequence")
                update(len(value))
                for item in value:
                    update(item)
            elif isinstance(value, bytes):
                digest.update(b"bytes")
                update(len(value))
                digest.update(value)
            elif value is None or isinstance(value, (str, int, float, bool)):
                encoded = repr((type(value).__name__, value)).encode("utf-8")
                digest.update(str(len(encoded)).encode() + b":" + encoded)
            else:
                raise ValueError(f"病例缓存不支持字段类型: {type(value).__name__}")

        update(index)
        update(raw)
        return digest.hexdigest()

    @staticmethod
    def atomic_write(path: Path, write):
        """同目录临时文件写入并原子替换；中断不会产生半成品缓存。"""
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as stream:
                write(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    def load(self, key: str):
        """读取缓存；损坏或缺少完整原始目标时重新计算。"""
        path = self.directory / f"{key}.npz"
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                count = int(archive["__count"])
                if not 1 <= count <= self.options["augment_variants"]:
                    raise ValueError("缓存目标数量异常")
                records = [{} for _ in range(count)]
                bank = {}
                for name in archive.files:
                    if name.startswith("r_"):
                        _, index, field = name.split("_", 2)
                        records[int(index)][field] = archive[name]
                    elif name.startswith("b_"):
                        bank[name[2:]] = archive[name].tobytes()
                if int(records[0]["variant_index"]) != 0:
                    raise ValueError("缓存缺少原始冠")
                for record in records:
                    if any(k not in record for k in ("fdg_coords", "fdg_dual_offsets", "fdg_intersected",
                                                     "target_vertices", "target_faces", "view_features",
                                                     "margin_line", "scan_keys")):
                        raise ValueError("缓存字段不完整")
                    if any(str(k) not in bank for k in np.asarray(record["scan_keys"]).reshape(-1)):
                        raise ValueError("扫描条件缓存缺失")
                return records, bank
        except (OSError, ValueError, TypeError, KeyError, IndexError, zipfile.BadZipFile):
            return None

    def save(self, key: str, records: list[dict], bank: dict[str, bytes]):
        """保存一个完整病例及扫描条件，不保存部分结果或失败病例。"""
        if not records or int(records[0]["variant_index"]) != 0:
            raise ValueError("缓存必须包含原始目标冠")
        arrays = {"__count": np.array(len(records), dtype=np.int32)}
        for index, record in enumerate(records):
            for name, value in record.items():
                array = np.asarray(value)
                if array.dtype.hasobject:
                    raise ValueError("缓存禁止对象数组")
                arrays[f"r_{index}_{name}"] = array
        for name, payload in bank.items():
            arrays[f"b_{name}"] = np.frombuffer(payload, dtype=np.uint8)
        self.atomic_write(self.directory / f"{key}.npz", lambda stream: np.savez(stream, **arrays))

    def import_verified_legacy(self, input_path: Path, source_reader) -> int:
        """恢复已完成的同版历史回归；无法验证时保留历史，不自动混用。

        旧版没有逐病例内容哈希，故仅允许已知工作流、原始库/权重文件
        未在当轮开始后修改、配置相同、全部已写记录通过几何检查时恢复。
        新缓存始终使用真实输入内容哈希。
        """
        from sindre.lmdb import Reader

        marker = self.directory / "legacy_import.json"
        if marker.exists():
            return 0
        if (self.sindre_sha != self.LEGACY_SINDRE
                or any(self.code_hashes.get(p) != sha for p, sha in self.LEGACY_COMPATIBLE.items())
                or input_path.name != "全牙_5.db"):
            return 0
        expected = {"fdg": self.options["fdg"], "num_jaw_points": 4096,
                    "image_size": 256, "augment_variants": 4, "augment_scale": .10,
                    "collision_check": True, "max_collision_attempts": 12,
                    "seed": 1024, "scan_augmentation": True}
        if self.options != expected or self.options["fdg"]["resolution"] != 128:
            return 0
        run = self.root.parent / "runs" / self.LEGACY_RUN
        if not run.is_dir():
            run = Path.home() / ".cache" / "toothforge3d-regression" / self.LEGACY_RUN
        try:
            stamp = (run / "unit.log").stat().st_mtime_ns
            source_file = input_path / "data.mdb" if input_path.is_dir() else input_path
            if source_file.stat().st_mtime_ns > stamp or any(p.stat().st_mtime_ns > stamp for p in self.model_files):
                return 0
            # 历史 CI 固定从当前项目的 datasets/全牙_5.db 读取。
            project = self.root.parents[2]
            if input_path.resolve() != (project / "datasets" / "全牙_5.db").resolve():
                return 0
            result = json.loads((run / "aggregate.json").read_text(encoding="utf-8"))
            report = json.loads((run / "fdg_v3_full.report.json").read_text(encoding="utf-8"))
            checked = result["preprocess"]
            if (checked.get("status") != "passed" or not checked.get("all_records_checked")
                    or checked.get("roundtrip_records_pass") != checked.get("written")
                    or report["total"] != len(source_reader)):
                return 0
            errors = {int(item["index"]) for item in report["errors"]}
            indices = [index for index in range(report["total"]) if index not in errors]
            if len(indices) != checked["original_records"]:
                return 0
            imported = 0
            with Reader(str(run / "fdg_v3_full.db")) as old:
                if (old.get_meta("build_complete") is not True
                        or old.get_meta("fdg_config") != self.options["fdg"]
                        or old.get_meta("schema_version") != "toothforge3d_fdg_v3"
                        or len(old) != checked["written"]):
                    return 0
                cursor = 0
                for index in indices:
                    records = [old[cursor]]
                    cursor += 1
                    if int(records[0]["variant_index"]) != 0:
                        raise ValueError("旧病例记录边界错误")
                    while cursor < len(old):
                        item = old[cursor]
                        if int(item["variant_index"]) == 0:
                            break
                        records.append(item)
                        cursor += 1
                    raw = source_reader[index]
                    design = raw.get("design_mesh", {})
                    if int(records[0]["source_fdi"]) != int(raw.get("target_fdi", design.get("fdi"))):
                        raise ValueError("旧病例牙位不一致")
                    keys = [str(k) for k in np.asarray(records[0]["scan_keys"]).reshape(-1)]
                    # scan_key 包含原始病例行号，避免按订单分组误关联多牙。
                    if not keys or any(not k.startswith(f"scan_{index}_") for k in keys):
                        continue
                    bank = {key: old.get_meta(key) for key in keys}
                    if any(not isinstance(value, bytes) for value in bank.values()):
                        raise ValueError("旧扫描条件丢失")
                    key = self.get_key(raw, index)
                    if self.load(key) is None:
                        self.save(key, records, bank)
                        imported += 1
                if cursor != len(old):
                    raise ValueError("旧病例记录数不一致")
            self.atomic_write(marker, lambda f: f.write(json.dumps(
                {"run": self.LEGACY_RUN, "imported": imported}, ensure_ascii=False).encode()))
            return imported
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return 0
