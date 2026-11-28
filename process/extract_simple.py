"""从原始 SindrePy LMDB 中按目标牙位提取病例，每个牙位最多保留 5 条。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

SAMPLES_PER_TOOTH = 5


def get_target_fdi(record: dict[str, Any], index: int) -> int:
    """读取病例的目标 FDI，缺失或格式错误时给出病例序号。"""
    target_fdi = record.get("target_fdi")
    if target_fdi is None:
        design_mesh = record.get("design_mesh")
        if isinstance(design_mesh, dict):
            target_fdi = design_mesh.get("fdi")
    if target_fdi is None or isinstance(target_fdi, bool):
        raise ValueError(f"病例 {index} 缺少有效的 target_fdi 或 design_mesh.fdi")
    try:
        return int(target_fdi)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"病例 {index} 的目标 FDI 无法解析：{target_fdi!r}"
        ) from exc


def extract_simple(
    input_db: Path,
    output_db: Path | None = None,
) -> dict[int, int]:
    """按目标 FDI 分组，依原始顺序保留每组前 5 条并写入新数据库。"""
    input_db = Path(input_db)
    if not input_db.exists():
        raise FileNotFoundError(f"输入数据库不存在：{input_db}")
    if output_db is None:
        output_db = input_db.with_name(f"{input_db.stem}_simple{input_db.suffix}")
    output_db = Path(output_db)
    if input_db.resolve() == output_db.resolve():
        raise ValueError("输出数据库不能与输入数据库相同")
    if output_db.exists():
        raise FileExistsError(f"禁止覆盖已有数据库：{output_db}")

    from sindre.lmdb import Reader, Writer

    with Reader(str(input_db)) as reader:
        source_count = len(reader)
        tooth_indices: dict[int, list[int]] = {}
        for index in tqdm(range(source_count), desc="扫描病例", unit="例"):
            fdi = get_target_fdi(reader[index], index)
            indices = tooth_indices.setdefault(fdi, [])
            if len(indices) < SAMPLES_PER_TOOTH:
                indices.append(index)

        selected_indices = sorted(
            index for indices in tooth_indices.values() for index in indices
        )
        if not selected_indices:
            raise ValueError(f"数据库中没有可提取的病例：{input_db}")

        tooth_counts = {
            fdi: len(indices) for fdi, indices in sorted(tooth_indices.items())
            if indices
        }
        output_db.parent.mkdir(parents=True, exist_ok=True)
        writer = Writer(str(output_db), map_size_limit=1024*1024*50)
        try:
            schema_version = reader.get_meta("schema_version")
            if schema_version is not None:
                writer.put_meta("schema_version", schema_version)
            writer.put_meta("samples_per_tooth", SAMPLES_PER_TOOTH)
            writer.put_meta("tooth_counts", {
                str(fdi): count for fdi, count in tooth_counts.items()
            })
            writer.put_meta("source_count", source_count)
            writer.put_meta("filtered_count", len(selected_indices))
            for index in tqdm(
                selected_indices, desc="写入病例", unit="例",
                total=len(selected_indices),
            ):
                writer.put_sample(reader[index])
        finally:
            writer.close()

    for fdi, count in tooth_counts.items():
        print(f"FDI {fdi}: {count} 条")
    print(f"共提取 {len(selected_indices)} / {source_count} 条")
    print(f"输出数据库：{output_db}")
    return tooth_counts


if __name__ == "__main__":
    INPUT_DB = Path(
        r"/mnt/c/Users/sindre/Downloads/网格生成_邻牙已分割_冠复原_5631.db"
    )
    output_db = Path(r"./datasets/全牙_5.db")
    extract_simple(input_db=INPUT_DB,output_db=output_db)
