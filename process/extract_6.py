"""从原始 SindrePy LMDB 中提取 FDI 6 号牙病例。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


INPUT_DB = Path(
    r"C:\Users\sindre\Downloads\网格生成_邻牙已分割_冠复原_5631.db"
)
TOOTH_6_FDI = {16, 26, 36, 46}


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


def extract_6(input_db: Path, output_db: Path | None = None) -> int:
    """筛选 FDI 16/26/36/46 病例并写入新 SindrePy LMDB。"""
    input_db = Path(input_db)
    if not input_db.exists():
        raise FileNotFoundError(f"输入数据库不存在：{input_db}")
    if output_db is None:
        output_db = input_db.with_name(f"{input_db.stem}_6{input_db.suffix}")
    output_db = Path(output_db)
    if input_db.resolve() == output_db.resolve():
        raise ValueError("输出数据库不能与输入数据库相同")
    if output_db.exists():
        raise FileExistsError(f"禁止覆盖已有数据库：{output_db}")

    from sindre.lmdb import Reader, Writer

    with Reader(str(input_db)) as reader:
        source_count = len(reader)
        selected_indices = [
            index for index in range(source_count)
            if get_target_fdi(reader[index], index) in TOOTH_6_FDI
        ]
        if not selected_indices:
            raise ValueError(f"数据库中没有 FDI 6 号牙病例：{input_db}")

        output_db.parent.mkdir(parents=True, exist_ok=True)
        writer = Writer(str(output_db), map_size_limit=1024)
        try:
            schema_version = reader.get_meta("schema_version")
            if schema_version is not None:
                writer.put_meta("schema_version", schema_version)
            writer.put_meta("filter_fdi", sorted(TOOTH_6_FDI))
            writer.put_meta("source_count", source_count)
            writer.put_meta("filtered_count", len(selected_indices))
            for index in selected_indices:
                writer.put_sample(reader[index])
        finally:
            writer.close()

    print(f"筛选完成：{len(selected_indices)} / {source_count} 条")
    print(f"输出数据库：{output_db}")
    return len(selected_indices)


if __name__ == "__main__":
    extract_6(INPUT_DB)
