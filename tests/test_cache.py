"""病例缓存专项测试：不依赖真实病例、DINOv3 权重或 GPU。"""
import io
import numpy as np
import pytest
from process.cache import FDGCaseCache


def make_cache(tmp_path):
    cache = object.__new__(FDGCaseCache)
    cache.directory = tmp_path
    cache.options = {"augment_variants": 4}
    return cache


def make_record():
    payload = io.BytesIO()
    np.savez(payload, view_features=np.ones((3, 5, 8), np.float32),
             view_mask=np.ones(3, bool), jaw_points=np.zeros((2, 6), np.float32),
             jaw_labels=np.zeros(2, np.int32))
    key = "scan_0_local_1"
    record = {
        "variant_index": 0, "source_fdi": 16, "size": np.ones(3, np.float32),
        "target_vertices": np.zeros((4, 3), np.float32),
        "target_faces": np.array([[0, 1, 2]], np.int32),
        "margin_line": np.zeros((128, 3), np.float32),
        "fdg_coords": np.zeros((2, 3), np.int32),
        "fdg_dual_offsets": np.zeros((2, 3), np.float32),
        "fdg_intersected": np.ones((2, 3), bool),
        "fdg_resolution": 128, "fdg_center_mm": np.zeros(3, np.float32),
        "fdg_scale_mm": 24., "view_features": np.ones((3, 5, 8), np.float32),
        "view_mask": np.ones(3, bool), "view_scale_mm": np.ones(3, np.float32),
        "jaw_points": np.zeros((2, 6), np.float32), "jaw_labels": np.zeros(2, np.int32),
        "scan_keys": [key],
    }
    return record, {key: payload.getvalue()}


def test_cache_key_input_content_and_index():
    raw = {"mesh": np.arange(12, dtype=np.float32).reshape(4, 3), "fdi": 16}
    key = FDGCaseCache.get_key(raw, 0)
    assert key == FDGCaseCache.get_key(dict(reversed(list(raw.items()))), 0)
    assert key != FDGCaseCache.get_key(raw, 1)
    changed = dict(raw, mesh=raw["mesh"].copy())
    changed["mesh"][0, 0] += 1
    assert key != FDGCaseCache.get_key(changed, 0)
    assert key != FDGCaseCache.get_key(dict(raw, mesh=raw["mesh"].astype(np.float64)), 0)


def test_cache_roundtrip_and_precise_invalidate(tmp_path):
    cache = make_cache(tmp_path)
    record, bank = make_record()
    key = FDGCaseCache.get_key({"test": 1}, 0)
    other = FDGCaseCache.get_key({"test": 2}, 1)
    assert cache.load(key) is None
    cache.save(key, [record, dict(record, variant_index=2)], bank)
    cache.save(other, [record], bank)
    records, actual_bank = cache.load(key)
    assert actual_bank == bank
    assert len(records) == 2 and int(records[1]["variant_index"]) == 2
    for field, expected in record.items():
        assert np.array_equal(records[0][field], expected), field
    FDGCaseCache.invalidate(tmp_path, key)
    assert cache.load(key) is None and cache.load(other) is not None
    with pytest.raises(ValueError):
        FDGCaseCache.invalidate(tmp_path, "../outside")


def test_cache_corrupt_and_missing_conditions(tmp_path):
    cache = make_cache(tmp_path)
    record, bank = make_record()
    key = FDGCaseCache.get_key({"case": 1}, 0)
    cache.save(key, [record], {})
    assert cache.load(key) is None
    cache.save(key, [dict(record, fdg_dual_offsets=np.full((2,3), np.nan))], bank)
    assert cache.load(key) is None
    cache.save(key, [record], {next(iter(bank)): b"broken"})
    assert cache.load(key) is None
    (tmp_path / f"{key}.npz").write_bytes(b"truncated")
    assert cache.load(key) is None


def test_atomic_write_keeps_prior_cache_on_failure(tmp_path):
    path = tmp_path / "cache.npz"
    path.write_bytes(b"prior")
    def fail(stream):
        stream.write(b"partial")
        raise RuntimeError("simulated write interruption")
    with pytest.raises(RuntimeError):
        FDGCaseCache.atomic_write(path, fail)
    assert path.read_bytes() == b"prior"
    assert list(tmp_path.iterdir()) == [path]
