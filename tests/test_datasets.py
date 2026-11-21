"""MyDataset / Occupancy / LMDB 重构回归测试。

无需 GPU、实际患者文件或在 CI 安装模型权重。
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from datasets import CrownOccupancySampler, SamplingConfig, MyDataset, build_database


class Tetra:
    def __init__(self):
        self.vertices = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
        ])
        self.faces = np.array([
            [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]
        ])


def register_fake_sdf(monkeypatch, *, constant=None):
    for name in ("sindre", "sindre.ai", "sindre.ai.vision3d",
                 "sindre.ai.vision3d.implicit_utils",
                 "sindre.ai.vision3d.implicit_utils.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = sys.modules["sindre.ai.vision3d.implicit_utils.utils"]

    class FakeGetSDF:
        def apply_pcu(self, vertices, faces, pts):
            if constant is not None:
                return np.full(len(pts), constant)
            return np.linalg.norm(np.asarray(pts) - np.array([0.5, 0.5, 0.5]), axis=1) - 0.72

    mod.GetSDF = FakeGetSDF


def test_config_counts_and_units():
    config = SamplingConfig()
    assert config.num_queries == 32768
    assert config.noise_mm == (0.05, 0.15, 0.35)
    with pytest.raises(ValueError):
        SamplingConfig(noise_mm=(-0.1, 0.1, 0.2))


def test_seeded_occupancy_sampling_and_region_counts(monkeypatch):
    register_fake_sdf(monkeypatch)
    sampler = CrownOccupancySampler()
    monkeypatch.setattr(sampler, "get_watertight", lambda v, f: Tetra())
    mesh = Tetra()
    a = sampler.sample(mesh.vertices, mesh.faces, seed=321)
    b = sampler.sample(mesh.vertices, mesh.faces, seed=321)
    assert np.array_equal(a["query_points"], b["query_points"])
    assert np.array_equal(a["occupancy"], b["occupancy"])
    assert a["query_points"].shape == (32768, 3)
    assert a["occupancy"].shape == (32768,)
    assert set(np.unique(a["occupancy"])) == {0, 1}
    assert np.bincount(a["query_region"], minlength=4).tolist() == [
        8192, 4096, 4096, 16384,
    ]
    assert np.all(a["occupancy"] == (a["sdf_mm"] < 0))
    assert a["inside_fraction_by_region"].shape == (4,)
    assert np.all((a["inside_fraction_by_region"] >= 0)
                  & (a["inside_fraction_by_region"] <= 1))


def test_invalid_occupancy_labels_rejected(monkeypatch):
    register_fake_sdf(monkeypatch, constant=1.0)
    sampler = CrownOccupancySampler()
    monkeypatch.setattr(sampler, "get_watertight", lambda v, f: Tetra())
    mesh = Tetra()
    with pytest.raises(ValueError, match="比例异常"):
        sampler.sample(mesh.vertices, mesh.faces)


def test_same_random_jaw_crop_does_not_change_occ_query_points(monkeypatch):
    record = {
        "jaw_points": np.array([
            [0., 0., 0., 0., 0., 1.],
            [2., 0., 0., 0., 0., 1.],
            [3., 0., 0., 0., 0., 1.],
        ], dtype=np.float32),
        "jaw_labels": np.array([1, 2, 9]),
        "jaw": 1, "target_position": 6,
        "margin_line": np.zeros((128, 3), dtype=np.float32),
        "size": np.ones(3, dtype=np.float32),
        "view_images": np.zeros((3, 64, 64, 3), dtype=np.uint8),
        "query_points": np.ones((8, 3), dtype=np.float32),
        "occupancy": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        "query_region": np.array([0, 1, 2, 3, 0, 1, 2, 3]),
        "sdf_mm": np.array([1, -1, 1, -1, 1, -1, 1, -1]),
        "inside_fraction_by_region": np.ones(4, dtype=np.float32) * 0.5,
        "target_vertices": Tetra().vertices.astype(np.float32),
        "target_faces": Tetra().faces.astype(np.int32),
    }

    fake_lmdb = types.ModuleType("sindre.lmdb")

    class Reader:
        def __init__(self, path):
            pass
        def get_meta(self, name):
            return {"build_complete": True, "schema_version": "toothforge3d_occ_v1"}.get(name)
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return record
        def close(self):
            pass

    fake_lmdb.Reader = Reader
    monkeypatch.setitem(sys.modules, "sindre.lmdb", fake_lmdb)

    dataset = MyDataset("fake.db", jaw_points=12, augment=True, seed=45)
    first, second = dataset[0], dataset[1]
    assert first["jaw_points"].shape == (12, 6)
    assert first["view_images"].shape == (3, 3, 64, 64)
    assert np.array_equal(first["occupancy"], second["occupancy"])
    assert np.array_equal(first["query_points"], second["query_points"])
    collated = dataset.collate([first, second])
    assert collated["occupancy"].shape == (2, 8)
    assert len(collated["target_vertices"]) == 2
    dataset.close()


def test_writer_persists_new_db_sample_without_touching_source(monkeypatch, tmp_path):
    original = tmp_path / "original.db"
    original.write_bytes(b"source")
    target = tmp_path / "toothforge_v1.db"
    rows = []
    meta = {}

    class Reader:
        def __init__(self, path):
            assert Path(path) == original
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {"fdi": 31}

    class Writer:
        def __init__(self, path, map_size_limit):
            assert Path(path) == target
            assert map_size_limit == 1024
        def put_meta(self, key, value):
            meta[key] = value
        def put_sample(self, row):
            rows.append(row)
        def close(self):
            pass

    modules = types.ModuleType("sindre.lmdb")
    modules.Reader, modules.Writer = Reader, Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", modules)

    class FakeCase:
        def __init__(self, raw):
            self.target_fdi = raw["fdi"]
        def get_condition(self, **kwargs):
            assert kwargs["render_views"] is True
            return dict(
                geometry_ready=True,
                target_vertices=(Tetra().vertices * 8).astype(np.float32),
                target_faces=Tetra().faces.astype(np.int32),
                source_fdi=31,
                jaw=1, target_position=1,
                margin_line=np.zeros((128, 3), dtype=np.float32),
                size=np.array([1, 1, 1], dtype=np.float32),
                jaw_points=np.zeros((0, 6), dtype=np.float32),
                jaw_labels=np.zeros((0,), dtype=np.int64),
                view_images=np.zeros((3, 64, 64, 3), dtype=np.uint8),
                view_features=None,
                origin_world=np.zeros(3, dtype=np.float32),
                local_frame=np.eye(3, dtype=np.float32),
                frame_verified=True,
            )
    monkeypatch.setattr("datasets.DentalCase", FakeCase)
    def fake_sample(self, *args, **kwargs):
        return dict(
            query_points=np.zeros((8, 3), dtype=np.float32),
            occupancy=np.array([0, 1] * 4, dtype=np.uint8),
            query_region=np.array([0, 1, 2, 3] * 2, dtype=np.uint8),
            sdf_mm=np.array([1., -1.] * 4, dtype=np.float32),
            inside_fraction=0.5,
            inside_fraction_by_region=np.ones(4, dtype=np.float32) * 0.5,
        )
    monkeypatch.setattr(CrownOccupancySampler, "sample", fake_sample)
    result = build_database(
        original, target, config=SamplingConfig(image_size=64),
        map_size_mb=1024,
    )
    assert result["written"] == 1
    assert meta["schema_version"] == "toothforge3d_occ_v1"
    assert meta["build_complete"] is True
    assert rows[0]["occupancy"].shape == (8,)
    assert rows[0]["source_fdi"] == 31
    assert original.read_bytes() == b"source"


def test_input_output_db_must_differ(tmp_path):
    a = tmp_path / "same.db"
    a.touch()
    with pytest.raises(ValueError, match="不能相同"):
        build_database(a, a)
