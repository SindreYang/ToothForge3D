"""CrownFDG CPU 回归测试：开放边界、颈缘线柔性 Loss、FDG 官方接口适配。

O-Voxel 的实际 CUDA hashmap 解码必须在安装原生依赖的 GPU 环境验证。
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import CrownFDG, FDGConfig
from process import FDGPreprocessor
from src.losses import CrownFDGLoss


def open_tetra():
    vertices = np.array(
        [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]],
        dtype=np.float32,
    )
    # 一个开放的三角边界 (0,1,2)，第 4 个点在上方。
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
    return vertices, faces, vertices[[0, 1, 2]].copy()


def test_margin_soft_loss_and_gradients():
    vertices, faces, margin = open_tetra()
    fdg = CrownFDG(FDGConfig(margin_tolerance_mm=0.05))
    v = torch.tensor(vertices, requires_grad=True)
    f = torch.from_numpy(faces).long()
    line = torch.tensor(margin)
    assert fdg.get_boundary_indices(f, len(v)).numel() == 3
    exact = CrownFDGLoss().get_margin_loss(v, f, line)
    assert exact["loss"].item() < 1e-8
    # 仅把一个边界点错开，应该有非零 Loss 和对顶点的反向梯度。
    perturb = vertices.copy()
    perturb[0] += np.array([0., 0., 0.4], dtype=np.float32)
    prediction = torch.tensor(perturb, requires_grad=True)
    out = CrownFDGLoss().get_margin_loss(prediction, f, line)
    assert out["loss"].item() > 0
    out["loss"].backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert out["max_distance_mm"].item() > 0.3


def test_margin_no_boundary_must_not_count_as_success():
    verts, faces, margin = open_tetra()
    closed = np.concatenate([faces, [[0, 2, 1]]], axis=0)
    with pytest.raises(ValueError, match="无开放边界"):
        CrownFDGLoss().get_margin_loss(
            torch.from_numpy(verts), torch.from_numpy(closed), torch.from_numpy(margin)
        )
    with pytest.raises(ValueError, match="没有开放边界"):
        CrownFDG().refine_margin(
            torch.from_numpy(verts), torch.from_numpy(closed), torch.from_numpy(margin)
        )


def test_refine_margin_is_bounded_and_does_not_move_interior():
    verts, faces, margin = open_tetra()
    verts[0, 2] += 0.35
    model = CrownFDG(FDGConfig(
        margin_tolerance_mm=0.05,
        max_refine_step_mm=0.1,
        refine_strength=1.0,
        max_refine_distance_mm=0.8,
    ))
    before = torch.from_numpy(verts)
    fixed = model.refine_margin(
        before, torch.from_numpy(faces).long(), torch.from_numpy(margin)
    )
    assert (fixed[0] - before[0]).norm().item() <= 0.100001
    assert fixed[0, 2] < before[0, 2]
    assert torch.allclose(fixed[3], before[3])  # 内部牙尖不会被移动
    assert torch.equal(before, torch.from_numpy(verts))


def test_open_crown_does_not_require_meshlib():
    verts, faces, margin = open_tetra()
    out_v, out_f = CrownFDG().get_open_crown(verts, faces, margin)
    assert np.array_equal(out_v, verts)
    assert np.array_equal(out_f, faces)


def test_closed_crown_can_use_sindre_meshlib(monkeypatch):
    verts, faces, margin = open_tetra()
    closed_faces = np.concatenate([faces, [[0, 2, 1]]], axis=0)
    monkeypatch.setitem(sys.modules, "sindre", types.ModuleType("sindre"))
    monkeypatch.setitem(sys.modules, "sindre.utils3d", types.ModuleType("sindre.utils3d"))
    algorithm = types.ModuleType("sindre.utils3d.core.algorithm")
    def fake_cut(v, f, loop_points, *, get_bigger_part, smooth_boundary):
        assert not get_bigger_part and not smooth_boundary
        assert np.allclose(loop_points, margin)
        return verts, faces, verts * [1, 1, -1], faces
    algorithm.cut_mesh_by_meshlib = fake_cut
    monkeypatch.setitem(sys.modules, "sindre.utils3d.core.algorithm", algorithm)
    out_v, out_f = CrownFDG().get_open_crown(verts, closed_faces, margin)
    assert np.array_equal(out_f, faces)


def test_fdg_encode_delegates_to_ovoxel_without_watertight_repair(monkeypatch):
    verts, faces, margin = open_tetra()
    calls = []
    cvt = types.ModuleType("o_voxel.convert")
    def fake_encode(v, f, *, grid_size, aabb, face_weight,
                    boundary_weight, regularization_weight):
        calls.append((v, f, grid_size, aabb, face_weight, boundary_weight,
                      regularization_weight))
        coords = torch.tensor([[10, 9, 8], [10, 9, 9]], dtype=torch.int32)
        # 模拟官方绝对的、归一化到 [0,1] 的 dual vertex 坐标。
        dual = (coords.float() + 0.25) / grid_size
        intersected = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
        return coords, dual, intersected
    cvt.mesh_to_flexible_dual_grid = fake_encode
    pkg = types.ModuleType("o_voxel")
    pkg.convert = cvt
    monkeypatch.setitem(sys.modules, "o_voxel", pkg)
    monkeypatch.setitem(sys.modules, "o_voxel.convert", cvt)
    cfg = FDGConfig(resolution=64)
    result = CrownFDG(cfg).get_fdg(verts, faces, margin)
    assert len(calls) == 1
    assert calls[0][2] == 64
    assert np.allclose(result["dual_offsets"], 0.25)
    assert result["intersected"].tolist() == [[1, 0, 0], [0, 1, 0]]
    center = result["center_mm"]
    scale = result["scale_mm"]
    restored = (calls[0][0].numpy() * scale + center)
    assert np.allclose(restored, verts, atol=1e-5)
    assert np.array_equal(calls[0][1].numpy(), faces)


def test_fdg_loss_gradient_and_strict_alignment():
    from src.losses import CrownFDGLoss

    pred = {
        "coarse_logits": torch.zeros(1, 4, 4, 4, requires_grad=True),
        "fine_logits": torch.zeros(1, 8, requires_grad=True),
        "dual_offsets": torch.zeros(1, 8, 3, requires_grad=True),
        "intersected_logits": torch.zeros(1, 8, 3, requires_grad=True),
    }
    coarse = torch.zeros(1, 4, 4, 4)
    coarse[0, 1, 1, 1] = 1
    batch = {
        "coarse_target": coarse,
        "fine_active": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32),
        "fdg_target_offsets": torch.ones(1, 8, 3) * 0.5,
        "fdg_target_intersected": torch.ones(1, 8, 3),
    }
    terms = CrownFDGLoss()(pred, batch)
    assert all(torch.isfinite(v).all() for v in terms.values())
    terms["loss"].backward()
    for key in pred:
        assert pred[key].grad is not None
        assert torch.isfinite(pred[key].grad).all()
    with pytest.raises(ValueError, match="形状"):
        CrownFDGLoss()(pred, {**batch, "fine_active": torch.zeros(1, 3)})


def test_fdg_config_errors():
    with pytest.raises(ValueError):
        FDGConfig(resolution=8)
    with pytest.raises(ValueError):
        FDGConfig(padding_ratio=0.45)
    with pytest.raises(ValueError):
        FDGConfig(margin_samples_per_segment=0)


def test_native_ovoxel_roundtrip_when_installed_on_cuda():
    """真实 FDG 转换/解码必须在配好 O-Voxel CUDA 扩展的环境运行。"""
    pytest.importorskip("o_voxel.convert")
    if not torch.cuda.is_available():
        pytest.skip("O-Voxel FDG mesh extraction requires CUDA")
    verts, faces, margin = open_tetra()
    converter = CrownFDG(FDGConfig(resolution=64))
    fdg = converter.get_fdg(verts, faces, margin)
    predicted, triangles = converter.get_mesh(fdg)
    assert predicted.ndim == 2 and predicted.shape[1] == 3
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert len(triangles) > 0 and torch.isfinite(predicted).all()
    measured = converter.get_roundtrip_geometry(verts, faces, margin, predicted, triangles)
    assert measured["surface_p95_mm"] <= 0.5
    assert measured["surface_sample_max_mm"] <= 0.5
    assert measured["margin_to_surface_max_mm"] <= 0.5
    # 原生解码拓扑单独记录，不作为数据阶段断言。
    assert isinstance(measured["topology_pass"], bool)



def test_raw_case_database_builds_fdg_without_any_sdf_or_occupancy(monkeypatch, tmp_path):
    """直接原始订单读取，不能要求先构造水密冠或闭合 SDF。"""
    src = tmp_path / "original_cases.db"
    dst = tmp_path / "native_fdg.db"
    src.write_bytes(b"raw-original")
    verts, faces, margin = open_tetra()
    rows, metas, calls = [], {}, []
    image_encoder = object()
    conditions = {
        "jaw": 0, "target_position": 6,
        "margin_line": margin,
        "size": np.array([3., 3., 3.], np.float32),
        "view_features": np.ones((3, 4, 8), np.float32),
        "jaw_points": np.zeros((5, 6), np.float32),
        "jaw_labels": np.zeros(5, np.int64),
        "jaw_present": True, "source_fdi": 16,
        "target_vertices": verts, "target_faces": faces,
        "origin_world": np.array([1., 2., 3.], np.float32),
        "local_frame": np.eye(3, dtype=np.float32),
        "world_to_local_transform": np.array([
            [1., 0., 0., -1.],
            [0., 1., 0., -2.],
            [0., 0., 1., -3.],
            [0., 0., 0., 1.],
        ], dtype=np.float32),
        "frame_source": "scan_arch_curve",
    }
    class RawReader:
        def __init__(self, path):
            assert path == str(src)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {"raw_order": 123}
        def get_meta(self, key):
            return None   # 原始病例没有版本号/完成状态，仍可读
    class Writer:
        def __init__(self, path, map_size_limit):
            assert path == str(dst)
        def put_meta(self, key, value):
            metas[key] = value
        def put_sample(self, row):
            rows.append(row)
        def close(self):
            pass
    class DentalCase:
        def __init__(self, data):
            assert data == {"raw_order": 123}
        def get_condition(self, *, image_encoder, crop_mode,
                          tooth_dropout, num_points, render_views, image_size):
            calls.append((image_encoder, crop_mode, tooth_dropout))
            return conditions

    fake_lmdb = types.ModuleType("sindre.lmdb")
    fake_lmdb.Reader, fake_lmdb.Writer = RawReader, Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", fake_lmdb)
    monkeypatch.setitem(sys.modules, "sindre", types.ModuleType("sindre"))
    process = types.ModuleType("process")
    process.DentalCase = DentalCase
    monkeypatch.setitem(sys.modules, "process", process)
    cvt = CrownFDG(FDGConfig(resolution=32))
    def fake_fdg(v, f, line, *, cut_closed):
        assert cut_closed is False
        return {
            "coords":np.ones((2,3), dtype=np.int32),
            "dual_offsets":np.ones((2,3),dtype=np.float32)*0.4,
            "intersected":np.ones((2,3),dtype=np.uint8),
            "center_mm":np.zeros(3,dtype=np.float32),
            "scale_mm":8.0,
            "margin_line_mm":line,
            "resolution":32,
        }
    monkeypatch.setattr(cvt, "get_fdg", fake_fdg)
    monkeypatch.setattr(cvt, "get_mesh_quality", lambda *args: {"boundary_loops": 1})
    result = FDGPreprocessor(cvt, case_class=DentalCase).build_database(src,dst,image_encoder=image_encoder,max_cases=1,augment_variants=1)
    assert result["written"] == 1 and not result["errors"]
    assert calls == [(image_encoder, "full", 0)]
    assert metas["coordinate_frame"] == "scan_arch_margin_v3"
    assert metas["build_complete"] is True
    assert "fdg_coords" in rows[0] and "view_features" in rows[0]
    assert np.array_equal(rows[0]["origin_world"], conditions["origin_world"])
    assert np.array_equal(rows[0]["local_frame"], conditions["local_frame"])
    assert np.array_equal(
        rows[0]["world_to_local_transform"],
        conditions["world_to_local_transform"],
    )
    assert rows[0]["frame_source"] == "scan_arch_curve"
    assert "sdf_mm" not in rows[0] and "occupancy" not in rows[0]
    assert "raw_order" not in rows[0]
    assert src.read_bytes() == b"raw-original"



def test_paired_dimension_augmentation_keeps_margin_vertices_fixed():
    """冠身形变改变 MD/BL/H 目标，同时颈缘线与上下颌图像保持固定。"""
    verts, faces, margin = open_tetra()
    # 牙尖的 MD/BL 外凸应超出颈缘范围，才能分别验证三个尺寸条件。
    verts[3, 0] = 2.6
    verts[3, 1] = 2.5
    original = verts.copy()
    original_size = np.array([3., 3., 3.], np.float32)
    updated, size = CrownFDG.augment_crown_size(
        verts, faces, margin, original_size, (1.10, 0.92, 1.15)
    )
    boundary = CrownFDG.get_boundary_indices(torch.from_numpy(faces).long(), len(verts))
    assert np.array_equal(updated[boundary], verts[boundary])
    assert not np.array_equal(updated[3], verts[3])
    assert np.all(size > 0)
    assert np.all(abs(size - original_size) > 0.001), "MD、BL、H 应分别改变"
    assert np.array_equal(verts, original)


def test_dimension_variants_use_one_source_group(monkeypatch, tmp_path):
    """同病例不同尺寸副本在拆分验证集时必须绑定为一个组。"""
    raw = tmp_path / "cases.db"
    raw.write_bytes(b"original")
    new = tmp_path / "fdg.db"
    verts, faces, margin = open_tetra()
    rows, meta = [], {}

    class Reader:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __len__(self): return 1
        def __getitem__(self, index): return {"case_id": 1}
        def get_meta(self, key): return None

    class Writer:
        def __init__(self, path, map_size_limit): pass
        def put_meta(self, key, value): meta[key] = value
        def put_sample(self, row): rows.append(row)
        def close(self): pass

    class DentalCase:
        def __init__(self, data): pass
        def get_condition(self, **kwargs):
            return {
                "target_vertices":verts,"target_faces":faces,
                "margin_line":margin,"size":np.array([3.,3.,3.],np.float32),
                "jaw":0, "target_position":6,
                "jaw_points":np.zeros((3,6),np.float32),
                "jaw_labels":np.zeros(3,np.int64),"jaw_present":True,
                "view_features":np.ones((3,4,8),np.float32),
                "frame_source":"provided",
            }
    lmdb = types.ModuleType("sindre.lmdb")
    lmdb.Reader, lmdb.Writer = Reader, Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", lmdb)
    process = types.ModuleType("process")
    process.DentalCase = DentalCase
    monkeypatch.setitem(sys.modules, "process", process)
    converter = CrownFDG(FDGConfig(resolution=32))
    def fake_get_fdg(v, f, line, **kwargs):
        return {
            "coords":np.array([[1,2,3]],np.int32),
            "dual_offsets":np.array([[0.4,0.4,0.4]],np.float32),
            "intersected":np.ones((1,3),np.uint8),
            "center_mm":np.zeros(3,np.float32),
            "scale_mm":10., "resolution":32,
            "margin_line_mm":line,
        }
    monkeypatch.setattr(converter,"get_fdg",fake_get_fdg)
    monkeypatch.setattr(converter,"get_mesh_quality",lambda *args: {"boundary_loops": 1})
    report = FDGPreprocessor(converter, case_class=DentalCase).build_database(
        raw, new, augment_variants=3, max_cases=1, augment_scale=.10,
        collision_check=False,  # 合成 Mock 没有完整牙颌环境，只验证尺寸标签配对。
    )
    assert report["written"] == 3
    assert meta["source_groups"] == [0,0,0]
    assert [r["variant_index"] for r in rows] == [0,1,2]
    assert not np.array_equal(rows[0]["target_vertices"],rows[1]["target_vertices"])
    assert not np.array_equal(rows[0]["size"],rows[1]["size"])
    assert np.array_equal(rows[0]["margin_line"],rows[1]["margin_line"])
    assert np.array_equal(rows[0]["view_features"],rows[1]["view_features"])



def test_contact_context_selects_real_neighbors_and_opposing_arch():
    """仅检验真实邻牙/对颌曲面，忽略目标牙和 2000+FDI 预备体标签。"""

    class Case:
        target_fdi = 16
        valid_fdi = (15, 16, 17)
        frame = np.eye(3, dtype=np.float32)
        origin_world = np.zeros(3, dtype=np.float32)
        world_to_local_transform = np.eye(4)
        # 四个独立三角形，分别是邻牙、目标冠、预备体、另一颗邻牙。
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 3, 0], [1, 3, 0], [0, 4, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
            [0, -1, 0], [1, -1, 0], [0, -2, 0],
        ], np.float32)
        faces = np.arange(12).reshape(-1, 3)
        fdi_labels = np.repeat([15, 16, 2016, 17], 3)
        opposite_vertices = np.array([
            [0, 0, 3], [1, 0, 3], [0, 1, 3],
        ], np.float32)
        opposite_faces = np.array([[0, 1, 2]], np.int64)

    target = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], np.float32)
    found = CrownFDG.get_environment_surfaces(Case(), target, padding_mm=4)
    assert set(found) == {"adjacent_15", "adjacent_17", "opposing"}
    assert np.array_equal(found["adjacent_15"][1], [[0, 1, 2]])
    assert np.array_equal(found["adjacent_17"][1], [[0, 1, 2]])
    assert not any("2016" in name or name == "adjacent_16" for name in found)


def test_preprocess_rejects_new_collisions_before_writing_fdg(monkeypatch, tmp_path):
    """增强候选相交时应丢弃重试，且 FDG 只能编码安全候选。"""
    src = tmp_path / "original.db"
    dst = tmp_path / "filtered.db"
    src.write_bytes(b"source")
    v, f, line = open_tetra()
    rows, metadata, collision_calls = [], {}, []

    class Reader:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __len__(self): return 1
        def __getitem__(self, index): return {"raw_id": 9}
        def get_meta(self, key): return None

    class Writer:
        def __init__(self, path, map_size_limit): pass
        def put_meta(self, key, value): metadata[key] = value
        def put_sample(self, row): rows.append(row)
        def close(self): pass

    class DentalCase:
        def __init__(self, raw): self.raw = raw
        def get_condition(self, **kwargs):
            return {
                "target_vertices":v, "target_faces":f, "margin_line":line,
                "size":np.array([3, 3, 3], np.float32),
                "jaw":0, "target_position":6,
                "jaw_points":np.zeros((3,6),np.float32),
                "jaw_labels":np.zeros(3,np.int64), "jaw_present":True,
                "view_features":np.ones((3,4,8),np.float32),
                "frame_source":"provided",
            }

    lmdb = types.ModuleType("sindre.lmdb")
    lmdb.Reader, lmdb.Writer = Reader, Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", lmdb)
    process = types.ModuleType("process")
    process.DentalCase = DentalCase
    monkeypatch.setitem(sys.modules, "process", process)

    converter = CrownFDG(FDGConfig(resolution=32))
    monkeypatch.setattr(
        converter, "get_environment_surfaces",
        lambda *a, **kw: {"opposing": (v, f)},
    )
    state = {"candidate": 0}

    def change_geometry(vertices, faces, margin, size, factors):
        state["candidate"] += 1
        modified = vertices.copy()
        # 第一个增强冠在被检查的位置产生交面片，第二个不会。
        modified[3, 0] = 10 if state["candidate"] == 1 else 0.5
        return modified, np.array([4, 4, 4], np.float32)

    def check_contact(vertices, faces, environment):
        collision_calls.append(vertices.copy())
        return {
            "opposing": {
                "collision": bool(vertices[3, 0] > 5),
                "surface_distance_mm": 0.0 if vertices[3, 0] > 5 else 1.0,
            }
        }

    def fake_get_fdg(vertices, faces, margin, *, cut_closed):
        assert vertices[3, 0] <= 5, "必须先执行碰撞检查，再计算 FDG"
        return {
            "coords": np.array([[1, 2, 3]], np.int32),
            "dual_offsets": np.ones((1, 3), np.float32) * .4,
            "intersected": np.ones((1, 3), np.uint8),
            "center_mm": np.zeros(3, np.float32), "scale_mm": 10.,
            "resolution":32, "margin_line_mm":margin,
        }

    monkeypatch.setattr(converter, "get_mesh_quality", lambda *a: {"boundary_loops": 1})
    monkeypatch.setattr(converter, "augment_crown_size", change_geometry)
    monkeypatch.setattr(converter, "get_collision_status", check_contact)
    monkeypatch.setattr(converter, "get_fdg", fake_get_fdg)
    report = FDGPreprocessor(converter, case_class=DentalCase).build_database(
        src, dst, augment_variants=2, max_cases=1,
        collision_check=True, max_collision_attempts=3,
    )
    assert report["written"] == 2
    assert report["rejected_collision"] == 1
    assert report["skipped_augmentation"] == 0
    assert metadata["augmentation"]["collision_check"] is True
    assert metadata["augmentation"]["penetration_depth_verified"] is False
    assert len(collision_calls) == 3  # 原始冠+碰撞冠+安全冠
    assert [row["variant_index"] for row in rows] == [0, 1]


def test_preprocess_skips_augmentation_if_original_already_collides(monkeypatch, tmp_path):
    """原始冠相交时仍允许尝试候选，但所有候选相交则只保留原始监督。"""
    src = tmp_path / "raw.db"
    dst = tmp_path / "fdg.db"
    src.write_bytes(b"source")
    v, f, line = open_tetra()
    rows, metadata = [], {}

    class Reader:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __len__(self): return 1
        def __getitem__(self, i): return {}
        def get_meta(self, key): return None

    class Writer:
        def __init__(self, path, map_size_limit): pass
        def put_meta(self, k, x): metadata[k] = x
        def put_sample(self, x): rows.append(x)
        def close(self): pass

    class DentalCase:
        def __init__(self, raw): pass
        def get_condition(self, **kwargs):
            return dict(
                target_vertices=v, target_faces=f, margin_line=line,
                size=np.array([3,3,3], np.float32), view_features=np.ones((3,4,8)),
                jaw=0, target_position=6, jaw_points=np.zeros((3,6)),
                jaw_labels=np.zeros(3, np.int64), jaw_present=True,
                frame_source="provided",
            )
    lmdb = types.ModuleType("sindre.lmdb")
    lmdb.Reader, lmdb.Writer = Reader, Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", lmdb)
    process = types.ModuleType("process")
    process.DentalCase = DentalCase
    monkeypatch.setitem(sys.modules, "process", process)

    converter = CrownFDG(FDGConfig(resolution=32))
    monkeypatch.setattr(converter, "get_mesh_quality", lambda *a: {"boundary_loops": 1})
    monkeypatch.setattr(converter, "get_environment_surfaces", lambda *args, **kwargs: {"opposing": (v, f)})
    monkeypatch.setattr(converter, "get_collision_status", lambda *a: {
        "opposing":{"collision":True, "surface_distance_mm":0.0}
    })
    monkeypatch.setattr(converter, "get_fdg", lambda v, f, margin, **kw: {
        "coords":np.array([[1,2,3]], np.int32),
        "dual_offsets":np.ones((1,3),np.float32),
        "intersected":np.ones((1,3),np.uint8),
        "resolution":32, "center_mm":np.zeros(3,np.float32),
        "scale_mm":10., "margin_line_mm":margin,
    })
    result = FDGPreprocessor(converter, case_class=DentalCase).build_database(src,dst,augment_variants=3)
    assert result["written"] == 1
    assert result["skipped_augmentation"] == 2
    assert result["warnings"]
    assert metadata["source_groups"] == [0]



def test_margin_only_frame_is_deterministic():
    """FDG 空间必须与用户是否指定 MD/BL/H 无关。"""
    line = np.array([
        [-4,-3,0],[4,-3,0],[4,3,0],[-4,3,0],
    ], np.float32)
    converter = CrownFDG()
    a = converter.get_condition_frame(line)
    b = converter.get_condition_frame(line.copy())
    c = converter.get_condition_frame(line * 1.0)
    assert np.allclose(a[0], b[0]) and np.allclose(a[0], c[0])
    assert a[1] == b[1] == c[1]

def test_get_mesh_quality_rejects_disconnected_and_duplicate_faces():
    """CAD 网格验证应拒绝重复面、多边界环与分离网格。"""
    pytest.importorskip("meshlib")
    v, faces, _ = open_tetra()
    result = CrownFDG.get_mesh_quality(v, faces)
    assert result["boundary_loops"] == 1
    assert result["connected_components"] == 1
    with pytest.raises(ValueError, match="重复三角面"):
        CrownFDG.get_mesh_quality(v, np.concatenate([faces, faces[:1]]))
    separate = np.concatenate((v, v + 10.0))
    with pytest.raises(ValueError, match="唯一颈缘环"):
        CrownFDG.get_mesh_quality(
            separate, np.concatenate((faces, faces + len(v))),
        )


def test_collision_status_reads_current_sindre_dictionary(monkeypatch):
    vertices, faces, _ = open_tetra()
    environment = {"opposing": (vertices, faces)}
    module_name = "sindre.utils3d.core.dental.utils"
    module = types.ModuleType(module_name)

    class Detector:
        def __init__(self, v, f, neighbors, *, solid):
            assert not solid
            assert neighbors is environment
            assert v.dtype == np.float64 and f.dtype == np.int64

        def query(self, *, clearance, all_contacts):
            assert clearance == 0.05 and not all_contacts
            return {"neighbors": ({"neighbor": "opposing", "collision": False,
                                   "within_clearance": True, "surface_distance": 0.02},)}

    module.AdjacentCollisionDetector = Detector
    monkeypatch.setitem(sys.modules, module_name, module)
    assert CrownFDG.get_collision_status(vertices, faces, environment) == {
        "opposing": {"collision": False, "within_clearance": True,
                     "surface_distance_mm": 0.02}
    }
