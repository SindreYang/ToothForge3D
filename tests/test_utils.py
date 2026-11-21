"""CrownFDG CPU 回归测试：开放边界、颈缘线柔性 Loss、FDG 官方接口适配。

O-Voxel 的实际 CUDA hashmap 解码必须在安装原生依赖的 GPU 环境验证。
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from utils import CrownFDG, FDGConfig
from losses import CrownFDGLoss


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
    algorithm = types.ModuleType("sindre.utils3d.algorithm")
    def fake_cut(v, f, loop_points, *, get_bigger_part, smooth_boundary):
        assert get_bigger_part and not smooth_boundary
        assert np.allclose(loop_points, margin)
        return verts, faces, verts, faces
    algorithm.cut_mesh_by_meshlib = fake_cut
    monkeypatch.setitem(sys.modules, "sindre.utils3d.algorithm", algorithm)
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
    from losses import CrownFDGLoss

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


def test_new_lmdb_writer_drops_closed_sdf_targets(monkeypatch, tmp_path):
    original = tmp_path / "source.db"
    original.write_bytes(b"original")
    target = tmp_path / "open_fdg.db"
    verts, faces, margin = open_tetra()
    output, meta = [], {}
    raw = {
        "target_vertices": verts, "target_faces": faces,
        "margin_line": margin,
        "query_points": np.zeros((10, 3), np.float32),
        "occupancy": np.zeros(10, np.uint8),
        "sdf_mm": np.ones(10, np.float32),
    }
    class Reader:
        def __init__(self, path):
            assert path == str(original)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return raw
        def get_meta(self, key):
            return True if key == "build_complete" else None
    class Writer:
        def __init__(self, path, map_size_limit):
            assert path == str(target) and map_size_limit == 16
        def put_meta(self, key, value):
            meta[key] = value
        def put_sample(self, data):
            output.append(data)
        def close(self):
            pass
    lmdb = types.ModuleType("sindre.lmdb")
    lmdb.Reader, lmdb.Writer = Reader, Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", lmdb)
    monkeypatch.setitem(sys.modules, "sindre", types.ModuleType("sindre"))

    model = CrownFDG()
    def fake_get_fdg(*args, **kwargs):
        return {
            "coords": np.ones((2, 3), np.int32),
            "dual_offsets": np.ones((2, 3), np.float32),
            "intersected": np.ones((2, 3), np.uint8),
            "resolution": 128, "center_mm": np.zeros(3, np.float32),
            "scale_mm": 5.0, "margin_line_mm": margin,
        }
    monkeypatch.setattr(model, "get_fdg", fake_get_fdg)
    report = model.build_database(original, target, max_cases=1, map_size_mb=16)
    assert report["written"] == 1
    assert meta["schema_version"] == "toothforge3d_fdg_v1"
    assert meta["build_complete"] is True
    assert "fdg_coords" in output[0] and "fdg_margin_line_mm" in output[0]
    assert "occupancy" not in output[0] and "sdf_mm" not in output[0]
    assert np.array_equal(output[0]["target_faces"], faces)
    assert original.read_bytes() == b"original"
    assert target.with_suffix(".report.json").exists()



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
    boundary = converter.get_boundary_indices(triangles, len(predicted))
    assert len(boundary) > 0, "FDG 复原时不应丢失开放冠边界"
    measured = CrownFDGLoss().get_margin_loss(
        predicted, triangles, torch.tensor(margin, device=predicted.device),
    )
    assert measured["mean_distance_mm"] < 0.5
