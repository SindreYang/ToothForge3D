"""Path 型推理：不读目标设计冠 Mesh，尺寸使用显式输入，输出还原世界坐标。"""

import json

import numpy as np
import pytest

from src.infer import get_case, get_world_vertices


def test_json_case_extracts_fdi_but_removes_design_geometry(tmp_path):
    """设计冠只能提供牙位标识，几何不能进入推理条件。"""
    case = {
        "current_mesh": {
            "vertices": [[0,0,0],[2,0,0],[0,2,0],[0,0,2]],
            "faces": [[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
            "vertex_labels": [0,0,0,0],
        },
        "design_mesh": {
            "fdi": 16,
            "vertices": [[100,100,100]] * 3,
            "faces": [[0,1,2]],
        },
        "opposite_mesh": None,
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    margin = np.array([[10,20,30],[12,20,30],[10,22,30]], np.float32)
    input_case = get_case(
        path, target_fdi=16, margin_line=margin,
        size_mm=(9.,8.,7.), local_frame=np.eye(3),
    )
    assert input_case["target_fdi"] == 16
    assert np.array_equal(input_case["margin_line"], margin)
    assert np.array_equal(input_case["size_mm"], [9.,8.,7.])
    assert "design_mesh" not in input_case
    assert "design_mesh" not in input_case
    assert "merge_info" not in input_case
    assert np.array_equal(input_case["local_frame"], np.eye(3))


def test_missing_explicit_size_does_not_read_design_crown(tmp_path):
    case = {"current_mesh": {"vertices": [[0,0,0]], "faces": []},
            "design_mesh": {"fdi":16,"size_mm":[9,8,7]}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    result = get_case(
        path, target_fdi=16,
        margin_line=np.ones((3,3)), local_frame=np.eye(3),
    )
    assert result.get("size_mm") is None
    assert "design_mesh" not in result


def test_frame_override_invalidates_old_dino_features(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({
        "current_mesh":{},
        "target_fdi":16,
        "margin_line":[[0,0,0],[1,0,0],[0,1,0]],
        "size_mm":[8,7,6],
        "local_frame":np.eye(3).tolist(),
        "view_features":[[[1]]]*3,
    }), encoding="utf-8")
    without_override = get_case(path)
    assert "view_features" in without_override
    with_override = get_case(path, local_frame=np.eye(3))
    assert "view_features" not in with_override


def test_world_mesh_uses_original_input_margin_and_frame():
    local = np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32)
    margin = np.array([[11,20,30],[13,20,30],[12,20,30]], np.float32)
    frame = np.array([[0,-1,0],[1,0,0],[0,0,1]], np.float32)
    from src.views import get_world_to_local_transform
    transform = get_world_to_local_transform(margin.mean(axis=0), frame)
    world = get_world_vertices(local, transform)
    assert np.allclose(world[0], [12,20,30])
    assert np.allclose(world[1], [12,21,30])
    assert np.allclose(world[2], [11,20,30])
    from sindre.utils3d.sindrepcd import SindrePCD
    restored = SindrePCD(world).apply_transform(transform)
    assert np.allclose(restored.points, local)


def test_inference_mode_accepts_no_design_mesh():
    """构造本地扫描+显式条件，DentalCase 不应访问真实设计冠。"""
    pytest.importorskip("sindre")
    from process import DentalCase
    data = {
        "current_mesh": {
            "vertices": np.array([
                [0,0,0],[1,0,0],[0,1,0],[0,0,1]
            ], np.float32),
            "faces": np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], np.int64),
            "vertex_labels": np.full(4, 2016, np.int64),
        },
        "target_fdi":16,
        "margin_line":np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32),
        "size_mm":np.array([8,7,6],np.float32),
        "local_frame":np.eye(3,dtype=np.float32),
        "view_features":np.ones((3,4,16),np.float32),
    }
    case = DentalCase(data,inference=True)
    result = case.get_condition(crop_mode="full",num_points=16)
    assert result["target_vertices"] is None
    assert result["ready_for_training"] is True
    assert np.array_equal(result["size"], [8.,7.,6.])



def test_vedo_show_renders_crown_and_margin_interactively(monkeypatch):
    """show=True 所用渲染器应打开交互窗口且可靠关闭。"""
    import sys
    import types
    from src.infer import show_mesh

    calls = {"actors": [], "closed": False, "interactive": False}
    class Mesh:
        def __init__(self, data):
            self.data = data
        def c(self, color):
            self.color = color
            return self
        def alpha(self, opacity):
            return self
    class Line:
        def __init__(self, points):
            self.points = points
        def c(self, color):
            self.color = color
            return self
        def lw(self, width):
            return self
    class Plotter:
        def __init__(self, *, offscreen, size, bg):
            assert offscreen is False
        def show(self, *actors, axes, interactive, resetcam):
            calls["actors"] = actors
            calls["interactive"] = interactive
            assert axes == 1 and resetcam is True
        def close(self):
            calls["closed"] = True

    monkeypatch.setitem(
        sys.modules, "vedo", types.SimpleNamespace(
            Mesh=Mesh, Line=Line, Plotter=Plotter,
        ),
    )
    crown = np.array([
        [0,0,0], [1,0,0], [0,1,0],
    ], np.float32)
    faces = np.array([[0,1,2]], np.int64)
    show_mesh(crown, faces, case_data={
        "margin_line": [[0,0,0],[1,0,0],[0,1,0]],
    }, show_context=False, show_opposite=False)
    assert calls["interactive"] is True and calls["closed"] is True
    assert len(calls["actors"]) == 2
    assert isinstance(calls["actors"][0], Mesh)
    assert isinstance(calls["actors"][1], Line)
    assert np.array_equal(calls["actors"][1].points[0], calls["actors"][1].points[-1])


def test_vedo_show_uses_only_observed_jaws(monkeypatch):
    """显示的参考牙弓应排除输入中误含的成品冠。"""
    import sys
    import types
    from src import infer as inference

    actors = []
    class Actor:
        def __init__(self, data): self.data = data
        def c(self, color): return self
        def alpha(self, opacity): return self
        def lw(self, width): return self
    class Plotter:
        def __init__(self, **kwargs): pass
        def show(self, *meshes, **kwargs): actors.extend(meshes)
        def close(self): pass
    monkeypatch.setitem(sys.modules, "vedo", types.SimpleNamespace(
        Mesh=Actor, Line=Actor, Plotter=Plotter,
    ))
    class FakeCase:
        def __init__(self, data, *, inference):
            assert inference is True
            self.scene_vertices = np.ones((3,3),np.float32)
            self.scene_faces = np.array([[0,1,2]],np.int64)
            self.opposite_vertices = np.ones((3,3),np.float32)*2
            self.opposite_faces = np.array([[0,1,2]],np.int64)
    monkeypatch.setattr(inference, "DentalCase", FakeCase)
    result = np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    inference.show_mesh(result,np.array([[0,1,2]],np.int64),case_data={
        "margin_line": [[0,0,0],[1,0,0],[0,1,0]],
    })
    assert len(actors) == 4  # 同颌+对颌+生成冠+颈缘线
    assert np.array_equal(actors[0].data[0], np.ones((3,3), np.float32))
    assert np.array_equal(actors[1].data[0], np.ones((3,3), np.float32)*2)
    assert np.array_equal(actors[2].data[0], result)



def test_unknown_size_collate_uses_mask_not_fake_dimensions():
    """用户未输入尺寸时，应显式 size_known=False，不能填入所谓真实冠值。"""
    from process import collate_conditions
    record = {
        "ready_for_training":True,
        "jaw":0,"target_position":6,
        "margin_line":np.zeros((128,3),np.float32),
        "size":None,
        "view_features":np.ones((3,4,8),np.float32),
        "jaw_points":np.zeros((0,6),np.float32),
        "jaw_labels":np.zeros(0,np.int64),
        "jaw_present":False,
    }
    batch = collate_conditions([record],num_points=4)
    assert batch["size"].shape == (1,3)
    assert not batch["size_known"][0]
    assert np.array_equal(batch["size"].numpy(), np.zeros((1,3)))



def test_design_crown_cached_size_is_not_used_as_optional_user_size(tmp_path):
    """同文件含设计成品冠时，顶层尺寸可能是 GT 标签；默认不允许读取。"""
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({
        "current_mesh": {},
        "design_mesh": {"fdi": 16, "vertices": [[10,10,10]]},
        "size_mm": [88, 88, 88],
        "margin_line": [[0,0,0],[1,0,0],[0,1,0]],
        "local_frame": np.eye(3).tolist(),
    }), encoding="utf-8")
    from src.infer import get_case
    case = get_case(path, target_fdi=16)
    assert case.get("size_mm") is None
    override = get_case(path, target_fdi=16, size_mm=(9.,8.,7.))
    assert np.array_equal(override["size_mm"], [9.,8.,7.])


def test_overriding_fdi_or_margin_invalidates_cached_view(tmp_path):
    """修改牙位或颈缘线后，不得复用旧的视图特征。"""
    data = {
        "current_mesh": {},
        "target_fdi": 16,
        "local_frame": np.eye(3).tolist(),
        "margin_line": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "view_features": [[[1]]] * 3,
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "view_features" in get_case(path)
    assert "view_features" not in get_case(path, target_fdi=15)
    assert "view_features" not in get_case(
        path, margin_line=np.array(data["margin_line"], np.float32) + 1,
    )


def test_design_mesh_cannot_supply_fdi_when_not_explicit(tmp_path):
    """禁止从真实设计冠中推断目标牙位。"""
    path = tmp_path / "case.json"
    data = {
        "current_mesh": {},
        "design_mesh": {"fdi": 16, "size_mm": [8, 7, 6]},
        "margin_line": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "local_frame": np.eye(3).tolist(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="target_fdi"):
        get_case(path)


def test_unlabeled_scan_with_provided_frame_retains_points():
    from process import DentalCase
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], np.float32)
    case = DentalCase({
        "current_mesh":{"vertices":vertices,"faces":np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])},
        "target_fdi":16,"margin_line":vertices[:3],
        "local_frame":np.eye(3),"view_features":np.ones((3,4,16),np.float32),
    }, inference=True)
    record = case.get_condition(crop_mode="full", tooth_dropout=0, num_points=8)
    assert not case.labels_known
    assert np.array_equal(record["view_mask"], [False, False, True])
    assert record["jaw_present"]
    assert np.all(record["jaw_labels"] == 17)
    assert np.allclose(get_world_vertices(record["jaw_points"][:, :3], case.world_to_local_transform), vertices)
