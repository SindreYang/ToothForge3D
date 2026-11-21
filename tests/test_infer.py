"""Path 型推理：不读目标设计冠 Mesh，尺寸使用显式输入，输出还原世界坐标。"""

import json

import numpy as np
import pytest

from infer import get_case, get_world_vertices


def test_json_case_extracts_fdi_but_removes_design_geometry(tmp_path):
    """设计冠只能提供牙位标识，几何不能进入推理条件。"""
    case = {
        "current_mesh": {
            "vertices": [[0,0,0],[2,0,0],[0,2,0],[0,0,2]],
            "faces": [[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
            "vertex_labels": [0,0,0,0],
        },
        "design": {
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
        path, margin_line=margin,
        size_mm=(9.,8.,7.), local_frame=np.eye(3),
    )
    assert input_case["target_fdi"] == 16
    assert np.array_equal(input_case["margin_line"], margin)
    assert np.array_equal(input_case["size_mm"], [9.,8.,7.])
    assert "design" not in input_case
    assert "design_mesh" not in input_case
    assert "merge_info" not in input_case
    assert np.array_equal(input_case["local_frame"], np.eye(3))


def test_missing_explicit_size_does_not_read_design_crown(tmp_path):
    case = {"current_mesh": {"vertices": [[0,0,0]], "faces": []},
            "design": {"fdi":16,"size_mm":[9,8,7]}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(ValueError, match="size_mm"):
        get_case(
            path, target_fdi=16,
            margin_line=np.ones((3,3)), local_frame=np.eye(3),
        )


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
    world = get_world_vertices(local, margin, frame)
    assert np.allclose(world[0], [12,20,30])
    assert np.allclose(world[1], [12,21,30])
    assert np.allclose(world[2], [11,20,30])


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
