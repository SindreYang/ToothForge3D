"""真实牙冠病例字段与随机牙弓裁剪的回归测试。"""

import random

import numpy as np
import pytest

pytest.importorskip("sindre.utils3d.dental_algorithms.dental_tools")
from process import DentalCase


def get_case():
    return {
        "is_upper": False,
        "current_mesh": {
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [2, 0, 0], [3, 0, 0], [2, 1, 0]],
            "faces": [[0, 1, 2], [3, 4, 5]],
            # 顶点值是 fdi 字典索引，不是实际 FDI。
            "vertex_labels": [1, 1, 1, 2, 2, 2],
            "fdi": [0, 32, 41, 2031],
        },
        "design_mesh": {
            "fdi": "31",
            "vertices": [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0.5, 0.5, 2]],
            "faces": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        },
        "size_mm": [7.2, 6.0, 8.3],
    }


def test_fdi_mapping_and_open_margin():
    case = DentalCase(get_case())
    assert case.jaw == 1
    assert case.target_position == 1  # SindrePy: 31→1
    assert np.array_equal(np.unique(case.context_labels), [2, 9])
    assert case.margin_line is not None
    assert case.margin_line.shape == (128, 3)
    assert case.get_size().tolist() == pytest.approx([7.2, 6.0, 8.3])


def test_missing_jaw_is_not_fake_points():
    case = DentalCase(get_case())
    sample = case.get_condition(
        num_points=10, crop_mode="full", tooth_dropout=1.0, rng=random.Random(7),
    )
    assert sample["jaw_present"] is False
    assert sample["jaw_points"].shape == (0, 6)
    assert sample["jaw_labels"].shape == (0,)
    assert sample["geometry_ready"] is True
    assert sample["ready_for_training"] is False


def test_closed_design_mesh_needs_real_margin():
    case = get_case()
    case["design_mesh"]["faces"].append([0, 3, 2])
    case["design_mesh"]["faces"].append([0, 2, 1])
    sample = DentalCase(case)
    assert sample.margin_line is None
    assert sample.get_condition(crop_mode="none")["geometry_ready"] is False


def test_invalid_target_jaw_raises():
    case = get_case()
    case["design_mesh"]["fdi"] = "11"
    with pytest.raises(ValueError, match="当前颌"):
        DentalCase(case)


def test_collate_missing_jaw_batch():
    from process import collate_conditions

    data = get_case()
    data["view_features"] = np.ones((3, 7, 24), dtype=np.float32).tolist()
    case = DentalCase(data)
    record = case.get_condition(crop_mode="none")
    assert record["ready_for_training"] is True
    batch = collate_conditions([record], num_points=8)
    assert batch["view_features"].shape == (1, 3, 7, 24)
    assert batch["jaw_points"].shape == (1, 8, 6)
    assert batch["jaw_present"].tolist() == [False]


def test_standalone_mesh_is_not_complete_case():
    from process import get_reference_mesh, process

    mesh = {"type": "mesh",
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "faces": [[0, 1, 2]],
            "colors": None}
    loaded = get_reference_mesh(mesh)
    assert loaded["vertices"].shape == (3, 3)
    with pytest.raises(ValueError, match="独立牙冠 Mesh"):
        process(mesh)
