"""真实牙冠病例字段与随机牙弓裁剪的回归测试。"""

import random

import numpy as np
import pytest

pytest.importorskip("sindre.utils3d.core.dental.utils")
from process import DentalCase


def get_case():
    return {
        "is_upper": False,
        "local_frame": np.eye(3, dtype=np.float32),
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
        "opposite_mesh": {
            "vertices": [[0, 0, -5], [1, 0, -5], [0, 1, -5]],
            "faces": [[0, 1, 2]],
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


def test_dental_case_normalizes_lmdb_scalar_arrays():
    data = get_case()
    data["is_upper"] = np.asarray(data["is_upper"])
    data["design_mesh"]["fdi"] = np.asarray(data["design_mesh"]["fdi"])

    case = DentalCase(data)

    assert case.target_fdi == 31
    assert case.jaw == 1


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
    with pytest.raises(ValueError, match="上下颌不一致"):
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


def test_jaw_is_inferred_from_design_fdi_without_is_upper():
    data = get_case()
    del data["is_upper"]
    case = DentalCase(data)
    assert case.jaw == 1
    assert case.opposite_jaw == 0
    assert case.target_position == 1
    assert case.frame_source == "provided"


def test_size_is_derived_from_design_crown_when_missing():
    data = get_case()
    del data["size_mm"]
    case = DentalCase(data)
    assert case.size.shape == (3,)
    assert (case.size > 0).all()
    assert case.get_condition(crop_mode="none")["geometry_ready"] is True


def test_images_render_current_jaw_without_design_crown():
    data = get_case()
    case = DentalCase(data)

    class StubEncoder:
        def encode(self, image, return_tokens=False):
            assert return_tokens
            assert image.ndim == 3
            return np.ones((5, 6), dtype=np.float32)

    # 只替换离屏渲染，保留真实的相机参数计算。
    # 断言送进相机的几何来自 current_mesh，而非 design_mesh。
    observed = {}
    case._views_builder.get_cut_base = lambda *args: None

    def fake_get_images(
        image_size, *, jaw_vertices, jaw_faces,
        opposite_vertices, opposite_faces, base_vertices,
    ):
        observed["vertices"] = jaw_vertices.copy()
        observed["faces"] = jaw_faces.copy()
        observed["opposite_vertices"] = opposite_vertices.copy()
        observed["opposite_faces"] = opposite_faces.copy()
        assert base_vertices is None
        return np.ones((3, image_size, image_size, 3), dtype=np.uint8)

    case._views_builder.get_images = fake_get_images
    record = case.get_condition(
        crop_mode="none", image_encoder=StubEncoder(), image_size=64
    )
    assert record["view_features"].shape == (3, 5, 6)
    assert record["view_source"] == "both_jaws_lateral_target_jaw_occlusal"
    assert record["camera_info"] is not None
    assert observed["vertices"].shape[0] == 6
    assert observed["faces"].shape[0] == 2
    assert observed["opposite_vertices"].shape[0] == 3
    assert observed["opposite_faces"].shape[0] == 1
    assert len(observed["vertices"]) != len(case.design_vertices)
    assert record["ready_for_training"] is True


def test_local_frame_is_shared_between_target_and_context():
    case = DentalCase(get_case())
    record = case.get_condition(crop_mode="full", tooth_dropout=0.0)
    expected = (
        (case.design_vertices - record["origin_world"]) @ record["local_frame"]
    )
    assert np.allclose(record["target_vertices"], expected, atol=1e-5)
    from sindre.utils3d.sindrepcd import SindrePCD

    restored = SindrePCD(record["target_vertices"]).apply_inv_transform(
        record["world_to_local_transform"]
    )
    assert np.allclose(restored.points, case.design_vertices, atol=1e-5)
    assert np.isclose(np.linalg.det(record["local_frame"]), 1.0, atol=1e-3)



def test_labeled_preparation_base_kept_only_for_scene():
    data = get_case()
    data["current_mesh"]["vertices"] += [[0, 0, -1], [0.5, 0, -1], [0, 0.5, -1]]
    data["current_mesh"]["faces"].append([6, 7, 8])
    data["current_mesh"]["vertex_labels"] += [3, 3, 3]
    case = DentalCase(data)
    assert case.base_label == 2031
    assert len(case.base_vertex_ids) == 3
    assert len(case.scene_faces) == 3
    assert len(case.context_vertex_ids) == 6  # 几何基座不伪装为普通邻牙
    assert not np.any(case.fdi_labels[case.context_vertex_ids] == 2031)
    assert case.get_condition(crop_mode="none")["base_present"] is True


def test_target_crown_in_current_mesh_is_not_rendered():
    data = get_case()
    data["current_mesh"]["vertices"] += [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    data["current_mesh"]["faces"].append([6, 7, 8])
    # 顶点直接用 FDI，此处不再用表索引。
    data["current_mesh"]["vertex_labels"] = [32, 32, 32, 41, 41, 41, 31, 31, 31]
    case = DentalCase(data)
    assert len(case.scene_faces) == 2
    assert len(case.context_vertex_ids) == 6



def test_missing_opposing_arch_masks_lateral_views():
    data = get_case()
    data.pop("opposite_mesh")
    data["view_features"] = np.ones((3, 7, 24), dtype=np.float32)
    case = DentalCase(data)
    record = case.get_condition(crop_mode="none")
    assert record["view_mask"].tolist() == [False, False, True]
    assert not np.any(record["view_features"][:2])
    assert np.all(record["view_features"][2] == 1)



def test_opposing_mesh_not_used_as_point_encoder_context():
    data = get_case()
    case = DentalCase(data)
    record = case.get_condition(crop_mode="full", tooth_dropout=0.0)
    assert record["opposite_present"] is True
    assert len(record["jaw_points"]) == 6
    assert not np.any(np.isclose(record["jaw_points"][:, 2], -5))
    assert record["view_images"] is None


def test_legacy_design_alias_is_rejected():
    """训练只接受 design_mesh，不再回退到 design/desig 旧字段。"""
    data = get_case()
    data["design"] = data.pop("design_mesh")
    with pytest.raises(ValueError, match="design_mesh"):
        DentalCase(data)


def test_identical_raw_margin_training_and_inference_frame():
    data = get_case()
    data["margin_line"] = np.array([[0.,0.,1.],[2.,0.,1.],[1.7,1.,1.2],[0.,1.,1.]])
    case = DentalCase(data)
    other = dict(data)
    other.pop("design_mesh")
    other.pop("size_mm")
    other["target_fdi"] = case.target_fdi
    other["margin_line"] = case.margin_source_points
    inference = DentalCase(other, inference=True)
    assert np.array_equal(case.margin_line, inference.margin_line)
    assert np.allclose(case.world_to_local_transform, inference.world_to_local_transform)
