"""设计冠视图坐标系的纯几何回归测试，无需 GPU 或 VTK。"""

import numpy as np
import pytest

from views import CrownViews


def get_mesh():
    vertices = np.array([
        [0., 0., 1.], [2., 0., 1.], [2., 1., 1.], [0., 1., 1.],
        [1., 0.5, 3.],
    ])
    faces = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
    ])
    return vertices, faces


def test_explicit_frame_and_dimensions():
    vertices, faces = get_mesh()
    frame = CrownViews(vertices, faces, local_frame=np.eye(3))
    assert np.isclose(np.linalg.det(frame.frame), 1.0)
    assert frame.anatomical_verified
    assert frame.frame_source == "provided"
    assert frame.get_local_vertices().shape == (5, 3)
    assert frame.get_size().tolist() == pytest.approx([2., 1., 1.6])


def test_invalid_frame_rejected():
    vertices, faces = get_mesh()
    with pytest.raises(ValueError, match="正交"):
        CrownViews(vertices, faces, local_frame=np.diag([-1., 1., 1.]))


def test_no_margin_no_frame_refuses_to_claim_occlusal_direction():
    vertices, faces = get_mesh()
    with pytest.raises(ValueError, match="冠底边界"):
        CrownViews(vertices, faces)



def test_camera_looks_at_margin_centre_not_crown_centre():
    vertices, faces = get_mesh()
    margin = vertices[:4]
    view = CrownViews(vertices, faces, margin_line=margin, local_frame=np.eye(3))
    specs = view.get_camera_setup()
    expected_centre = margin.mean(axis=0)
    for name in ("buccal", "lingual", "occlusal"):
        assert np.allclose(specs[name]["focal_point"], expected_centre)
        assert np.isclose(
            np.linalg.norm(specs[name]["position"] - expected_centre),
            specs[name]["distance"],
        )
    assert np.isclose(specs["buccal"]["distance"], specs["lingual"]["distance"])
    assert np.allclose(
        specs["buccal"]["position"] - expected_centre,
        -(specs["lingual"]["position"] - expected_centre),
    )


def test_camera_distance_uses_preparation_base_when_available():
    vertices, faces = get_mesh()
    margin = vertices[:4]
    view = CrownViews(vertices, faces, margin_line=margin, local_frame=np.eye(3))
    margin_only = view.get_camera_setup()["occlusal"]
    base = np.array([[0, 0, -5], [1.5, 0, -5], [0, 1, 0]], dtype=float)
    adjusted = view.get_camera_setup(base_vertices=base)["occlusal"]
    assert adjusted["distance"] > margin_only["distance"]
    assert adjusted["parallel_scale"] > margin_only["parallel_scale"]
    assert np.allclose(adjusted["focal_point"], margin.mean(axis=0))
