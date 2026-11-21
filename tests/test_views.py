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
