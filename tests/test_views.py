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



def test_vedo_scene_contains_both_jaws_laterally_but_one_occlusally(monkeypatch):
    """模拟 Vedo 离屏接口，只验证三视图的实际场景分配。"""
    import sys
    import types

    image_module = pytest.importorskip("PIL.Image")
    shown = []

    class Actor:
        def compute_normals(self, **kwargs):
            return self
        def c(self, color):
            return self
        def lighting(self, name):
            return self

    class Camera:
        def SetFocalPoint(self, *args):
            pass
        def SetPosition(self, *args):
            pass
        def SetViewUp(self, *args):
            pass
        def ParallelProjectionOn(self):
            pass
        def SetParallelScale(self, *args):
            pass

    class Renderer:
        def ResetCameraClippingRange(self):
            pass

    class Plotter:
        def __init__(self, **kwargs):
            self.camera = Camera()
            self.renderer = Renderer()
        def show(self, *actors, **kwargs):
            shown.append(len(actors))
        def render(self):
            pass
        def screenshot(self, path):
            image_module.new("RGB", (64, 64), "white").save(path)
        def close(self):
            pass

    fake_vedo = types.SimpleNamespace(Mesh=lambda arrays: Actor(), Plotter=Plotter)
    monkeypatch.setitem(sys.modules, "vedo", fake_vedo)

    verts, faces = get_mesh()
    frame = CrownViews(
        verts, faces, margin_line=verts[:4], local_frame=np.eye(3)
    )
    images = frame.get_images(
        64,
        jaw_vertices=verts,
        jaw_faces=faces,
        opposite_vertices=verts + [0, 0, -5],
        opposite_faces=faces,
    )
    assert images.shape == (3, 64, 64, 3)
    assert shown == [2, 2, 1]


def test_meshlib_cut_base_uses_rootward_part(monkeypatch):
    """模拟 SindrePy MeshLib 结果，验证选择根侧而不是按网格面积决定。"""
    import sys
    import types

    verts, faces = get_mesh()
    margin = verts[:4]
    frame = CrownViews(verts, faces, margin_line=margin, local_frame=np.eye(3))
    region = types.SimpleNamespace(vertices=verts, cells=faces)
    class FakeMesh:
        def __init__(self, arrays):
            pass
        def split(self):
            return [region]
    monkeypatch.setitem(sys.modules, "vedo", types.SimpleNamespace(Mesh=FakeMesh))
    # 隔离真实 MeshLib 依赖，但验证调用的确是 cut_mesh_by_meshlib。
    algorithm = types.ModuleType("sindre.utils3d.algorithm")
    seen = {}
    def fake_cut(vertices, faces, loop, *, get_bigger_part, smooth_boundary):
        seen["called"] = True
        seen["loop"] = loop
        assert not get_bigger_part
        assert not smooth_boundary
        root = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]
        ])
        crown = np.array([
            [0., 0., 2.], [1., 0., 2.], [0., 1., 2.]
        ])
        return crown, np.array([[0, 1, 2]]), root, np.array([[0, 1, 2]])
    algorithm.cut_mesh_by_meshlib = fake_cut
    monkeypatch.setitem(sys.modules, "sindre.utils3d.algorithm", algorithm)
    result = frame.get_cut_base(verts, faces)
    assert seen["called"]
    assert np.allclose(seen["loop"], margin)
    assert result is not None
    assert np.allclose(result[:, 2], 0.)
