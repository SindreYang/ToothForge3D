import importlib
import sys
import types

import numpy as np
import pytest


def test_process_loads_sindre_lmdb_scalars(monkeypatch, tmp_path):
    processing = importlib.import_module("process.process")
    source = tmp_path / "cases.db"
    source.touch()
    records = [
        {
            "is_upper": np.asarray(True),
            "design_mesh": {
                "fdi": np.asarray(16),
                "vertices": np.ones((3, 3), dtype=np.float32),
                "design_info": np.arange(6, dtype=np.float32),
            },
        }
    ]
    observed = {}

    class Reader:
        def __init__(self, path):
            assert path == str(source)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            return records[index]

    class DentalCase:
        def __init__(self, data):
            observed["data"] = data

        def get_condition(self, **kwargs):
            return kwargs

    fake_lmdb = types.ModuleType("sindre.lmdb")
    fake_lmdb.Reader = Reader
    monkeypatch.setitem(sys.modules, "sindre", types.ModuleType("sindre"))
    monkeypatch.setitem(sys.modules, "sindre.lmdb", fake_lmdb)
    monkeypatch.setattr(processing, "Reader", Reader)
    monkeypatch.setattr(processing, "DentalCase", DentalCase)

    result = processing.process(source, sample_index=0, crop_mode="full")
    normalized = observed["data"]
    assert normalized["is_upper"] is True
    assert normalized["design_mesh"]["fdi"] == 16
    assert isinstance(normalized["design_mesh"]["vertices"], np.ndarray)
    assert normalized["design_mesh"]["design_info"].shape == (6,)
    assert result["crop_mode"] == "full"


def test_lmdb_processing_requires_valid_sample_index(tmp_path):
    processing = importlib.import_module("process.process")
    with pytest.raises(ValueError, match="必须指定 sample_index"):
        processing.load_case_data(tmp_path / "cases.db")


@pytest.mark.parametrize(
    ("frame_source", "should_raise"),
    [
        ("design_crown+same_jaw", True),
        ("scan_arch_curve", False),
        ("scan_fdi_anchors", False),
        ("provided", False),
        ("design_crown_pca", True),
    ],
)
def test_training_frame_policy(frame_source, should_raise):
    utils = importlib.import_module("src.utils")
    if should_raise:
        with pytest.raises(ValueError, match="插入轴"):
            utils._validate_training_frame(frame_source)
    else:
        utils._validate_training_frame(frame_source)


def test_build_training_database_forwards_db_paths(monkeypatch, tmp_path):
    processing = importlib.import_module("process.process")
    monkeypatch.setattr(processing, "DEBUG", False)
    observed = {}

    class Config:
        def __init__(self, *, resolution):
            observed["resolution"] = resolution

    class Converter:
        def __init__(self, config):
            pass

    class Preprocessor:
        def __init__(self, converter):
            pass

        def build_database(self, input_db, output_db, **kwargs):
            observed.update(input_db=input_db, output_db=output_db, **kwargs)
            return {"written": 1}

    monkeypatch.setattr(processing, "CrownFDG", Converter)
    monkeypatch.setattr(processing, "FDGPreprocessor", Preprocessor)
    monkeypatch.setattr(processing, "FDGConfig", Config)
    source, output = tmp_path / "raw.db", tmp_path / "train.db"
    encoder = object()

    result = processing.build_training_database(
        source, output, resolution=64, image_encoder=encoder,
    )
    assert result == {"written": 1}
    assert observed == {
        "resolution": 64,
        "input_db": source,
        "output_db": output,
        "image_encoder": encoder,
        "max_cases": None,
        "augment_variants": 1,
        "collision_check": True,
    }


def test_debug_renders_saved_mesh_with_saved_point_cloud(monkeypatch):
    processing = importlib.import_module("process.process")
    row = {
        "target_vertices": np.zeros((3, 3), dtype=np.float32),
        "target_faces": np.array([[0, 1, 2]], dtype=np.int64),
        "margin_line": np.ones((4, 3), dtype=np.float32),
        "jaw_points": np.ones((2, 6), dtype=np.float32),
        "jaw_labels": np.array([3, 4], dtype=np.int64),
        "source_fdi": 16,
        "frame_source": "design_crown+same_jaw",
    }
    observed = {}

    class Reader:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return row

    class SindreMesh:
        def __init__(self, geometry):
            observed["mesh"] = geometry

        def show(self, *, show_append, create_axes):
            observed["actors"] = show_append
            observed["create_axes"] = create_axes

    class SindrePCD:
        def __init__(self, data):
            self.data = data

        def show(self, **kwargs):
            return [self.data]

    sindre = types.ModuleType("sindre")
    sindre.__path__ = []
    utils3d = types.ModuleType("sindre.utils3d")
    utils3d.__path__ = []
    lmdb = types.ModuleType("sindre.lmdb")
    lmdb.Reader = Reader
    mesh_module = types.ModuleType("sindre.utils3d.sindremesh")
    mesh_module.SindreMesh = SindreMesh
    pcd_module = types.ModuleType("sindre.utils3d.sindrepcd")
    pcd_module.SindrePCD = SindrePCD
    monkeypatch.setitem(sys.modules, "sindre", sindre)
    monkeypatch.setitem(sys.modules, "sindre.utils3d", utils3d)
    monkeypatch.setitem(sys.modules, "sindre.lmdb", lmdb)
    monkeypatch.setitem(sys.modules, "sindre.utils3d.sindremesh", mesh_module)
    monkeypatch.setitem(sys.modules, "sindre.utils3d.sindrepcd", pcd_module)
    monkeypatch.setattr(processing, "Reader", Reader)
    monkeypatch.setattr(processing, "SindreMesh", SindreMesh)
    monkeypatch.setattr(processing, "SindrePCD", SindrePCD)

    processing.debug_render_saved_database("training.db")
    assert np.array_equal(observed["mesh"][0], row["target_vertices"])
    assert observed["create_axes"] is True
    assert len(observed["actors"]) == 2
    assert observed["actors"][0]["point_labels"].tolist() == [3, 4]
    assert observed["actors"][1]["points"].shape == (4, 3)


def test_debug_empty_database_reports_incomplete_build(monkeypatch):
    processing = importlib.import_module("process.process")

    class EmptyReader:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return 0

        def get_meta(self, key):
            return False

    monkeypatch.setattr(processing, "Reader", EmptyReader)
    with pytest.raises(ValueError, match="训练数据库没有样本"):
        processing.debug_render_saved_database("incomplete.db")
