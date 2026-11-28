import sys
import types

import pytest

from process.extract_6 import extract_6, get_target_fdi


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"design_mesh": {"fdi": "16"}}, 16),
        ({"target_fdi": 46}, 46),
    ],
)
def test_get_target_fdi(record, expected):
    assert get_target_fdi(record, 0) == expected


def test_extract_6_copies_only_first_molars(monkeypatch, tmp_path):
    source = tmp_path / "source.db"
    source.touch()
    records = [
        {"design_mesh": {"fdi": 16}, "id": "keep"},
        {"design_mesh": {"fdi": 15}, "id": "skip"},
        {"target_fdi": 46, "id": "keep-lower"},
    ]
    written = []
    metadata = {}

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

        def get_meta(self, key):
            return "raw_cases" if key == "schema_version" else None

    class Writer:
        def __init__(self, path, map_size_limit):
            assert map_size_limit == 1024

        def put_meta(self, key, value):
            metadata[key] = value

        def put_sample(self, record):
            written.append(record)

        def close(self):
            pass

    fake_lmdb = types.ModuleType("sindre.lmdb")
    fake_lmdb.Reader = Reader
    fake_lmdb.Writer = Writer
    monkeypatch.setitem(sys.modules, "sindre.lmdb", fake_lmdb)
    monkeypatch.setitem(sys.modules, "sindre", types.ModuleType("sindre"))

    output = tmp_path / "tooth_6.db"
    assert extract_6(source, output) == 2
    assert [record["id"] for record in written] == ["keep", "keep-lower"]
    assert metadata["schema_version"] == "raw_cases"
    assert metadata["filter_fdi"] == [16, 26, 36, 46]


def test_get_target_fdi_reports_missing_tooth():
    with pytest.raises(ValueError, match="病例 3 缺少"):
        get_target_fdi({}, 3)
