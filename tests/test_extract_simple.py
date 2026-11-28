import sys
import types

from process.extract_simple import extract_simple


def test_extract_simple_keeps_at_most_five_per_tooth(monkeypatch, tmp_path):
    source = tmp_path / "source.db"
    source.touch()
    records = [
        {"design_mesh": {"fdi": fdi}, "id": index}
        for fdi, count in ((16, 7), (26, 3), (36, 6))
        for index in range(count)
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
            return None

    class Writer:
        def __init__(self, path, map_size_limit):
            assert map_size_limit == 1024 * 1024 * 50

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

    counts = extract_simple(source, tmp_path / "simple.db")
    assert counts == {16: 5, 26: 3, 36: 5}
    assert [record["id"] for record in written] == [
        *range(5), *range(3), *range(5)
    ]
    assert metadata["samples_per_tooth"] == 5
    assert metadata["tooth_counts"] == {"16": 5, "26": 3, "36": 5}
