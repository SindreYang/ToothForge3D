"""FDG v2-only 接口契约；纯 AST 检查，无需 GPU/原生依赖。"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _definitions(file_name):
    module = ast.parse((ROOT / file_name).read_text(encoding="utf-8"))
    names = {node.name for node in module.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    methods = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            methods[node.name] = {
                child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return names, methods


def test_obsolete_occupancy_apis_are_deleted():
    names, _ = _definitions("datasets.py")
    assert "FDGDataset" in names
    assert names.isdisjoint({
        "SamplingConfig", "CrownOccupancySampler", "MyDataset",
        "build_database", "main",
    })


def test_fdg_canonical_frame_has_no_legacy_size_argument():
    _, methods = _definitions("utils.py")
    assert "get_normalization" not in methods["CrownFDG"]
    assert "get_frame" not in methods["CrownFDG"]
    module = ast.parse((ROOT / "utils.py").read_text(encoding="utf-8"))
    method = next(node for node in module.body if isinstance(node, ast.ClassDef)
                  and node.name == "CrownFDG")
    canonical = next(node for node in method.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "get_condition_frame")
    assert [arg.arg for arg in canonical.args.args] == ["self", "margin_line"]


def test_no_reference_mesh_compatibility_helper():
    names, _ = _definitions("process.py")
    assert "get_reference_mesh" not in names
    assert "DentalCase" in names
