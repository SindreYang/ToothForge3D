"""轻量 FDG 真正端到端结构/几何监督：可在 CPU 上运行数学与 Mock 训练检查。"""

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("sindre.ai.losses")
from fdg_model import LightFDGConfig, LightFDGModel
from losses import CrownFDGLoss
from datasets import FDGDataset


class DummyCondition(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(5, hidden_dim)

    def forward(self, **kwargs):
        views = kwargs["view_features"]
        jaw = kwargs["jaw"].float()
        size = kwargs["size"]
        margin = kwargs["margin_line"]
        x = torch.cat(
            (views.mean(dim=(1, 2, 3))[:, None],
             jaw[:, None], size.mean(dim=1)[:, None],
             margin.square().mean(dim=(1, 2))[:, None],
             torch.ones_like(jaw)[:, None]),
            dim=-1,
        )
        return {"global_condition": self.projection(x)}


def get_batch(batch_size=2, res=8):
    torch.manual_seed(42)
    coords = torch.randint(0, res, (batch_size, 40, 3))
    gt = torch.zeros(batch_size, 4, 4, 4)
    for i in range(batch_size):
        positives = coords[i, :20] // 2
        gt[i, positives[:, 0], positives[:, 1], positives[:, 2]] = 1
    active = torch.zeros(batch_size, 40)
    active[:, :20] = 1
    return {
        "jaw": torch.tensor([0, 1][:batch_size]),
        "view_features": torch.randn(batch_size, 3, 5, 8),
        "size": torch.tensor([[7., 8., 6.]]).repeat(batch_size, 1),
        "margin_line": torch.randn(batch_size, 8, 3),
        "fine_coords": coords,
        "coarse_target": gt,
        "fine_active": active,
        "fdg_target_offsets": torch.rand(batch_size, 40, 3),
        "fdg_target_intersected": torch.randint(
            0, 2, (batch_size, 40, 3)
        ).float(),
    }


def get_model():
    return LightFDGModel(
        DummyCondition(),
        LightFDGConfig(
            resolution=8, coarse_resolution=4,
            hidden_dim=16, base_channels=8,
            query_hidden_dim=24, max_coarse_cells=4,
            max_output_cells=30,
        ),
    )


def test_coarse_fine_and_geometry_all_receive_gradients():
    model = get_model()
    batch = get_batch()
    pred = model(batch, batch["fine_coords"])
    assert pred["coarse_logits"].shape == (2, 4, 4, 4)
    assert pred["fine_logits"].shape == (2, 40)
    assert pred["dual_offsets"].shape == (2, 40, 3)
    terms = CrownFDGLoss()(pred, batch)
    assert all(torch.isfinite(v) for v in terms.values())
    terms["loss"].backward()
    assert model.grid_stem.weight.grad is not None
    assert model.local_head[-1].weight.grad is not None
    assert model.condition_encoder.projection.weight.grad is not None
    assert torch.isfinite(model.local_head[-1].weight.grad).all()


def test_inference_not_depends_on_target_fdg_coords():
    model = get_model().eval()
    batch = get_batch()
    # 只传真实病例条件，不提供监督标签、不提供任何 FDG/GT 坐标。
    only_condition = {
        "jaw":batch["jaw"],
        "view_features":batch["view_features"],
        "size":batch["size"],
        "margin_line":batch["margin_line"],
    }
    # 未训练模型结构结果可能为空，但必须能独立完成推理且不泄漏 GT。
    with torch.no_grad():
        predictions = model.predict(only_condition)
    assert len(predictions) == 2
    for record in predictions:
        assert "coords" in record and record["coords"].shape[-1] == 3
        if record["valid"]:
            assert record["coords"].size(0) <= model.config.max_output_cells
            assert record["dual_offsets"].shape == record["coords"].shape
    altered = get_batch()
    altered["fine_coords"][:] = 0
    with torch.no_grad():
        result2 = model.predict(only_condition)
    for a, b in zip(predictions, result2):
        assert torch.equal(a["coords"], b["coords"])


def test_no_positive_structure_is_rejected_not_silently_optimized():
    model = get_model()
    batch = get_batch()
    pred = model(batch, batch["fine_coords"])
    batch["coarse_target"][:] = 0
    with pytest.raises(ValueError, match="正体素"):
        CrownFDGLoss()(pred, batch)


def test_fdg_loss_rejects_invalid_floats():
    model = get_model()
    batch = get_batch()
    pred = model(batch, batch["fine_coords"])
    pred["fine_logits"] = pred["fine_logits"].clone()
    pred["fine_logits"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        CrownFDGLoss()(pred, batch)


def test_inference_uses_its_coarse_predictions():
    model = get_model().eval()
    # 只允许选择一个 coarse 体素，细体素是否保留由 local head 决定。
    cache = {
        "volume": torch.zeros(1, 8, 4, 4, 4),
        "global": torch.zeros(1, 16),
        "coarse_logits": torch.full((1, 4, 4, 4), -20.),
    }
    cache["coarse_logits"][0, 2, 1, 0] = 20
    with torch.no_grad():
        result = model.generate_coords(cache)
    assert len(result) == 1
    if len(result[0]):
        assert (result[0] // 2 == torch.tensor([2, 1, 0])).all()


def test_canonical_frame_uses_only_margin_and_size():
    from utils import CrownFDG
    line = np.array(
        [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]],
        dtype=np.float32,
    )
    convert = CrownFDG()
    center, scale = convert.get_condition_frame(line, np.array([7, 8, 6]))
    assert np.allclose(center, line.mean(0))
    assert scale > 16
    # 没有 target vertices 的参与，重复调用得到同样结果
    assert scale == convert.get_condition_frame(line, np.array([7, 8, 6]))[1]


def test_dataset_validation_is_deterministic(monkeypatch):
    import sys
    import types
    fdg_coords = np.stack([
        np.arange(14) + 4,
        np.arange(14) + 5,
        np.arange(14) + 6,
    ], axis=-1).astype(np.int32)
    row = {
        "fdg_resolution": 32,
        "fdg_coords": fdg_coords,
        "fdg_dual_offsets": np.full((14, 3), 0.4, np.float32),
        "fdg_intersected": np.ones((14, 3), np.uint8),
        "fdg_center_mm": np.zeros(3, np.float32),
        "fdg_scale_mm": 20.,
        "jaw": 0, "target_position": 6,
        "jaw_points": np.zeros((16, 6), np.float32),
        "jaw_labels": np.zeros(16, np.int64),
        "view_features": np.ones((3, 4, 8), np.float32),
        "margin_line": np.zeros((8, 3), np.float32),
        "size": np.ones(3, np.float32) * 8,
    }
    class Reader:
        def __init__(self, path):
            pass
        def get_meta(self, name):
            return {
                "build_complete":True,
                "schema_version":"toothforge3d_fdg_v1",
                "coordinate_frame":"margin_size_v1",
            }.get(name)
        def __getitem__(self, index):
            return row
        def __len__(self):
            return 5
        def close(self):
            pass

    db = types.ModuleType("sindre.lmdb")
    db.Reader = Reader
    monkeypatch.setitem(sys.modules, "sindre.lmdb", db)
    dataset = FDGDataset(
        "fake.db", resolution=32, coarse_resolution=8,
        query_count=16, jaw_points=12, augment=False, seed=7,
    )
    first, second = dataset[1], dataset[1]
    assert torch.equal(first["fine_coords"], second["fine_coords"])
    assert torch.equal(first["fine_active"], second["fine_active"])
    assert first["fine_active"].sum() == 8
    assert first["coarse_target"].sum() > 0
    assert FDGDataset.collate([first, second])["fine_coords"].shape == (2, 16, 3)
    assert "view_images" not in first
    dataset.close()
