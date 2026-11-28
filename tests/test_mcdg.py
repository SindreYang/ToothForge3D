"""条件编码器的 CPU smoke tests，不加载 DINOv3/TRELLIS.2 权重。"""

import pytest
import torch
from torch import nn

pytest.importorskip("sindre.ai.layers")
from src.model import MCDGConditionEncoder


class FakePointEncoder(nn.Module):
    """模拟 SindrePy PTv3 forward(x,p) 的逐点输出契约。"""

    def forward(self, x, p):
        assert p.shape == x.shape[:2] + (3,)
        return x.reshape(-1, x.shape[-1])


def get_inputs(batch_size=2):
    torch.manual_seed(7)
    return dict(
        view_features=torch.randn(batch_size, 3, 7, 24),
        jaw=torch.tensor([0, 1][:batch_size]),
        target_position=torch.tensor([6, 14][:batch_size]),
        margin_line=torch.randn(batch_size, 8, 3),
        size=torch.tensor([[10., 9., 7.]] * batch_size),
    )


def get_model(point_encoder=None):
    return MCDGConditionEncoder(
        image_dim=24, hidden_dim=32, point_dim=6, num_heads=4,
        frequencies=2, margin_points=8, view_tokens=2, jaw_tokens=4,
        point_encoder=point_encoder,
    )


def test_no_jaw_context_is_valid():
    model = get_model()
    conditions = model(**get_inputs())
    assert conditions["tokens"].shape == (2, 20, 32)
    assert conditions["mask"].shape == (2, 20)
    assert torch.all(conditions["mask"][:, :16])
    assert not torch.any(conditions["mask"][:, -4:])
    assert torch.isfinite(conditions["global_condition"]).all()


def test_full_jaw_and_missing_jaw_in_same_batch():
    model = get_model(FakePointEncoder())
    points = torch.randn(2, 9, 6)
    labels = torch.tensor([[2, 2, 4, 4, 5, 5, 0, 0, 0], [0] * 9])
    conditions = model(
        **get_inputs(), jaw_points=points, jaw_labels=labels,
        jaw_present=torch.tensor([True, False]),
    )
    assert torch.all(conditions["mask"][0, -4:])
    assert not torch.any(conditions["mask"][1, -4:])
    assert torch.isfinite(conditions["tokens"]).all()


def test_invalid_tooth_and_missing_labels_fail():
    model = get_model(FakePointEncoder())
    inputs = get_inputs()
    inputs["target_position"] = torch.tensor([0, 17])
    with pytest.raises(ValueError, match="target_position"):
        model(**inputs)
    inputs = get_inputs()
    with pytest.raises(ValueError, match="jaw_labels"):
        model(**inputs, jaw_points=torch.randn(2, 9, 6))


def test_padded_empty_jaw_does_not_require_point_encoder():
    model = get_model()
    conditions = model(
        **get_inputs(),
        jaw_points=torch.zeros(2, 9, 6),
        jaw_labels=torch.zeros(2, 9, dtype=torch.long),
        jaw_present=torch.tensor([False, False]),
    )
    assert not torch.any(conditions["mask"][:, -4:])



def test_real_condition_encoder_supports_no_user_dimensions():
    """未填写 MD/BL/H 时，尺寸预测应依赖 FDI/颈缘线和邻牙等条件。"""
    model = get_model().eval()
    inputs = get_inputs()
    inputs.pop("size")
    with torch.no_grad():
        out = model(**inputs)
    assert out["predicted_size"].shape == (2,3)
    assert torch.all(out["predicted_size"] > 0)
    assert torch.allclose(out["effective_size"],out["predicted_size"])
    assert not out["size_known"].any()


def test_masked_gt_size_cannot_leak_into_condition_encoder():
    """即使训练标签传入 size，size_known=False 时条件输出也必须一致。"""
    model = get_model().eval()
    inputs = get_inputs()
    known = torch.zeros(2,dtype=torch.bool)
    with torch.no_grad():
        first = model(**inputs,size_known=known)
        changed = dict(inputs)
        changed["size"] = inputs["size"] * 10.
        second = model(**changed,size_known=known)
    assert torch.allclose(first["global_condition"],second["global_condition"],atol=1e-6)
    assert torch.allclose(first["effective_size"],second["effective_size"])


def test_masked_view_and_unknown_teeth_are_valid_conditions():
    model = get_model(FakePointEncoder()).eval()
    inputs = get_inputs()
    inputs["view_mask"] = torch.tensor([[False, False, True], [True, True, True]])
    inputs["view_scale_mm"] = torch.tensor([[0., 0., 20.], [22., 22., 20.]])
    inputs["jaw_points"] = torch.randn(2, 9, 6)
    inputs["jaw_labels"] = torch.full((2, 9), 17, dtype=torch.long)
    with torch.no_grad():
        before = model(**inputs)
        inputs["view_features"][0, :2] = 1000.
        inputs["view_scale_mm"][0, :2] = 1000.
        after = model(**inputs)
    assert torch.allclose(before["global_condition"], after["global_condition"], atol=1e-6)
    assert not before["mask"][0, :4].any()
    assert before["mask"][0, 4:6].all()
    assert torch.isfinite(before["global_condition"]).all()
