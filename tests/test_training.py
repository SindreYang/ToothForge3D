"""网络、几何 Loss、原生 TRELLIS Flow 委托与梯度回归测试。

不下载模型权重，使用小张量和 PTv3 输出契约模拟器测试结构逻辑。
"""

import pytest
import torch
from torch import nn

pytest.importorskip("sindre.ai.losses")
pytest.importorskip("sindre.ai.embedder")

from mcdg import (
    MCDGConditionEncoder, CrownImplicitDecoder, MCDGReconstructionModel,
    TrellisFlowAdapter,
)
from losses import CrownReconstructionLoss, FlowMatchingLoss


class FakePTv3(nn.Module):
    def forward(self, x, p):
        assert x.shape[-1] == 6
        assert p.shape == x.shape[:2] + (3,)
        return x.reshape(-1, 6)


def get_batch():
    torch.manual_seed(123)
    b, q = 2, 20
    return {
        "view_features": torch.randn(b, 3, 7, 24),
        "jaw": torch.tensor([0, 1]),
        "target_position": torch.tensor([6, 14]),
        "margin_line": torch.randn(b, 8, 3),
        "size": torch.ones(b, 3) * 8,
        "jaw_points": torch.randn(b, 9, 6),
        "jaw_labels": torch.tensor([[0, 2, 2, 4, 4, 6, 6, 9, 9], [0] * 9]),
        "jaw_present": torch.tensor([True, False]),
        "query_points": torch.randn(b, q, 3),
        "query_region": (torch.arange(q) % 4).unsqueeze(0).repeat(b, 1),
        "occupancy": (torch.arange(q) % 2).float().unsqueeze(0).repeat(b, 1),
        "sdf_mm": torch.linspace(-0.9, 0.9, q).unsqueeze(0).repeat(b, 1),
    }


def get_model():
    encoder = MCDGConditionEncoder(
        image_dim=24, hidden_dim=32, point_dim=6,
        num_heads=4, frequencies=2,
        margin_points=8, view_tokens=2, jaw_tokens=4,
        point_encoder=FakePTv3(),
    )
    decoder = CrownImplicitDecoder(
        hidden_dim=32, num_heads=4,
        query_frequencies=2, query_chunk_size=6,
    )
    return MCDGReconstructionModel(encoder, decoder)


def test_implicit_decoder_and_losses_backpropagate():
    batch = get_batch()
    model = get_model()
    result = model(batch)
    assert result["occupancy_logits"].shape == (2, 20)
    assert result["sdf_mm"].shape == (2, 20)
    criterion = CrownReconstructionLoss()
    terms = criterion(result, batch)
    assert all(torch.isfinite(v).all() for v in terms.values())
    assert 0 <= terms["iou"].item() <= 1
    terms["loss"].backward()
    assert model.query_decoder.output_projection.weight.grad is not None
    assert model.condition_encoder.view_projection.c_fc.weight.grad is not None


def test_no_reference_view_features_is_a_clear_error():
    batch = get_batch()
    del batch["view_features"]
    with pytest.raises(ValueError, match="DINOv3"):
        get_model()(batch)


def test_losses_reject_wrong_label_shapes_and_types():
    criterion = CrownReconstructionLoss()
    logits = torch.randn(1, 8)
    outputs = {"occupancy_logits": logits, "sdf_mm": torch.zeros_like(logits)}
    batch = {
        "occupancy": torch.zeros(1, 8),
        "query_region": torch.arange(8).reshape(1, 8) % 4,
        "sdf_mm": torch.zeros(1, 8),
    }
    assert torch.isfinite(criterion(outputs, batch)["loss"])
    batch["occupancy"][0, 0] = 2
    with pytest.raises(ValueError, match="0/1"):
        criterion(outputs, batch)


class OfficialFlowStub(nn.Module):
    """只实现官方 flow_model(x, t, cond) 公共契约。"""

    def __init__(self):
        super().__init__()
        self.cond_channels = 12
        self.calls = []

    def forward(self, x, t, cond):
        self.calls.append((x.shape, t.shape, cond.shape))
        return x + cond.mean() * 0


def test_trellis_adapter_calls_native_flow_with_variable_context():
    batch = get_batch()
    conditions = get_model().condition_encoder(
        **{key: batch[key] for key in (
            "view_features", "jaw", "target_position", "margin_line", "size",
            "jaw_points", "jaw_labels", "jaw_present",
        )}
    )
    flow = OfficialFlowStub()
    adapter = TrellisFlowAdapter(flow, condition_dim=32)
    x0 = torch.randn(2, 1, 4, 4, 4)
    out = adapter(x0, torch.ones(2) * 500, conditions)
    assert out.shape == x0.shape
    # 部分病例无同颌点 token，变长 Dense Flow 应分样本调用。
    assert len(flow.calls) == 2
    assert flow.calls[0][2][1] > flow.calls[1][2][1]


def test_flow_matching_matches_official_trellis_formula():
    class CapturingFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.5))
            self.input_seen = None
            self.t_seen = None
        def forward(self, x, t, conditions):
            self.input_seen, self.t_seen = x, t
            return x * self.scale

    flow = CapturingFlow()
    criterion = FlowMatchingLoss(sigma_min=1e-5)
    clean = torch.ones((1, 1, 2, 2, 2))
    noise = torch.ones_like(clean) * 0.25
    t = torch.tensor([0.4])
    terms = criterion(flow, clean, {}, t=t, noise=noise)
    expected_xt = (1 - 0.4) * clean + (1e-5 + (1 - 1e-5) * 0.4) * noise
    expected_v = (1 - 1e-5) * noise - clean
    assert torch.allclose(flow.input_seen, expected_xt)
    assert torch.allclose(flow.t_seen, t * 1000)
    assert torch.allclose(terms["loss"], torch.square(expected_xt * 0.5 - expected_v).mean())
    terms["loss"].backward()
    assert flow.scale.grad is not None


def test_flow_rejects_using_occupancy_points_as_latent():
    with pytest.raises(ValueError, match="VAE|latent|Occupancy"):
        FlowMatchingLoss()(OfficialFlowStub(), torch.randn(2, 128, 3), {})



def test_cpu_training_loop_tqdm_validation_and_checkpoints(monkeypatch, tmp_path):
    """不用 GPU/PTv3 权重，模拟新 LMDB 检查 train/val/保存流程。"""
    from torch.utils.data import Dataset
    import train as trainer

    class ToyDataset(Dataset):
        def __init__(self, db_path, **kwargs):
            self.augment = kwargs.get("augment", False)
        def __len__(self):
            return 6
        def __getitem__(self, index):
            queries = torch.randn(8, 3)
            return {
                "jaw": torch.tensor(index % 2),
                "view_features": torch.ones(3, 4, 24),
                "margin_line": torch.zeros(8, 3),
                "query_points": queries,
                "occupancy": (torch.arange(8) % 2).float(),
                "sdf_mm": torch.linspace(-0.4, 0.4, 8),
                "query_region": torch.arange(8) % 4,
            }
        @staticmethod
        def collate(batch):
            return {
                key: torch.stack([item[key] for item in batch])
                for key in batch[0]
            }
        def close(self):
            pass

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = nn.Parameter(torch.tensor(0.0))
        def forward(self, batch):
            target = batch["occupancy"]
            return {
                "occupancy_logits": torch.ones_like(target) * self.logit,
                "sdf_mm": torch.ones_like(target) * self.logit,
            }

    seen = []
    monkeypatch.setattr(trainer, "MyDataset", ToyDataset)
    monkeypatch.setattr(trainer, "create_model", lambda **kwargs: ToyModel())
    monkeypatch.setattr(trainer, "set_global_seeds", lambda seed: torch.manual_seed(seed))
    monkeypatch.setattr(
        trainer, "save_checkpoint",
        lambda **kwargs: seen.append((kwargs["save_path"], kwargs["curr_iter"])),
    )
    config = trainer.TrainingConfig(
        epochs=1, batch_size=2, num_workers=0,
        train_queries=8, val_queries=8,
        hidden_dim=32, num_heads=4,
        checkpoint_dir=str(tmp_path / "checkpoints"), device="cpu",
    )
    result = trainer.train(tmp_path / "fake.db", config=config)
    assert result["train_loss"] >= 0
    assert result["val_loss"] >= 0
    assert len(seen) == 2
    assert sorted(str(name).split("/")[-1] for name, _ in seen) == ["best.pt", "last.pt"]
    assert all(epoch == 1 for _, epoch in seen)
