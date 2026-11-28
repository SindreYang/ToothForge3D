"""FDG training CPU smoke：训练与验证分开、tqdm、Loss 梯度及 checkpoint 调用。"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("sindre.ai.losses")
from src import train as runner
from src.model import LightFDGConfig


def test_fdg_trainer_one_cpu_epoch_without_cuda_or_render(monkeypatch, tmp_path):
    saved = []

    class FakeData:
        def __init__(self, db, *, augment, **kwargs):
            self.augment = augment
        def __len__(self):
            return 5
        def get_source_groups(self):
            return tuple(range(5))
        def __getitem__(self, idx):
            coarse = torch.zeros(4, 4, 4)
            coarse[1, 1, 1] = 1
            fine_active = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
            return {
                "jaw": torch.tensor(0),
                "margin_line": torch.zeros(8, 3),
                "view_features": torch.ones(3, 4, 8),
                "size": torch.ones(3) * 6,
                "coarse_target": coarse,
                "fine_coords": torch.ones(4, 3, dtype=torch.long),
                "fine_active": fine_active,
                "fdg_target_offsets": torch.ones(4, 3) * 0.5,
                "fdg_target_intersected": torch.ones(4, 3),
                "fdg_coords": torch.ones(2, 3, dtype=torch.long),
                "fdg_intersected": torch.ones(2, 3),
                "fdg_offsets": torch.zeros(2, 3),
                "fdg_center_mm": torch.zeros(3),
                "fdg_scale_mm": torch.tensor(16.),
            }
        @staticmethod
        def collate(rows):
            ragged = ("fdg_coords", "fdg_intersected", "fdg_offsets")
            return {
                key: [r[key] for r in rows] if key in ragged else
                torch.stack([r[key] for r in rows]) for key in rows[0]
            }
        def close(self):
            pass

    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.01))
        def forward(self, batch, coords):
            bs, count = coords.shape[:2]
            logits = self.weight * torch.ones_like(batch["coarse_target"])
            fine = self.weight * torch.ones(bs, count)
            offsets = (self.weight + 0.1) * torch.ones(bs, count, 3)
            intersections = self.weight * torch.ones(bs, count, 3)
            return {
                "coarse_logits":logits, "fine_logits":fine,
                "dual_offsets":offsets, "intersected_logits":intersections,
            }

    monkeypatch.setattr(runner, "FDGDataset", FakeData)
    monkeypatch.setattr(runner, "set_global_seeds", lambda seed: torch.manual_seed(seed))
    monkeypatch.setattr(
        runner, "save_checkpoint",
        lambda **kw: saved.append((kw["save_path"], kw["curr_iter"])),
    )
    config = runner.FDGTrainingConfig(
        epochs=1, batch_size=2, query_count=4,
        num_workers=0, margin_every=0, device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    out = runner.train(
        tmp_path / "fake.db", config=config,
        model_config=LightFDGConfig(
            resolution=8, coarse_resolution=4,
            hidden_dim=16, base_channels=8,
        ), model=ToyNet(),
    )
    assert out["train_loss"] > 0 and out["val_loss"] > 0
    assert 0 <= out["val_coarse_recall"] <= 1
    assert len(saved) == 2
    assert {Path(path).name for path, _ in saved} == {"best.pt", "last.pt"}
    assert all(epoch == 1 for _, epoch in saved)


def test_cpu_margin_requires_explicit_disable():
    with pytest.raises(ValueError, match="CUDA"):
        runner.train(
            "does_not_exist.db",
            config=runner.FDGTrainingConfig(device="cpu", margin_every=1),
        )
