import torch
import torch.nn as nn
from mamba import PurePyTorchMambaBlock
from config import DrishtiConfig

class DrishtiMamba(nn.Module):
    def __init__(self, cfg: DrishtiConfig, target_dim: int = None):
        super().__init__()
        self.cfg = cfg
        self.target_dim = target_dim if target_dim is not None else cfg.target_dim

        self.in_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.layers = nn.ModuleList([
            PurePyTorchMambaBlock(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv
            ) for _ in range(cfg.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 2, self.target_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))

        last_state = h[:, -1, :]
        out = self.head(last_state)
        return out

if __name__ == "__main__":
    cfg = DrishtiConfig(d_model=64, n_layers=2)
    dev = cfg.setup_runtime()
    model = DrishtiMamba(cfg).to(dev)
    dummy_x = torch.randn(4, cfg.seq_len, cfg.feature_dim, device=dev)
    pred = model(dummy_x)
    assert pred.shape == (4, cfg.target_dim)
    print(f"DrishtiMamba model self-check passed on {dev.type}.")
