import torch
import torch.nn as nn
from config import DrishtiConfig

class DrishtiTransformer(nn.Module):
    def __init__(self, cfg: DrishtiConfig, target_dim: int = None):
        super().__init__()
        self.cfg = cfg
        self.target_dim = target_dim if target_dim is not None else cfg.target_dim

        self.in_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 2, self.target_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = self.in_proj(x) + self.pos_emb[:, :seq_len, :]

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)

        last_state = h[:, -1, :]
        out = self.head(last_state)
        return out

if __name__ == "__main__":
    cfg = DrishtiConfig()
    dev = cfg.setup_runtime()
    model = DrishtiTransformer(cfg).to(dev)
    dummy_x = torch.randn(4, cfg.seq_len, cfg.feature_dim, device=dev)
    pred = model(dummy_x)
    assert pred.shape == (4, cfg.target_dim)
    print(f"DrishtiTransformer self-check passed on {dev.type}.")
