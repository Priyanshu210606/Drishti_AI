import os
import torch
from dataclasses import dataclass

@dataclass
class DrishtiConfig:
    data_dir: str = "Data"
    checkpoint_path: str = "checkpoint.pt"
    mamba_checkpoint_path: str = "mamba_checkpoint.pt"
    transformer_checkpoint_path: str = "transformer_checkpoint.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = 4
    max_runs: int = None
    max_frames: int = None
    cache_path: str = "Data/preprocessed_cache.pt"
    use_cache: bool = True
    seq_len: int = 16
    feature_dim: int = 16
    target_dim: int = 4
    depth_stride: int = 8
    depth_clip_meters: float = 10.0
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    n_layers: int = 2
    n_heads: int = 4
    dim_feedforward: int = 256
    batch_size: int = 128
    epochs: int = 100
    samples_per_epoch: int = 5000
    val_samples: int = 1000
    resume: bool = True
    patience: int = 100
    val_split: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4

    def setup_runtime(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                self.device = "cpu"
        dev = torch.device("cpu")
        torch.set_num_threads(self.num_threads)
        return dev

if __name__ == "__main__":
    cfg = DrishtiConfig()
    dev = cfg.setup_runtime()
    assert dev.type in ("cuda", "cpu")
    print(f"DrishtiConfig verified: epochs={cfg.epochs}, patience={cfg.patience}")
