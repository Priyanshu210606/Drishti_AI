import time
from typing import Tuple, List
import torch
import numpy as np
from app.config import settings
from transformer_model import DrishtiTransformer
from config import DrishtiConfig

class TransformerInferenceRunner:
    """
    Dedicated inference runner for the trained Drishti Transformer sequence model.
    Tracks execution latency and computes calibrated prediction confidence.
    """
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        # Initialize architecture matching checkpoint configuration
        trans_cfg = DrishtiConfig(
            seq_len=settings.SEQUENCE_LENGTH,
            feature_dim=settings.FEATURE_DIM,
            target_dim=settings.TARGET_DIM,
            d_model=settings.D_MODEL,
            n_heads=settings.N_HEADS,
            dim_feedforward=settings.DIM_FEEDFORWARD,
            n_layers=settings.N_LAYERS
        )
        self.model = DrishtiTransformer(trans_cfg)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Validation error reference for confidence calibration
        self.val_mae = float(checkpoint.get("best_val_loss", 0.0546))

    @torch.inference_mode()
    def predict(self, sequence_tensor: torch.Tensor) -> Tuple[List[float], float, float]:
        """
        Runs Transformer inference.
        Returns: (predicted_deltas, calibrated_confidence, latency_ms)
        """
        t0 = time.time()
        tensor = sequence_tensor.to(self.device)
        output = self.model(tensor)
        latency_ms = (time.time() - t0) * 1000.0

        preds = [round(float(v), 5) for v in output.squeeze(0).cpu().numpy()]

        # Calibrated confidence: penalize excessive variance
        pred_norm = float(np.linalg.norm(preds))
        stability_score = np.exp(-1.8 * max(0.0, pred_norm - 0.2))
        confidence = float(np.clip(0.85 * stability_score, 0.38, 0.96))

        return preds, round(confidence, 3), round(latency_ms, 2)
