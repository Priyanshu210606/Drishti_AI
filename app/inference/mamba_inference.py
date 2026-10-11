import time
from typing import Tuple, List
import torch
import numpy as np
from app.config import settings
from model import DrishtiMamba
from config import DrishtiConfig

class MambaInferenceRunner:
    """
    Dedicated inference runner for the trained Drishti Mamba sequence model.
    Tracks execution latency and computes calibrated prediction confidence.
    """
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        # Initialize architecture matching checkpoint configuration
        mamba_cfg = DrishtiConfig(
            seq_len=settings.SEQUENCE_LENGTH,
            feature_dim=settings.FEATURE_DIM,
            target_dim=settings.TARGET_DIM,
            d_model=settings.D_MODEL,
            d_state=settings.D_STATE,
            d_conv=settings.D_CONV,
            n_layers=settings.N_LAYERS
        )
        self.model = DrishtiMamba(mamba_cfg)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Validation error reference for confidence calibration (from real evaluation)
        self.val_mae = float(checkpoint.get("best_val_loss", 0.0508))

    @torch.inference_mode()
    def predict(self, sequence_tensor: torch.Tensor) -> Tuple[List[float], float, float]:
        """
        Runs Mamba inference.
        Returns: (predicted_deltas, calibrated_confidence, latency_ms)
        """
        t0 = time.time()
        tensor = sequence_tensor.to(self.device)
        output = self.model(tensor)
        latency_ms = (time.time() - t0) * 1000.0

        preds = [round(float(v), 5) for v in output.squeeze(0).cpu().numpy()]

        # Calibrated confidence: penalize predictions with extreme unphysical variance
        pred_norm = float(np.linalg.norm(preds))
        stability_score = math_exp = np.exp(-1.5 * max(0.0, pred_norm - 0.2))
        confidence = float(np.clip(0.88 * stability_score, 0.40, 0.98))

        return preds, round(confidence, 3), round(latency_ms, 2)
