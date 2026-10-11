import math
from typing import List, Tuple
import numpy as np
from app.config import settings

class ConfidenceEngine:
    """
    Evaluates individual model confidences and inter-model agreement.
    Provides reliability metrics to drive confidence-aware gating.
    """
    @staticmethod
    def calculate_agreement(preds_mamba: List[float], preds_trans: List[float]) -> Tuple[float, bool]:
        """
        Computes normalized agreement score [0.0, 1.0] and uncertainty flag.
        """
        arr_m = np.array(preds_mamba, dtype=float)
        arr_t = np.array(preds_trans, dtype=float)

        # L2 norm Euclidean distance between prediction vectors
        diff = float(np.linalg.norm(arr_m - arr_t))

        # Exponential decay: diff = 0 -> 1.0, diff = 0.5 -> 0.22
        agreement_score = float(np.exp(-3.0 * diff))
        agreement_score = max(0.0, min(1.0, agreement_score))

        # Flag uncertain if divergence exceeds configured threshold
        is_uncertain = (diff > settings.UNCERTAINTY_DISAGREEMENT_THRESHOLD)

        return round(agreement_score, 3), is_uncertain

    @staticmethod
    def blend_confidences(conf_mamba: float, conf_trans: float, agreement_score: float) -> Tuple[float, float]:
        """
        Adjusts model confidence based on agreement consensus.
        High agreement boosts reliability; disagreement lowers confidence.
        """
        boost = (agreement_score - 0.5) * 0.15
        calibrated_mamba = float(np.clip(conf_mamba + boost, 0.20, 0.99))
        calibrated_trans = float(np.clip(conf_trans + boost, 0.20, 0.99))
        return round(calibrated_mamba, 3), round(calibrated_trans, 3)
