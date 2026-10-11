from typing import List, Optional
import numpy as np
from app.config import settings
from app.models.schemas import GatingDecision, DetectedObject
from app.inference.confidence import ConfidenceEngine

class ConfidenceGatingModule:
    """
    Decides between Mamba and Transformer predictions at runtime using
    calibrated confidence scores, agreement metrics, and safety-first policies.
    """
    def __init__(self):
        self.confidence_engine = ConfidenceEngine()

    def select(self,
               mamba_preds: Optional[List[float]],
               mamba_conf: Optional[float],
               trans_preds: Optional[List[float]],
               trans_conf: Optional[float],
               detected_objects: List[DetectedObject]) -> GatingDecision:
        """
        Evaluates both models and selects the safest, most reliable prediction.
        Gracefully handles degraded single-model operation if one fails.
        """
        # 1. Degraded Mode Handling
        if mamba_preds is None and trans_preds is not None:
            return GatingDecision(
                selected_model="transformer",
                prediction=trans_preds,
                confidence=trans_conf or 0.60,
                agreement_score=0.50,
                is_uncertain=False,
                reason="Degraded mode: Mamba unavailable, Transformer selected"
            )
        if trans_preds is None and mamba_preds is not None:
            return GatingDecision(
                selected_model="mamba",
                prediction=mamba_preds,
                confidence=mamba_conf or 0.60,
                agreement_score=0.50,
                is_uncertain=False,
                reason="Degraded mode: Transformer unavailable, Mamba selected"
            )
        if mamba_preds is None and trans_preds is None:
            return GatingDecision(
                selected_model="conservative_fallback",
                prediction=[0.0, 0.0, 0.0, 0.0],
                confidence=0.30,
                agreement_score=0.0,
                is_uncertain=True,
                reason="Fallback: Both sequence models unavailable"
            )

        # 2. Inter-model Agreement & Calibration
        agreement, is_uncertain = self.confidence_engine.calculate_agreement(mamba_preds, trans_preds)
        calib_mamba, calib_trans = self.confidence_engine.blend_confidences(mamba_conf, trans_conf, agreement)

        # 3. Score weighting combining confidence and historical validation performance
        score_mamba = calib_mamba * settings.HISTORICAL_MAMBA_WEIGHT
        score_trans = calib_trans * settings.HISTORICAL_TRANSFORMER_WEIGHT

        # 4. Safety-First Policy: Check for obstacle hazards in direct path
        has_critical_obstacle = any(o.is_in_path and o.distance_m < settings.CRITICAL_DISTANCE_METERS for o in detected_objects)

        # High consensus situation
        if agreement >= 0.85:
            if score_mamba >= score_trans:
                selected = "mamba"
                final_preds = mamba_preds
                final_conf = calib_mamba
            else:
                selected = "transformer"
                final_preds = trans_preds
                final_conf = calib_trans
            reason = f"High model consensus (agreement {agreement:.2f})"
            return GatingDecision(
                selected_model=selected,
                prediction=final_preds,
                confidence=final_conf,
                agreement_score=agreement,
                is_uncertain=False,
                reason=reason
            )

        # Disagreement / Divergence Situation
        if is_uncertain or has_critical_obstacle:
            # Conservative safety choice: Blend predictions with heavier damping to prevent erratic shifts
            blended = [(m + t) / 2.0 for m, t in zip(mamba_preds, trans_preds)]
            selected = "mamba" if score_mamba >= score_trans else "transformer"
            return GatingDecision(
                selected_model=selected,
                prediction=blended,
                confidence=min(calib_mamba, calib_trans),
                agreement_score=agreement,
                is_uncertain=True,
                reason=f"Model divergence detected (agreement {agreement:.2f}); conservative safety blending applied"
            )

        # Standard confident selection
        if score_mamba >= score_trans:
            selected = "mamba"
            final_preds = mamba_preds
            final_conf = calib_mamba
            reason = f"Mamba higher confidence ({calib_mamba:.2f} vs {calib_trans:.2f})"
        else:
            selected = "transformer"
            final_preds = trans_preds
            final_conf = calib_trans
            reason = f"Transformer higher confidence ({calib_trans:.2f} vs {calib_mamba:.2f})"

        return GatingDecision(
            selected_model=selected,
            prediction=final_preds,
            confidence=final_conf,
            agreement_score=agreement,
            is_uncertain=False,
            reason=reason
        )
