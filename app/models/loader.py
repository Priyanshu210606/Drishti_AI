import os
import torch
from app.config import settings
from app.inference.detector import ObjectDetector
from app.inference.mamba_inference import MambaInferenceRunner
from app.inference.transformer_inference import TransformerInferenceRunner

class ModelRegistry:
    """
    Singleton-style container holding warm instances of all AI models.
    Loaded once at FastAPI application startup.
    """
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.detector: ObjectDetector = None
        self.mamba_runner: MambaInferenceRunner = None
        self.trans_runner: TransformerInferenceRunner = None

    def load_all_models(self):
        print(f"[ModelRegistry] Initializing models on target device: {self.device}")

        # 1. Load Object Detector
        try:
            self.detector = ObjectDetector(self.device)
            print("[ModelRegistry] SSDLite Object Detector loaded.")
        except Exception as e:
            print(f"[ModelRegistry] Failed loading Object Detector: {e}")

        # 2. Load Mamba Runner
        mamba_path = settings.MAMBA_CHECKPOINT_PATH
        if not os.path.exists(mamba_path) and os.path.exists(settings.FALLBACK_CHECKPOINT_PATH):
            mamba_path = settings.FALLBACK_CHECKPOINT_PATH
        try:
            self.mamba_runner = MambaInferenceRunner(mamba_path, self.device)
            print(f"[ModelRegistry] Drishti Mamba loaded from {mamba_path}.")
        except Exception as e:
            print(f"[ModelRegistry] Failed loading Mamba: {e}")

        # 3. Load Transformer Runner
        trans_path = settings.TRANSFORMER_CHECKPOINT_PATH
        if not os.path.exists(trans_path) and os.path.exists(settings.FALLBACK_CHECKPOINT_PATH):
            trans_path = settings.FALLBACK_CHECKPOINT_PATH
        try:
            self.trans_runner = TransformerInferenceRunner(trans_path, self.device)
            print(f"[ModelRegistry] Drishti Transformer loaded from {trans_path}.")
        except Exception as e:
            print(f"[ModelRegistry] Failed loading Transformer: {e}")

        # Warmup forward pass
        self._warmup()

    def _warmup(self):
        dummy_seq = torch.randn(1, settings.SEQUENCE_LENGTH, settings.FEATURE_DIM, device=self.device)
        if self.mamba_runner:
            self.mamba_runner.predict(dummy_seq)
        if self.trans_runner:
            self.trans_runner.predict(dummy_seq)
        print("[ModelRegistry] Model warmup completed. System ready.")

models_registry = ModelRegistry()
