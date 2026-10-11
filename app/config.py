import os
from typing import Literal
import torch
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    """
    Central application configuration supporting environment variable overrides.
    Governs server, camera streaming, model paths, and navigation parameters.
    """
    # Server & Networking
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["*"]

    # Model Checkpoints (relative to project root)
    MAMBA_CHECKPOINT_PATH: str = "mamba_checkpoint.pt"
    TRANSFORMER_CHECKPOINT_PATH: str = "transformer_checkpoint.pt"
    FALLBACK_CHECKPOINT_PATH: str = "checkpoint.pt"

    # Compute Device (Auto-detect GPU or CPU)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_THREADS: int = 4

    # Temporal Sequence Constraints (must match trained model architecture)
    SEQUENCE_LENGTH: int = 16
    FEATURE_DIM: int = 16
    TARGET_DIM: int = 4
    D_MODEL: int = 128
    D_STATE: int = 16
    D_CONV: int = 4
    N_LAYERS: int = 2
    N_HEADS: int = 4
    DIM_FEEDFORWARD: int = 256

    # Vision & Camera Parameters
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_TARGET_FPS: int = 20
    INFERENCE_TARGET_FPS: int = 12
    JPEG_QUALITY: int = 75
    HORIZONTAL_FOV_DEG: float = 70.0
    VERTICAL_FOV_DEG: float = 55.0

    # Detection & Spatial Reasoning
    DETECTION_CONF_THRESHOLD: float = 0.35
    DEPTH_CLIP_METERS: float = 8.0
    CRITICAL_DISTANCE_METERS: float = 1.0
    CAUTION_DISTANCE_METERS: float = 2.2
    PATH_CORRIDOR_ANGLE_DEG: float = 16.0

    # Confidence Gating & Safety
    CONFIDENCE_AGREEMENT_THRESHOLD: float = 0.15
    UNCERTAINTY_DISAGREEMENT_THRESHOLD: float = 0.40
    HISTORICAL_MAMBA_WEIGHT: float = 0.55
    HISTORICAL_TRANSFORMER_WEIGHT: float = 0.45

    # Speech Management & Anti-Spam
    MIN_SPEECH_INTERVAL_SEC: float = 2.0
    DISTANCE_CHANGE_THRESHOLD_M: float = 0.45
    ANGLE_CHANGE_THRESHOLD_DEG: float = 12.0
    CLEAR_PATH_REMINDER_SEC: float = 8.0
    DEFAULT_SPEECH_MODE: Literal["brief", "normal", "detailed"] = "normal"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = AppSettings()
