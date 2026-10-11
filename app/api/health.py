from fastapi import APIRouter
from app.config import settings
from app.models.loader import models_registry

router = APIRouter()

@router.get("/health")
def health_check():
    """System diagnostic and model availability health check endpoint."""
    return {
        "status": "ok",
        "device": str(models_registry.device),
        "models_loaded": (models_registry.detector is not None) and (
            models_registry.mamba_runner is not None or models_registry.trans_runner is not None
        ),
        "detector_loaded": models_registry.detector is not None,
        "mamba_loaded": models_registry.mamba_runner is not None,
        "transformer_loaded": models_registry.trans_runner is not None,
        "camera_config": {
            "target_fps": settings.CAMERA_TARGET_FPS,
            "width": settings.CAMERA_WIDTH,
            "height": settings.CAMERA_HEIGHT
        }
    }
