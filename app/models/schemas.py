from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Normalized coordinates [0.0, 1.0] for pixel-independent spatial reasoning."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox
    distance_m: float
    horizontal_angle_deg: float
    direction: Literal["far_left", "left", "slightly_left", "center", "slightly_right", "right", "far_right"]
    is_in_path: bool

class ModelPrediction(BaseModel):
    model_name: Literal["mamba", "transformer"]
    prediction: List[float] = Field(..., description="[delta_yaw, delta_pitch, delta_roll, delta_heading] in radians")
    confidence: float
    latency_ms: float

class GatingDecision(BaseModel):
    selected_model: Literal["mamba", "transformer", "conservative_fallback"]
    prediction: List[float]
    confidence: float
    agreement_score: float
    is_uncertain: bool
    reason: str

class NavigationMessage(BaseModel):
    text: str
    priority: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    should_speak: bool
    speech_mode: Literal["brief", "normal", "detailed"]
    timestamp: float

class LatencyBreakdown(BaseModel):
    decode_ms: float = 0.0
    detection_ms: float = 0.0
    depth_ms: float = 0.0
    mamba_ms: float = 0.0
    transformer_ms: float = 0.0
    gating_ms: float = 0.0
    navigation_ms: float = 0.0
    total_ms: float = 0.0

class InferenceResponse(BaseModel):
    type: str = "inference_result"
    timestamp: float
    fps: float
    latencies: LatencyBreakdown
    objects: List[DetectedObject]
    primary_obstacle: Optional[DetectedObject] = None
    path_status: Literal["clear", "partially_blocked", "blocked", "uncertain"]
    gating: GatingDecision
    navigation: NavigationMessage

class ClientConfigUpdate(BaseModel):
    speech_mode: Optional[Literal["brief", "normal", "detailed"]] = None
    min_speech_interval: Optional[float] = None
    critical_distance: Optional[float] = None
