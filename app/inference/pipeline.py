import io
import time
from typing import Optional, Tuple
from PIL import Image
import torch

from app.config import settings
from app.models.schemas import (
    InferenceResponse, LatencyBreakdown, NavigationMessage,
    ModelPrediction, GatingDecision
)
from app.models.loader import ModelRegistry
from app.inference.depth import SpatialDepthEstimator
from app.inference.temporal_buffer import TemporalSequenceBuffer
from app.inference.gating import ConfidenceGatingModule
from app.navigation.reasoning import SpatialReasoningEngine
from app.navigation.message_generator import NavigationMessageGenerator
from app.navigation.speech_manager import SpeechManager

class ContinuousInferencePipeline:
    """
    Asynchronous continuous inference pipeline.
    Coordinates detector, depth estimator, sequence models, confidence gate,
    and speech manager to process frames and generate actionable navigation.
    """
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.depth_estimator = SpatialDepthEstimator()
        self.temporal_buffer = TemporalSequenceBuffer()
        self.gating = ConfidenceGatingModule()
        self.reasoning = SpatialReasoningEngine()
        self.msg_generator = NavigationMessageGenerator()
        self.speech_manager = SpeechManager()

        self.last_frame_time = time.time()
        self.fps_smoothed = 0.0
        self.speech_mode = settings.DEFAULT_SPEECH_MODE

    def process_frame(self, image_bytes: bytes) -> InferenceResponse:
        t_start = time.time()
        latencies = LatencyBreakdown()

        # 1. Decode JPEG image
        t0 = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        latencies.decode_ms = round((time.time() - t0) * 1000.0, 2)

        # 2. Object Detection
        detections = []
        if self.registry.detector:
            detections, latencies.detection_ms = self.registry.detector.detect(image)
            latencies.detection_ms = round(latencies.detection_ms, 2)

        # 3. Spatial Depth & Direction Estimation
        objects, latencies.depth_ms = self.depth_estimator.estimate(detections, w, h)
        latencies.depth_ms = round(latencies.depth_ms, 2)

        # 4. Temporal Sequence Buffer Update
        self.temporal_buffer.add_observation(objects)
        seq_tensor = self.temporal_buffer.get_sequence_tensor(self.registry.device)

        # 5. Dual Model Inference: Mamba & Transformer
        mamba_preds, mamba_conf = None, None
        trans_preds, trans_conf = None, None

        if self.registry.mamba_runner:
            mamba_preds, mamba_conf, latencies.mamba_ms = self.registry.mamba_runner.predict(seq_tensor)

        if self.registry.trans_runner:
            trans_preds, trans_conf, latencies.transformer_ms = self.registry.trans_runner.predict(seq_tensor)

        # 6. Confidence-Aware Gating & Selection
        t0 = time.time()
        gating_decision: GatingDecision = self.gating.select(
            mamba_preds=mamba_preds,
            mamba_conf=mamba_conf,
            trans_preds=trans_preds,
            trans_conf=trans_conf,
            detected_objects=objects
        )
        latencies.gating_ms = round((time.time() - t0) * 1000.0, 2)

        # 7. Spatial Path Reasoning
        path_status, primary_obstacle = self.reasoning.evaluate_path(objects, gating_decision)

        # 8. Natural Language Generation
        t0 = time.time()
        spoken_text, priority = self.msg_generator.generate(
            path_status=path_status,
            primary=primary_obstacle,
            mode=self.speech_mode,
            total_obstacles=len(objects)
        )

        # 9. Speech Anti-Spam Manager Decision
        should_speak = self.speech_manager.should_speak(
            text=spoken_text,
            priority=priority,
            path_status=path_status,
            primary=primary_obstacle
        )
        latencies.navigation_ms = round((time.time() - t0) * 1000.0, 2)

        # Calculate FPS & Total Latency
        now = time.time()
        latencies.total_ms = round((now - t_start) * 1000.0, 2)
        instant_fps = 1.0 / max(0.001, (now - self.last_frame_time))
        self.fps_smoothed = round(0.85 * self.fps_smoothed + 0.15 * instant_fps, 1)
        self.last_frame_time = now

        nav_msg = NavigationMessage(
            text=spoken_text,
            priority=priority,
            should_speak=should_speak,
            speech_mode=self.speech_mode,
            timestamp=now
        )

        return InferenceResponse(
            timestamp=now,
            fps=self.fps_smoothed,
            latencies=latencies,
            objects=objects,
            primary_obstacle=primary_obstacle,
            path_status=path_status,
            gating=gating_decision,
            navigation=nav_msg
        )
