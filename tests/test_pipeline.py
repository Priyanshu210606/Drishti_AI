import io
import math
import unittest
from PIL import Image
import torch

from app.config import settings
from app.models.schemas import BoundingBox, DetectedObject
from app.inference.temporal_buffer import wrap_angle_rad, TemporalSequenceBuffer
from app.inference.depth import SpatialDepthEstimator
from app.inference.confidence import ConfidenceEngine
from app.inference.gating import ConfidenceGatingModule
from app.navigation.reasoning import SpatialReasoningEngine
from app.navigation.message_generator import NavigationMessageGenerator
from app.navigation.speech_manager import SpeechManager
from app.models.loader import models_registry
from app.inference.pipeline import ContinuousInferencePipeline

class TestDrishtiAIPipeline(unittest.TestCase):
    def test_angle_wrapping(self):
        """Verify angle wrapping properly eliminates +-2pi boundary jumps."""
        # 180 deg + 10 deg = 190 deg (-170 deg in [-pi, pi])
        wrapped = wrap_angle_rad(math.radians(190))
        self.assertAlmostEqual(wrapped, math.radians(-170), places=4)

        # -180 deg - 10 deg = -190 deg (+170 deg in [-pi, pi])
        wrapped_neg = wrap_angle_rad(math.radians(-190))
        self.assertAlmostEqual(wrapped_neg, math.radians(170), places=4)

        # Small deltas stay untouched
        self.assertAlmostEqual(wrap_angle_rad(0.1), 0.1, places=4)

    def test_temporal_sequence_buffer(self):
        """Verify rolling buffer maintains sequence length and pads causally on startup."""
        buf = TemporalSequenceBuffer(seq_len=16, feature_dim=16)
        # Empty buffer produces valid shape
        t0 = buf.get_sequence_tensor(torch.device("cpu"))
        self.assertEqual(t0.shape, (1, 16, 16))

        # Add observations
        dummy_obj = DetectedObject(
            label="person",
            confidence=0.9,
            bbox=BoundingBox(x1=0.4, y1=0.2, x2=0.6, y2=0.8),
            distance_m=2.0,
            horizontal_angle_deg=0.0,
            direction="center",
            is_in_path=True
        )
        for i in range(5):
            buf.add_observation([dummy_obj], estimated_yaw_deg=float(i))

        t_partial = buf.get_sequence_tensor(torch.device("cpu"))
        self.assertEqual(t_partial.shape, (1, 16, 16))

    def test_spatial_depth_estimation(self):
        """Verify metric distance estimation and direction classification."""
        estimator = SpatialDepthEstimator(h_fov_deg=70.0, v_fov_deg=55.0)
        raw_det = [{
            "label": "person",
            "confidence": 0.92,
            "bbox": BoundingBox(x1=0.4, y1=0.2, x2=0.6, y2=0.8),
            "pixel_height": 288.0,
            "pixel_width": 128.0
        }]
        objects, lat = estimator.estimate(raw_det, img_width=640, img_height=480)
        self.assertEqual(len(objects), 1)
        obj = objects[0]
        self.assertGreater(obj.distance_m, 0.5)
        self.assertLess(obj.distance_m, 6.0)
        self.assertEqual(obj.direction, "center")
        self.assertTrue(obj.is_in_path)

    def test_confidence_and_gating(self):
        """Verify confidence agreement and safety-first gating policies."""
        gating = ConfidenceGatingModule()

        # 1. High agreement scenario
        preds_a = [0.01, -0.02, 0.00, 0.01]
        preds_b = [0.01, -0.01, 0.00, 0.02]
        dec_agree = gating.select(preds_a, 0.90, preds_b, 0.88, [])
        self.assertFalse(dec_agree.is_uncertain)
        self.assertGreaterEqual(dec_agree.agreement_score, 0.85)

        # 2. Divergent scenario
        preds_divergent = [0.45, 0.30, -0.20, 0.50]
        dec_diverge = gating.select(preds_a, 0.70, preds_divergent, 0.65, [])
        self.assertTrue(dec_diverge.is_uncertain)

        # 3. Degraded single-model fallback
        dec_fallback = gating.select(None, None, preds_b, 0.85, [])
        self.assertEqual(dec_fallback.selected_model, "transformer")

    def test_navigation_generation_and_anti_spam(self):
        """Verify natural language generator and speech anti-spam manager."""
        generator = NavigationMessageGenerator()
        speech_mgr = SpeechManager(min_interval_sec=2.0, dist_change_thresh_m=0.5)

        close_obj = DetectedObject(
            label="pole",
            confidence=0.88,
            bbox=BoundingBox(x1=0.4, y1=0.1, x2=0.6, y2=0.9),
            distance_m=0.8,
            horizontal_angle_deg=0.0,
            direction="center",
            is_in_path=True
        )

        # Critical hazard
        text_crit, prio_crit = generator.generate("blocked", close_obj, mode="normal")
        self.assertEqual(prio_crit, "CRITICAL")
        self.assertIn("Stop", text_crit)

        # Speech manager should speak critical immediately
        should_speak_1 = speech_mgr.should_speak(text_crit, prio_crit, "blocked", close_obj)
        self.assertTrue(should_speak_1)

        # Immediately repeating non-critical should be blocked by anti-spam
        norm_obj = DetectedObject(
            label="pole",
            confidence=0.88,
            bbox=BoundingBox(x1=0.7, y1=0.2, x2=0.9, y2=0.8),
            distance_m=2.5,
            horizontal_angle_deg=20.0,
            direction="right",
            is_in_path=False
        )
        text_norm, prio_norm = generator.generate("clear", norm_obj, mode="normal")
        should_speak_2 = speech_mgr.should_speak(text_norm, prio_norm, "clear", norm_obj)
        # Should be false because of cooldown from the previous speech
        self.assertFalse(should_speak_2)

    def test_full_pipeline_synthetic_frame(self):
        """End-to-end test of continuous pipeline on synthesized frame bytes."""
        models_registry.load_all_models()
        pipeline = ContinuousInferencePipeline(models_registry)

        # Create dummy RGB image
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_bytes = buf.getvalue()

        response = pipeline.process_frame(frame_bytes)
        self.assertIsNotNone(response)
        self.assertIn(response.path_status, ("clear", "partially_blocked", "blocked", "uncertain"))
        self.assertGreater(response.fps, 0.0)
        self.assertIsNotNone(response.navigation.text)

if __name__ == "__main__":
    unittest.main()
