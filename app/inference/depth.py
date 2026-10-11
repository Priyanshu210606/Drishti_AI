import math
import time
from typing import List, Tuple
from app.config import settings
from app.models.schemas import BoundingBox, DetectedObject

# Approximate real-world heights (meters) for known physical obstacles
OBJECT_HEIGHT_PRIORS = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.25,
    "bus": 3.10,
    "truck": 2.80,
    "traffic light": 2.50,
    "fire hydrant": 0.75,
    "stop sign": 2.20,
    "bench": 0.85,
    "chair": 0.85,
    "couch": 0.90,
    "potted plant": 0.70,
    "dining table": 0.75,
    "tv": 0.60,
    "laptop": 0.30,
    "backpack": 0.50,
    "umbrella": 0.90,
    "bottle": 0.25,
    "dog": 0.55,
    "cat": 0.30
}
DEFAULT_HEIGHT_PRIOR = 0.90  # Default general obstacle height

class SpatialDepthEstimator:
    """
    Computes metric distances and angular directions from camera geometry
    and object bounding boxes without requiring heavy depth neural networks.
    """
    def __init__(self,
                 h_fov_deg: float = settings.HORIZONTAL_FOV_DEG,
                 v_fov_deg: float = settings.VERTICAL_FOV_DEG,
                 camera_height_m: float = 1.30):
        self.h_fov_rad = math.radians(h_fov_deg)
        self.v_fov_rad = math.radians(v_fov_deg)
        self.camera_height_m = camera_height_m

    def estimate(self, raw_detections: List[dict], img_width: int, img_height: int) -> Tuple[List[DetectedObject], float]:
        """
        Enriches detections with estimated distance (meters), horizontal angle, and direction.
        """
        t0 = time.time()
        focal_y = (img_height / 2.0) / math.tan(self.v_fov_rad / 2.0)

        processed_objects: List[DetectedObject] = []
        for det in raw_detections:
            label = det["label"]
            bbox: BoundingBox = det["bbox"]
            conf = det["confidence"]

            # 1. Height-based optical distance calculation
            prior_h = OBJECT_HEIGHT_PRIORS.get(label, DEFAULT_HEIGHT_PRIOR)
            pixel_h = max(4.0, det["pixel_height"])
            dist_height = (focal_y * prior_h) / pixel_h

            # 2. Ground-contact lower edge perspective heuristic
            # Bottom edge y2 in [0, 1]. Lower edge near bottom of frame implies proximity to user.
            norm_y2 = max(0.1, min(1.0, bbox.y2))
            dist_ground = self.camera_height_m / max(0.15, math.tan(0.1 + (norm_y2 * (self.v_fov_rad / 1.5))))

            # Blend height prior and ground contact for stability
            estimated_dist = 0.65 * dist_height + 0.35 * dist_ground
            estimated_dist = round(max(0.4, min(settings.DEPTH_CLIP_METERS, estimated_dist)), 2)

            # 3. Horizontal angular offset from camera center
            # Center x in [0.0, 1.0] -> offset in [-1.0, 1.0]
            norm_cx = (bbox.center_x - 0.5) * 2.0
            angle_rad = math.atan(norm_cx * math.tan(self.h_fov_rad / 2.0))
            angle_deg = round(math.degrees(angle_rad), 1)

            # 4. Direction classification
            if angle_deg < -28.0:
                direction = "far_left"
            elif angle_deg < -12.0:
                direction = "left"
            elif angle_deg < -4.0:
                direction = "slightly_left"
            elif angle_deg <= 4.0:
                direction = "center"
            elif angle_deg <= 12.0:
                direction = "slightly_right"
            elif angle_deg <= 28.0:
                direction = "right"
            else:
                direction = "far_right"

            # 5. Direct walking path corridor test
            is_in_path = (abs(angle_deg) <= settings.PATH_CORRIDOR_ANGLE_DEG) and (estimated_dist <= 3.8)

            processed_objects.append(DetectedObject(
                label=label,
                confidence=conf,
                bbox=bbox,
                distance_m=estimated_dist,
                horizontal_angle_deg=angle_deg,
                direction=direction,
                is_in_path=is_in_path
            ))

        latency_ms = (time.time() - t0) * 1000.0
        return processed_objects, latency_ms
