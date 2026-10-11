from typing import List, Tuple, Optional
from app.config import settings
from app.models.schemas import DetectedObject, GatingDecision

class SpatialReasoningEngine:
    """
    Evaluates pedestrian corridor safety and determines the primary obstacle
    and path clearance state (clear, partially_blocked, blocked, uncertain).
    """
    def evaluate_path(self,
                      objects: List[DetectedObject],
                      gating: GatingDecision) -> Tuple[str, Optional[DetectedObject]]:
        """
        Ranks detected obstacles by collision risk and calculates path clearance.
        """
        if gating.is_uncertain and not objects:
            return "uncertain", None

        if not objects:
            return "clear", None

        # Filter objects within walking corridor
        path_obstacles = [o for o in objects if o.is_in_path]

        # Prioritize objects by threat score:
        # Threat = (1 / distance) * (2.0 if in_path else 1.0) * confidence
        def threat_score(obj: DetectedObject) -> float:
            base = (1.0 / max(0.3, obj.distance_m)) * obj.confidence
            if obj.is_in_path:
                base *= 2.5
            if obj.label in ("person", "car", "truck", "motorcycle", "bicycle"):
                base *= 1.3
            return base

        sorted_objects = sorted(objects, key=threat_score, reverse=True)
        primary_obstacle = sorted_objects[0]

        # Determine path clearance
        if path_obstacles:
            closest_path_dist = min(o.distance_m for o in path_obstacles)
            if closest_path_dist <= settings.CRITICAL_DISTANCE_METERS:
                status = "blocked"
            elif closest_path_dist <= settings.CAUTION_DISTANCE_METERS or len(path_obstacles) > 1:
                status = "partially_blocked"
            else:
                status = "partially_blocked"
        else:
            # Obstacles exist, but they are off to the sides
            closest_dist = min(o.distance_m for o in objects)
            if closest_dist <= settings.CRITICAL_DISTANCE_METERS:
                status = "partially_blocked"
            else:
                status = "clear"

        if gating.is_uncertain and status == "clear":
            status = "uncertain"

        return status, primary_obstacle
