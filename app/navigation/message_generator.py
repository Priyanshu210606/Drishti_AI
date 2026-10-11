from typing import Optional, Tuple
from app.config import settings
from app.models.schemas import DetectedObject, NavigationMessage

DIRECTION_PHRASES = {
    "center": "directly ahead",
    "slightly_right": "slightly to your right",
    "right": "to your right",
    "far_right": "far to your right",
    "slightly_left": "slightly to your left",
    "left": "to your left",
    "far_left": "far to your left"
}

class NavigationMessageGenerator:
    """
    Translates raw perception and path states into concise, natural,
    actionable spoken navigation instructions.
    """
    def generate(self,
                 path_status: str,
                 primary: Optional[DetectedObject],
                 mode: str = "normal",
                 total_obstacles: int = 0) -> Tuple[str, str]:
        """
        Generates spoken text and priority level (CRITICAL, HIGH, MEDIUM, LOW, INFO).
        """
        # Critical collision threat
        if primary and primary.is_in_path and primary.distance_m <= settings.CRITICAL_DISTANCE_METERS:
            label = primary.label.capitalize()
            if mode == "brief":
                text = f"Stop. {label} close."
            elif mode == "detailed":
                text = f"Stop immediately. {label} directly ahead, {primary.distance_m} meters away."
            else:
                text = f"Stop. {label} directly ahead, very close."
            return text, "CRITICAL"

        # Clear path scenario
        if path_status == "clear":
            if not primary:
                text = "Path clear ahead." if mode != "brief" else "Path clear."
                return text, "INFO"
            else:
                dir_phrase = DIRECTION_PHRASES.get(primary.direction, "to your side")
                if mode == "brief":
                    text = f"{primary.label.capitalize()}, {primary.distance_m}m {primary.direction}. Path clear."
                elif mode == "detailed":
                    text = f"{primary.label.capitalize()} detected {primary.distance_m} meters away {dir_phrase}. Path ahead is clear."
                else:
                    text = f"{primary.label.capitalize()} {dir_phrase}, about {primary.distance_m} meters away. Path clear."
                return text, "LOW"

        # Blocked or partially blocked path scenario
        if primary:
            dir_phrase = DIRECTION_PHRASES.get(primary.direction, "ahead")
            label = primary.label.capitalize()

            if path_status == "blocked":
                priority = "HIGH"
                if mode == "brief":
                    text = f"Blocked. {label}, {primary.distance_m}m."
                elif mode == "detailed":
                    text = f"Path blocked. {label} ahead at {primary.distance_m} meters, {abs(primary.horizontal_angle_deg):.0f} degrees {primary.direction}."
                else:
                    text = f"Obstacle ahead. {label} {dir_phrase}, {primary.distance_m} meters away."
            else:
                priority = "MEDIUM"
                multi_hint = "Multiple obstacles. " if total_obstacles > 2 else ""
                if mode == "brief":
                    text = f"{multi_hint}{label}, {primary.distance_m}m {primary.direction}."
                elif mode == "detailed":
                    text = f"{multi_hint}{label} detected approximately {primary.distance_m} meters away {dir_phrase}."
                else:
                    text = f"{multi_hint}{label} ahead, about {primary.distance_m} meters away, {dir_phrase}."
            return text, priority

        # Uncertain path
        if path_status == "uncertain":
            text = "Scene uncertain, please proceed carefully."
            return text, "MEDIUM"

        return "Path clear ahead.", "INFO"
