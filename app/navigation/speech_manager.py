import time
from typing import Optional
from app.config import settings
from app.models.schemas import DetectedObject, NavigationMessage

class SpeechManager:
    """
    Stateful speech manager that eliminates repetitive spam.
    Decides when speech is genuinely necessary based on priority,
    distance variations, and path status shifts.
    """
    def __init__(self,
                 min_interval_sec: float = settings.MIN_SPEECH_INTERVAL_SEC,
                 dist_change_thresh_m: float = settings.DISTANCE_CHANGE_THRESHOLD_M,
                 clear_reminder_sec: float = settings.CLEAR_PATH_REMINDER_SEC):
        self.min_interval_sec = min_interval_sec
        self.dist_change_thresh_m = dist_change_thresh_m
        self.clear_reminder_sec = clear_reminder_sec

        self.last_speech_time: float = 0.0
        self.last_spoken_text: str = ""
        self.last_primary_label: Optional[str] = None
        self.last_primary_dist: float = 0.0
        self.last_primary_dir: Optional[str] = None
        self.last_path_status: str = "clear"

    def should_speak(self,
                     text: str,
                     priority: str,
                     path_status: str,
                     primary: Optional[DetectedObject]) -> bool:
        """
        Evaluates whether the candidate message warrants speech emission.
        """
        now = time.time()
        elapsed = now - self.last_speech_time

        # 1. Critical safety warnings always speak immediately
        if priority == "CRITICAL":
            # Avoid repeating the identical critical sentence faster than 1 second
            if elapsed < 1.0 and text == self.last_spoken_text:
                return False
            self._record_speech(now, text, path_status, primary)
            return True

        # 2. Path status change (e.g., clear -> blocked, or blocked -> clear)
        if path_status != self.last_path_status and elapsed >= 1.5:
            self._record_speech(now, text, path_status, primary)
            return True

        # 3. Primary obstacle changed
        if primary:
            label_changed = (primary.label != self.last_primary_label)
            dir_changed = (primary.direction != self.last_primary_dir)
            dist_dropped = (self.last_primary_dist - primary.distance_m) >= self.dist_change_thresh_m

            if (label_changed or dir_changed or dist_dropped) and (elapsed >= self.min_interval_sec):
                self._record_speech(now, text, path_status, primary)
                return True

        # 4. Periodic reassurance for clear path
        if path_status == "clear" and elapsed >= self.clear_reminder_sec:
            self._record_speech(now, text, path_status, primary)
            return True

        return False

    def _record_speech(self, timestamp: float, text: str, path_status: str, primary: Optional[DetectedObject]):
        self.last_speech_time = timestamp
        self.last_spoken_text = text
        self.last_path_status = path_status
        if primary:
            self.last_primary_label = primary.label
            self.last_primary_dist = primary.distance_m
            self.last_primary_dir = primary.direction
        else:
            self.last_primary_label = None
            self.last_primary_dist = 0.0
            self.last_primary_dir = None

    def reset(self):
        self.last_speech_time = 0.0
        self.last_spoken_text = ""
        self.last_primary_label = None
        self.last_primary_dist = 0.0
        self.last_primary_dir = None
        self.last_path_status = "clear"
