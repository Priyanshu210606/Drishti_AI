import math
from collections import deque
from typing import List, Optional
import torch
import numpy as np
from app.config import settings
from app.models.schemas import DetectedObject

def wrap_angle_rad(delta_rad: float) -> float:
    """
    Computes wrapped angular difference in [-pi, pi].
    Eliminates artificial +-2*pi jumps caused by modular angle discontinuity.
    """
    return (delta_rad + math.pi) % (2.0 * math.pi) - math.pi

class TemporalSequenceBuffer:
    """
    Maintains a rolling causal buffer of multimodal scene representations.
    Constructs the exact 16-dimensional feature vector expected by Drishti Mamba and Transformer.
    """
    def __init__(self, seq_len: int = settings.SEQUENCE_LENGTH, feature_dim: int = settings.FEATURE_DIM):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=seq_len)
        self.prev_yaw_rad: float = 0.0
        self.prev_pitch_rad: float = 0.0
        self.prev_roll_rad: float = 0.0
        self.prev_head_rad: float = 0.0

    def add_observation(self,
                        detected_objects: List[DetectedObject],
                        estimated_yaw_deg: float = 0.0,
                        estimated_pitch_deg: float = 0.0,
                        estimated_roll_deg: float = 0.0,
                        estimated_heading_deg: float = 0.0):
        """
        Processes a single frame's scene summary and appends the 16-D feature vector.
        """
        # Convert degrees to radians
        yaw_rad = math.radians(estimated_yaw_deg)
        pitch_rad = math.radians(estimated_pitch_deg)
        roll_rad = math.radians(estimated_roll_deg)
        head_rad = math.radians(estimated_heading_deg)

        # Count obstacle categories
        total_obstacles = float(len(detected_objects))
        person_count = float(sum(1 for o in detected_objects if o.label == "person"))
        vehicle_count = float(sum(1 for o in detected_objects if o.label in ("car", "truck", "bus", "motorcycle", "bicycle")))

        # Distance metrics
        if detected_objects:
            min_dist = min(o.distance_m for o in detected_objects)
            med_dist = float(np.median([o.distance_m for o in detected_objects]))
        else:
            min_dist = settings.DEPTH_CLIP_METERS
            med_dist = settings.DEPTH_CLIP_METERS

        # Normalized feature vector (16 dimensions matching checkpoint trained format)
        features = [
            math.sin(yaw_rad),
            math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad),
            math.sin(roll_rad),
            math.cos(roll_rad),
            math.sin(head_rad),
            math.cos(head_rad),
            0.0,  # relative lat delta
            0.0,  # relative lon delta
            0.0,  # relative alt delta
            total_obstacles,
            person_count,
            vehicle_count,
            min(min_dist, settings.DEPTH_CLIP_METERS) / settings.DEPTH_CLIP_METERS,
            min(med_dist, settings.DEPTH_CLIP_METERS) / settings.DEPTH_CLIP_METERS
        ]

        self.buffer.append(features)
        self.prev_yaw_rad = yaw_rad
        self.prev_pitch_rad = pitch_rad
        self.prev_roll_rad = roll_rad
        self.prev_head_rad = head_rad

    def get_sequence_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Returns a (1, sequence_length, feature_dim) float32 tensor.
        Pads causally by replicating earliest frame if buffer is not full.
        """
        if not self.buffer:
            # Baseline zero observation if buffer empty
            zero_vec = [0.0] * self.feature_dim
            zero_vec[1] = 1.0  # cos(0) = 1
            zero_vec[3] = 1.0
            zero_vec[5] = 1.0
            zero_vec[7] = 1.0
            zero_vec[14] = 1.0
            zero_vec[15] = 1.0
            seq = [zero_vec] * self.seq_len
        else:
            current_list = list(self.buffer)
            # Replicate earliest entry to fill window up to seq_len
            if len(current_list) < self.seq_len:
                pad_count = self.seq_len - len(current_list)
                seq = [current_list[0]] * pad_count + current_list
            else:
                seq = current_list

        tensor = torch.tensor([seq], dtype=torch.float32, device=device)
        return tensor

    def clear(self):
        self.buffer.clear()
