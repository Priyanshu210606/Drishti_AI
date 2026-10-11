import time
from typing import List, Tuple
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torchvision.models.detection as detection_models
from app.config import settings
from app.models.schemas import BoundingBox

# Standard COCO 91 classes mapped to clean, spoken navigation labels
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "street sign", 13: "stop sign", 14: "parking meter",
    15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

class ObjectDetector:
    """
    Lightweight, real-time object detector utilizing Torchvision SSDLite MobileNetV3.
    Extracts obstacle classes, bounding boxes, and confidence scores.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.model = detection_models.ssdlite320_mobilenet_v3_large(
            weights=detection_models.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> Tuple[List[dict], float]:
        """
        Runs object detection on a PIL Image.
        Returns a list of raw detections and execution latency in milliseconds.
        """
        t0 = time.time()
        w, h = image.size

        # Convert to float tensor normalized [0.0, 1.0]
        img_tensor = TF.to_tensor(image).to(self.device).unsqueeze(0)

        outputs = self.model(img_tensor)[0]
        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()

        detections = []
        for box, score, label_id in zip(boxes, scores, labels):
            conf = float(score.item())
            if conf < settings.DETECTION_CONF_THRESHOLD:
                continue

            lid = int(label_id.item())
            class_name = COCO_LABELS.get(lid, "obstacle")

            # Normalize bounding boxes to [0.0, 1.0]
            x1 = max(0.0, min(1.0, float(box[0].item()) / w))
            y1 = max(0.0, min(1.0, float(box[1].item()) / h))
            x2 = max(0.0, min(1.0, float(box[2].item()) / w))
            y2 = max(0.0, min(1.0, float(box[3].item()) / h))

            detections.append({
                "label": class_name,
                "confidence": conf,
                "bbox": BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                "pixel_height": float(box[3].item() - box[1].item()),
                "pixel_width": float(box[2].item() - box[0].item())
            })

        latency_ms = (time.time() - t0) * 1000.0
        return detections, latency_ms
