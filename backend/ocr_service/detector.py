import cv2
import numpy as np
from typing import List
from backend.schemas import DetectedRegion, BoundingBox

class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLOv8n detector. 
        Will download the pretrained weights automatically if not present.
        """
        from ultralytics import YOLO
        self.model = YOLO(model_path)
    
    def detect_regions(self, image_path: str) -> List[DetectedRegion]:
        """
        Runs region detection and returns a list of bounding boxes with labels.
        """
        results = self.model(image_path)
        detected_regions = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # xyxy format: [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = self.model.names.get(cls, "unknown")
                
                detected_regions.append(
                    DetectedRegion(
                        region_label=label,
                        bbox=BoundingBox(
                            x_min=int(x1), 
                            y_min=int(y1), 
                            x_max=int(x2), 
                            y_max=int(y2)
                        ),
                        yolo_confidence=float(conf)
                    )
                )
        return detected_regions
