import cv2
import numpy as np
from typing import List
from backend.schemas import DetectedRegion, OCRResult, BoundingBox

class OCRExtractor:
    def __init__(self):
        """
        Initializes PaddleOCR. 
        use_angle_cls=True helps with rotated packaging text.
        """
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
    def extract_text(self, image_path: str, regions: List[DetectedRegion]) -> List[OCRResult]:
        """
        Crops the image based on detected regions and extracts text & confidence.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        ocr_results = []
        for region in regions:
            b = region.bbox
            
            # Ensure valid bounds
            h, w = img.shape[:2]
            y_min, y_max = max(0, b.y_min), min(h, b.y_max)
            x_min, x_max = max(0, b.x_min), min(w, b.x_max)
            
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
                
            # Run OCR on the cropped region
            result = self.ocr.ocr(crop, cls=True)
            
            if not result or not result[0]:
                continue
                
            combined_text = []
            confidences = []
            
            for line in result[0]:
                # PaddleOCR format: [box_coords, (text, confidence)]
                text, conf = line[1]
                combined_text.append(text)
                confidences.append(conf)
                
            if combined_text:
                avg_conf = sum(confidences) / len(confidences)
                ocr_results.append(
                    OCRResult(
                        region_label=region.region_label,
                        raw_text=" ".join(combined_text),
                        ocr_confidence=avg_conf,
                        bbox=region.bbox  # Pass through the original bbox for downstream reporting
                    )
                )
                
        return ocr_results
