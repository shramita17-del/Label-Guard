import cv2
import numpy as np
from typing import List
from backend.schemas import DetectedRegion, OCRResult, BoundingBox

class OCRExtractor:
    def __init__(self):
        """
        Initializes PaddleOCR with backward-compatible parameter handling across versions.
        """
        self.ocr = None
        try:
            from paddleocr import PaddleOCR
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            except TypeError:
                self.ocr = PaddleOCR(lang='en')
            except Exception:
                self.ocr = PaddleOCR()
        except Exception as e:
            print(f"PaddleOCR load notice: {e}")
        
    def extract_text(self, image_path: str, regions: List[DetectedRegion]) -> List[OCRResult]:
        """
        Crops the image based on detected regions and extracts text & confidence.
        If no regions are provided, runs OCR across the full image.
        """
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((800, 800, 3), dtype=np.uint8)
            
        h, w = img.shape[:2]
        
        # If YOLO returned no specific bounding boxes, OCR the full image
        if not regions:
            regions = [
                DetectedRegion(
                    region_label="full_package",
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=w, y_max=h),
                    yolo_confidence=1.0
                )
            ]
            
        ocr_results = []
        
        for region in regions:
            b = region.bbox
            
            # Ensure valid bounds
            y_min, y_max = max(0, b.y_min), min(h, b.y_max)
            x_min, x_max = max(0, b.x_min), min(w, b.x_max)
            
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
                
            combined_text = []
            confidences = []
            
            if self.ocr is not None:
                try:
                    # Support PaddleOCR prediction calls
                    try:
                        result = self.ocr.ocr(crop, cls=True)
                    except TypeError:
                        result = self.ocr.ocr(crop)
                    except Exception:
                        result = None
                        
                    if result:
                        # Handle PaddleOCR 2.x standard format: [[ [box], (text, conf) ]]
                        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                            for line in result[0]:
                                if isinstance(line, (list, tuple)) and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                    text, conf = line[1][0], float(line[1][1])
                                    combined_text.append(str(text))
                                    confidences.append(conf)
                        # Handle PaddleOCR 3.x / PaddleX dict/object format
                        elif isinstance(result, list):
                            for item in result:
                                if isinstance(item, dict):
                                    texts = item.get("rec_texts", item.get("text", []))
                                    scores = item.get("rec_scores", item.get("score", []))
                                    if isinstance(texts, list):
                                        combined_text.extend([str(t) for t in texts])
                                    if isinstance(scores, list):
                                        confidences.extend([float(s) for s in scores])
                except Exception as err:
                    print(f"OCR region extraction notice: {err}")
                    
            if combined_text:
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.90
                ocr_results.append(
                    OCRResult(
                        region_label=region.region_label,
                        raw_text=" ".join(combined_text),
                        ocr_confidence=round(avg_conf, 3),
                        bbox=region.bbox
                    )
                )
            else:
                # If no text extracted in this specific sub-region, create fallback entry
                ocr_results.append(
                    OCRResult(
                        region_label=region.region_label,
                        raw_text="",
                        ocr_confidence=0.0,
                        bbox=region.bbox
                    )
                )
                
        return ocr_results
