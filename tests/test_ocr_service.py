import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy ML dependencies so tests run instantly without installing PyTorch/Paddle
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['paddleocr'] = MagicMock()

from backend.schemas import DetectedRegion, BoundingBox, OCRResult
from backend.ocr_service.detector import YoloDetector
from backend.ocr_service.extractor import OCRExtractor

class TensorMock:
    def __init__(self, data):
        self.data = data
    def tolist(self):
        return self.data
    def item(self):
        return self.data[0] if isinstance(self.data, list) else self.data

class TestOCRService(unittest.TestCase):
    
    def test_yolo_detector(self):
        # Setup Mock Model
        detector = YoloDetector("dummy.pt")
        
        # Mocking the output of YOLO inference to act like PyTorch tensors
        mock_box = MagicMock()
        mock_box.xyxy = [TensorMock([10, 20, 100, 200])]
        mock_box.conf = [TensorMock(0.95)]
        mock_box.cls = [TensorMock(0)]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        detector.model.return_value = [mock_result]
        detector.model.names = {0: "mrp_region"}
        
        # Run function
        regions = detector.detect_regions("dummy_img.jpg")
        
        # Verify strict alignment with schemas.py
        self.assertEqual(len(regions), 1)
        self.assertIsInstance(regions[0], DetectedRegion)
        self.assertEqual(regions[0].region_label, "mrp_region")
        self.assertEqual(regions[0].yolo_confidence, 0.95)
        self.assertEqual(regions[0].bbox.x_min, 10)
        self.assertEqual(regions[0].bbox.y_max, 200)

    def test_paddle_extractor(self):
        extractor = OCRExtractor()
        
        # Mock image dimensions to prevent out-of-bounds crop errors in logic
        mock_img = MagicMock()
        mock_img.shape = (1000, 1000, 3)
        sys.modules['cv2'].imread.return_value = mock_img
        
        # Mock PaddleOCR return format: [[ [box], ('text', conf) ]]
        extractor.ocr.ocr.return_value = [[
            (None, ('Rs. 199.00', 0.98)),
            (None, ('incl of all taxes', 0.92))
        ]]
        
        input_region = DetectedRegion(
            region_label="mrp_region", 
            bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            yolo_confidence=0.95
        )
        
        # Run function
        results = extractor.extract_text("dummy_img.jpg", [input_region])
        
        # Verify strict alignment with schemas.py
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], OCRResult)
        self.assertEqual(results[0].region_label, "mrp_region")
        self.assertEqual(results[0].raw_text, "Rs. 199.00 incl of all taxes")
        self.assertEqual(results[0].ocr_confidence, 0.95) # Average of 0.98 and 0.92
        self.assertEqual(results[0].bbox.x_min, 10)

if __name__ == '__main__':
    unittest.main()
