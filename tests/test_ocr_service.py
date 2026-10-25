import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock only the heavy ML dependencies that aren't installed in test env
# NEVER mock cv2 or numpy globally — it poisons other tests in the same process
sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('paddleocr', MagicMock())

import numpy as np
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

    @patch('backend.ocr_service.extractor.cv2.imread')
    def test_paddle_extractor(self, mock_imread):
        extractor = OCRExtractor()
        
        # Use a real numpy array so crop slicing and cv2 preprocessing work correctly
        mock_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
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
        self.assertEqual(results[0].raw_text, "Rs. 199.00\nincl of all taxes")
        self.assertEqual(results[0].ocr_confidence, 0.95) # Average of 0.98 and 0.92
        self.assertEqual(results[0].bbox.x_min, 10)

if __name__ == '__main__':
    unittest.main()

