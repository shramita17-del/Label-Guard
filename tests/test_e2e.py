import sys
import unittest
import os
from unittest.mock import patch, MagicMock

# Mock Heavy ML before imports to keep E2E tests incredibly fast
mock_cv2 = MagicMock()
mock_cv2.imencode.return_value = (True, b"fake_base64_data")
sys.modules['cv2'] = mock_cv2
sys.modules['numpy'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['paddleocr'] = MagicMock()

from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import BoundingBox, DetectedRegion, OCRResult

client = TestClient(app)

class TestE2EPipeline(unittest.TestCase):
    @patch('backend.main.get_ml')
    def test_full_flow_a_and_e(self, mock_get_ml):
        # 1. Setup mock detector & extractor (simulating ML outputs)
        mock_detector = MagicMock()
        mock_extractor = MagicMock()
        
        # Region Detection mock outputs
        mock_detector.detect_regions.return_value = [
            DetectedRegion(region_label="mrp_region", bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10), yolo_confidence=0.99),
            DetectedRegion(region_label="fssai_region", bbox=BoundingBox(x_min=20, y_min=20, x_max=30, y_max=30), yolo_confidence=0.99)
        ]
        
        # OCR Text mock outputs
        mock_extractor.extract_text.return_value = [
            OCRResult(region_label="mrp_region", raw_text="Rs. 500 incl tax", ocr_confidence=0.95, bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)),
            OCRResult(region_label="fssai_region", raw_text="10012011000168", ocr_confidence=0.95, bbox=BoundingBox(x_min=20, y_min=20, x_max=30, y_max=30))
        ]
        
        mock_get_ml.return_value = (mock_detector, mock_extractor)
        
        # Create a dummy payload file
        dummy_img_path = "dummy_e2e_test.jpg"
        with open(dummy_img_path, "wb") as f:
            f.write(b"fake_image_data")
            
        # 2. Trigger Flow A (Full Scan) via FastAPI route
        with open(dummy_img_path, "rb") as f:
            response = client.post("/scan", files={"file": ("dummy_e2e_test.jpg", f, "image/jpeg")})
            
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verification: Flow E (Auto-Routing) correctly sensed FSSAI -> FOOD_BEVERAGE
        self.assertEqual(data["category"], "food_beverage")
        
        # Verification: Rule Engine executed multiple checks
        self.assertGreater(len(data["check_results"]), 0)
        
        # Verification: Rule F14 (Tier-3 FSSAI Cache Fallback) properly resolved the FSSAI License
        fssai_check = next(c for c in data["check_results"] if c["check_name"] == "FSSAI License No.")
        self.assertEqual(fssai_check["status"], "PASS")
        self.assertIn("Tier 3 (Cache)", fssai_check["reason"])

        # Verification: Tier 1 MRP Check parsed successfully
        mrp_check = next(c for c in data["check_results"] if c["check_name"] == "MRP")
        self.assertEqual(mrp_check["status"], "PASS")
        self.assertEqual(mrp_check["evidence_text"], "Rs. 500 incl tax")

        # Verification: Flow 3 (Reporting) added the base64 annotation
        self.assertIsNotNone(data["annotated_image_base64"])
        self.assertTrue(data["annotated_image_base64"].startswith("data:image/jpeg;base64"))
        
        # Cleanup
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)

if __name__ == "__main__":
    unittest.main()
