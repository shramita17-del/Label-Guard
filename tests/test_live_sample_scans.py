import os
import unittest
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import CheckStatus

class TestLiveSampleScans(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.samples_dir = os.path.join(os.getcwd(), "frontend", "samples")

    def test_scan_sample_general_package(self):
        img_path = os.path.join(self.samples_dir, "sample_general.jpg")
        if not os.path.exists(img_path):
            self.skipTest("sample_general.jpg not found on disk")

        with open(img_path, "rb") as f:
            res = self.client.post("/scan", files={"file": ("sample_general.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("scan_id", data)
        self.assertIn("category", data)
        self.assertIn("overall_verdict", data)
        self.assertIn("check_results", data)
        self.assertGreater(len(data["check_results"]), 0)
        self.assertIn("authenticity", data)
        self.assertIsNotNone(data["authenticity"])
        # Visual annotation & report url
        self.assertIsNotNone(data["annotated_image_base64"])
        self.assertTrue(data["annotated_image_base64"].startswith("data:image/jpeg;base64"))
        self.assertIsNotNone(data["report_url"])

    def test_scan_sample_food(self):
        img_path = os.path.join(self.samples_dir, "sample_food.jpg")
        if not os.path.exists(img_path):
            self.skipTest("sample_food.jpg not found on disk")

        with open(img_path, "rb") as f:
            res = self.client.post("/scan", files={"file": ("sample_food.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("scan_id", data)
        self.assertIn("check_results", data)
        self.assertIsNotNone(data["authenticity"])

    def test_scan_sample_textile(self):
        img_path = os.path.join(self.samples_dir, "sample_textile.jpg")
        if not os.path.exists(img_path):
            self.skipTest("sample_textile.jpg not found on disk")

        with open(img_path, "rb") as f:
            res = self.client.post("/scan", files={"file": ("sample_textile.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("scan_id", data)
        self.assertIn("check_results", data)
        self.assertIsNotNone(data["authenticity"])

    def test_scan_sample_ai_package_triggers_ai_detection_and_fail(self):
        img_path = os.path.join(self.samples_dir, "sample_ai_package.png")
        if not os.path.exists(img_path):
            self.skipTest("sample_ai_package.png not found on disk")

        with open(img_path, "rb") as f:
            res = self.client.post("/scan", files={"file": ("sample_ai_package.png", f, "image/png")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("authenticity", data)
        auth = data["authenticity"]
        self.assertIsNotNone(auth)
        # Teammate's AI detection must flag the AI package as synthetic
        self.assertTrue(auth["is_ai_generated"])
        self.assertEqual(data["overall_verdict"], "FAIL")
        # Ensure AUTH_01 check is present in check_results
        auth_check = next((c for c in data["check_results"] if c["check_id"] == "AUTH_01"), None)
        self.assertIsNotNone(auth_check)
        self.assertEqual(auth_check["status"], "FAIL")

if __name__ == '__main__':
    unittest.main()
