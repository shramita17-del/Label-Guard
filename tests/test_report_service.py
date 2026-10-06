import os
import unittest
from datetime import datetime
from backend.schemas import ComplianceReport, CheckResult, CheckStatus, Severity, ProductCategory, BoundingBox
from backend.report_service.annotator import annotate_image
from backend.report_service.pdf_generator import generate_pdf_report

class TestReportService(unittest.TestCase):
    def setUp(self):
        self.mock_report = ComplianceReport(
            scan_id="SCAN_12345",
            category=ProductCategory.FOOD_BEVERAGE,
            overall_verdict=CheckStatus.REVIEW,
            check_results=[
                CheckResult(
                    check_id="F07", check_name="MRP", rule_ref="Rule_6_1_e",
                    severity=Severity.BLOCKING, status=CheckStatus.PASS,
                    evidence_text="Rs. 199 incl tax",
                    evidence_bbox=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=50),
                    confidence=0.95, reason="Found correctly."
                ),
                CheckResult(
                    check_id="F14", check_name="FSSAI License No.", rule_ref="FSSAI_Licensing",
                    severity=Severity.BLOCKING, status=CheckStatus.FAIL,
                    evidence_text=None, evidence_bbox=None,
                    confidence=None, reason="License missing entirely."
                )
            ],
            annotated_image_base64=None,
            generated_at=datetime.utcnow().isoformat()
        )
        self.template_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        self.output_dir = os.path.join(self.template_dir, "generated")

    def test_annotator(self):
        # We don't have a real image, the annotator will fallback to a dummy image
        b64 = annotate_image("dummy.jpg", self.mock_report)
        self.assertTrue(b64.startswith("data:image/jpeg;base64,"))
        self.mock_report.annotated_image_base64 = b64

    def test_pdf_generation_fallback(self):
        # Attach a dummy base64 to ensure template rendering works
        self.mock_report.annotated_image_base64 = "data:image/jpeg;base64,dummy"
        output_path = os.path.join(self.output_dir, "test_report.pdf")
        
        actual_path = generate_pdf_report(self.mock_report, self.template_dir, output_path)
        
        # It might fallback to .html if WeasyPrint isn't installed
        self.assertTrue(os.path.exists(actual_path))
        
        # Cleanup
        if os.path.exists(actual_path):
            os.remove(actual_path)

if __name__ == '__main__':
    unittest.main()
